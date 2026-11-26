"""Public PyTorch API. Transform normalization is orthonormal throughout."""
import importlib.util
import torch
from ._torch import _axis, _power, _work, _apply, _native_gate


def _backend(backend, x, w=None):
    if backend not in {"auto", "torch", "triton", "cuda"}:
        raise ValueError(f"unknown backend {backend!r}")
    # Explicit opt-in to unvalidated GPU kernels; auto is safe and portable.
    if backend in {"auto", "torch"}:
        return "torch"
    if not x.is_cuda or torch.version.hip is not None:
        raise RuntimeError(f"{backend} requires an NVIDIA CUDA tensor")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("accelerated backend requires float32, float16 or bfloat16")
    if x.shape[-1] > 1024:
        raise ValueError("accelerated backend currently supports lengths <= 1024")
    if w is not None and (w.ndim != 1 or w.shape[0] != x.shape[-1] or w.dtype != x.dtype):
        raise ValueError("accelerated gate requires a same-dtype vector of transform length")
    if backend == "triton" and importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is not installed; install the triton extra")
    return backend


def _launch(x, w, backend):
    if backend == "triton":
        from ._triton import launch
    else:
        from ._cuda import launch
    return launch(x, w)


class _Hadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, backend):
        ctx.backend = backend
        return _launch(x, None, backend)

    @staticmethod
    def backward(ctx, grad):
        # H is symmetric and orthonormal. Recursive autograd also supports double backward.
        return _Hadamard.apply(grad.contiguous(), ctx.backend), None


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, backend):
        ctx.save_for_backward(x, w)
        ctx.backend = backend
        return _launch(x, w, backend)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _Gate.apply(grad.contiguous(), w, ctx.backend)
        if ctx.needs_input_grad[1]:
            # Compute the weight gradient with reference ops in fp32 for low precision.
            a = _apply(_work(x), "hadamard")
            b = _apply(_work(grad), "hadamard")
            product = a*b
            dw = product.reshape(-1, w.numel()).sum(0).to(w.dtype)
        return dx, dw, None


def hadamard(x, dim=-1, *, backend="auto"):
    """Orthonormal Sylvester WHT, natural order. Its own inverse."""
    dim = _axis(x, dim)
    z = x.movedim(dim, -1)
    _power(z.shape[-1])
    b = _backend(backend, z)
    out = _apply(_work(z), "hadamard").to(x.dtype) if b == "torch" else _Hadamard.apply(z.contiguous(), b)
    return out.movedim(-1, dim)


def _transform(x, dim, name, inverse=False, levels=None):
    dim = _axis(x, dim)
    return _apply(_work(x.movedim(dim, -1)), name, inverse, levels).to(x.dtype).movedim(-1, dim)


def dct(x, dim=-1):
    """Orthonormal DCT-II; implemented using an even extension and FFT."""
    return _transform(x, dim, "dct")


def idct(x, dim=-1):
    """Inverse orthonormal DCT-II (DCT-III convention)."""
    return _transform(x, dim, "dct", True)


def haar(x, dim=-1, *, levels=None):
    """Packed Haar coefficients [approximation, coarsest details, ..., finest details]."""
    return _transform(x, dim, "haar", levels=levels)


def ihaar(x, dim=-1, *, levels=None):
    return _transform(x, dim, "haar", True, levels)


def _2d(x, dims, fn, **kwargs):
    a, b = dims
    a, b = _axis(x, a), _axis(x, b)
    if a == b:
        raise ValueError("2D transform axes must be distinct")
    return fn(fn(x, dim=a, **kwargs), dim=b, **kwargs)


def haar2d(x, dims=(-2,-1), *, levels=None):
    """Separable packed Haar transform over two axes."""
    return _2d(x, dims, haar, levels=levels)


def ihaar2d(x, dims=(-2,-1), *, levels=None):
    return _2d(x, dims, ihaar, levels=levels)


def dct2d(x, dims=(-2,-1)):
    return _2d(x, dims, dct)


def idct2d(x, dims=(-2,-1)):
    return _2d(x, dims, idct)


def spectral_gate(x, weights, *, transform="hadamard", dim=-1, levels=None, backend="auto"):
    """Compute T^-1(weights * T(x)). A vector weights the selected axis.

    Higher-rank weights broadcast against x's original shape without expanding it.
    Only Hadamard supports explicit CUDA/Triton backends, currently vector weights only.
    """
    dim = _axis(x, dim)
    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        raise TypeError("weights must be a real floating-point torch.Tensor")
    if x.device != weights.device:
        raise ValueError("x and weights must be on the same device")
    if transform not in {"hadamard", "dct", "haar"}:
        raise ValueError(f"unknown transform {transform!r}")
    if levels is not None and transform != "haar":
        raise ValueError("levels applies only to Haar")
    z = x.movedim(dim, -1)
    if transform in {"hadamard", "haar"}:
        _power(z.shape[-1])
    if weights.ndim == 1:
        if weights.shape[0] != z.shape[-1]:
            raise ValueError("vector weights must match the transform length")
        w = weights
    elif weights.ndim == 0:
        w = weights
    else:
        if torch.broadcast_shapes(x.shape, weights.shape) != x.shape:
            raise ValueError("weights must broadcast without expanding x")
        w = weights.reshape((1,)*(x.ndim-weights.ndim) + weights.shape).movedim(dim, -1)
    if transform != "hadamard" and backend not in {"auto", "torch"}:
        raise ValueError("accelerated gate is implemented only for Hadamard")
    b = _backend(backend, z, w)
    out = _native_gate(z, w, transform, levels) if b == "torch" else _Gate.apply(z.contiguous(), w.contiguous(), b)
    return out.movedim(-1, dim)
