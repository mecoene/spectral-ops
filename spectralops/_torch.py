"""Reference implementations; all arithmetic is expressed in PyTorch ops."""
import math
import torch


def _axis(x, dim):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("expected a real floating-point torch.Tensor")
    if not isinstance(dim, int) or not -x.ndim <= dim < x.ndim:
        raise ValueError(f"invalid axis {dim} for rank {x.ndim}")
    dim %= x.ndim
    if x.shape[dim] == 0:
        raise ValueError("transform axis must not be empty")
    return dim


def _power(n):
    if n <= 0 or n & (n - 1):
        raise ValueError("transform length must be a positive power of two")


def _work(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _fwht(x):
    n = x.shape[-1]
    _power(n)
    shape = x.shape
    h = 1
    while h < n:
        z = x.reshape(*shape[:-1], n // (2*h), 2, h)
        a, b = z.unbind(-2)
        x = torch.cat((a+b, a-b), dim=-1).reshape(shape)
        h *= 2
    return x * (1 / math.sqrt(n))


def _haar(x, levels, inverse=False):
    n = x.shape[-1]
    _power(n)
    max_levels = n.bit_length()-1
    if levels is None:
        levels = max_levels
    if not isinstance(levels, int) or not 0 <= levels <= max_levels:
        raise ValueError(f"levels must be between 0 and {max_levels}")
    scale = 1 / math.sqrt(2)
    if inverse:
        for step in reversed(range(levels)):
            width = n >> step
            a, b = x[..., :width//2], x[..., width//2:width]
            z = torch.stack(((a+b)*scale, (a-b)*scale), dim=-1).flatten(-2)
            x = torch.cat((z, x[..., width:]), dim=-1)
    else:
        for step in range(levels):
            width = n >> step
            a, b = x[..., :width:2], x[..., 1:width:2]
            x = torch.cat(((a+b)*scale, (a-b)*scale, x[..., width:]), dim=-1)
    return x


def _dct(x, inverse=False):
    n = x.shape[-1]
    # MKL FFT cannot process an empty batch; the transform is vacuous there.
    if x.numel() == 0:
        return x.clone()
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    theta = k * (math.pi / (2*n))
    scale = torch.where(k == 0, 1/math.sqrt(n), math.sqrt(2/n))
    if inverse:
        # Construct via real/imag instead of differentiating abs/sign at zero.
        bins = torch.complex(2*x/scale*theta.cos(), 2*x/scale*theta.sin())
        bins = torch.cat((bins, torch.zeros_like(bins[..., :1])), dim=-1)
        return torch.fft.irfft(bins, n=2*n, dim=-1)[..., :n]
    bins = torch.fft.rfft(torch.cat((x, x.flip(-1)), dim=-1), dim=-1)[..., :n]
    return (bins.real*theta.cos() + bins.imag*theta.sin()) * (scale/2)


def _apply(x, transform, inverse=False, levels=None):
    if transform == "hadamard":
        return _fwht(x)
    if transform == "dct":
        return _dct(x, inverse)
    if transform == "haar":
        return _haar(x, levels, inverse)
    raise ValueError(f"unknown transform {transform!r}")


def _native_gate(x, w, transform, levels):
    dtype = torch.promote_types(x.dtype, w.dtype)
    x, w = _work(x.to(dtype)), _work(w.to(dtype))
    return _apply(_apply(x, transform, levels=levels)*w, transform,
                  inverse=True, levels=levels).to(dtype)
