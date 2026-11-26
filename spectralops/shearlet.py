"""Undecimated, real, band-limited cone-adapted discrete shearlet frame.

All FFTs act on the final two axes, with periodic boundary conditions.
Frequency masks use unshifted torch.fft ordering. Scales are coarse to fine.
This is a discrete Parseval-normalized construction, not ShearLab compatibility.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ShearletBand:
    scale: int
    cone: str
    shear: int
    shear_level: int


def _smooth(t):
    t = t.clamp(0, 1)
    return t**4 * (35 + t*(-84 + t*(70-20*t)))


def _low_power(r, cutoff):
    return torch.cos((math.pi/2)*_smooth(r/cutoff-1)).square()


def _window_power(t):
    return torch.cos((math.pi/2)*_smooth(t.abs())).square() * (t.abs() < 1)


def _make_filters(shape, scales, shear_levels, device, dtype):
    h, w = shape
    # Construction in double precision, followed by casting the normalized frame.
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(w, device=device, dtype=torch.float64)[None, :]
    ax, ay = fx.abs(), fy.abs()
    r = torch.maximum(ax, ay)
    horizontal = ax >= ay
    tiny = torch.finfo(torch.float64).tiny
    # Slopes are evaluated only in the appropriate cone; safe denominators at DC.
    u = torch.where(horizontal, fy / ax.clamp_min(tiny) * fx.sign(), 0.)
    v = torch.where(~horizontal, fx / ay.clamp_min(tiny) * fy.sign(), 0.)
    c0 = 0.5 / (2**scales)
    previous = _low_power(r, c0)
    powers = [previous]
    bands = [ShearletBand(-1, 'lowpass', 0, 0)]
    for j, ell in enumerate(shear_levels):
        q = 2**ell
        following = torch.ones_like(r) if j == scales-1 else _low_power(r, c0 * 2**(j+1))
        radial = (following - previous).clamp_min(0)
        previous = following
        # Interior shears on each cone; shared diagonal seam filters below.
        for cone, slope, support in [('horizontal', u, horizontal), ('vertical', v, ~horizontal)]:
            for k in range(-q+1, q):
                powers.append(radial * support * _window_power(q*slope-k))
                bands.append(ShearletBand(j, cone, k, ell))
        for k in (-q, q):
            powers.append(radial * torch.where(horizontal, _window_power(q*u-k), _window_power(q*v-k)))
            bands.append(ShearletBand(j, 'seam', k, ell))
    p = torch.stack(powers)
    # Enforce conjugate symmetry on the actual discrete FFT grid, especially Nyquist lines.
    iy = (-torch.arange(h, device=device)) % h
    ix = (-torch.arange(w, device=device)) % w
    p = (p + p.index_select(-2, iy).index_select(-1, ix)) * 0.5
    total = p.sum(0)
    if not bool((total > 0).all()):
        raise RuntimeError('uncovered frequency in shearlet frame')
    return (p / total).clamp_min(0).sqrt().to(dtype).contiguous(), tuple(bands)


def _validate_shape(shape, scales, shear_levels):
    if len(shape) != 2 or any(not isinstance(s, int) or s < 4 for s in shape):
        raise ValueError('shape must contain two integers >= 4')
    if not isinstance(scales, int) or not 1 <= scales <= int(math.log2(min(shape))):
        raise ValueError('scales must be between 1 and floor(log2(min(shape)))')
    if shear_levels is None:
        shear_levels = tuple((j+1)//2 for j in range(scales))
    else:
        shear_levels = tuple(shear_levels)
    if len(shear_levels) != scales or any(not isinstance(v, int) or not 0 <= v <= 6 for v in shear_levels):
        raise ValueError('shear_levels must contain one integer in [0,6] per scale')
    return tuple(shape), shear_levels


class ShearletSystem(nn.Module):
    """Fixed Parseval filter bank; move it to the input device with .to(device).

    Input: (..., H, W); coefficients: (..., K, H, W). Lowpass first, then
    increasing scale, horizontal/vertical interior shears and two diagonal seams.
    Filters are full-size periodic frequency responses, not compact spatial kernels.
    Use float32 or float64 filters even when inputs are fp16/bf16.
    """
    def __init__(self, shape, scales=3, *, shear_levels=None, device=None, dtype=torch.float32):
        super().__init__()
        self.shape, self.shear_levels = _validate_shape(shape, scales, shear_levels)
        self.scales = scales
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('filter dtype must be float32 or float64')
        filters, self.bands = _make_filters(self.shape, scales, self.shear_levels, device, dtype)
        self.register_buffer('filters', filters, persistent=False)

    @property
    def num_bands(self):
        return len(self.bands)

    def frame_weights(self):
        return self.filters.square().sum(0)

    def spatial_kernels(self, *, centered=True):
        """Full circular convolution impulse responses; ordinary FFT convention."""
        k = torch.fft.ifft2(self.filters).real
        return torch.fft.fftshift(k, dim=(-2, -1)) if centered else k

    def _input(self, x, coeff=False):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('expected real floating-point torch tensor')
        if x.ndim < (3 if coeff else 2) or tuple(x.shape[-2:]) != self.shape:
            raise ValueError(f'expected spatial dimensions {self.shape}')
        if coeff and x.shape[-3] != self.num_bands:
            raise ValueError('wrong number of shearlet bands')
        if x.device != self.filters.device:
            raise ValueError('move the system to the input device first')
        if self.filters.dtype not in (torch.float32, torch.float64):
            raise ValueError('keep shearlet filters in float32 or float64')
        dtype = torch.promote_types(x.dtype, self.filters.dtype)
        return x.to(dtype), self.filters.to(dtype)

    def analysis(self, x, *, chunk_size=None):
        z, f = self._input(x)
        if chunk_size is None: chunk_size = self.num_bands
        if not isinstance(chunk_size, int) or chunk_size <= 0: raise ValueError('invalid chunk_size')
        X = torch.fft.fft2(z).unsqueeze(-3)
        return torch.cat([torch.fft.ifft2(X * b).real for b in f.split(chunk_size)], dim=-3)

    def forward(self, x, *, chunk_size=None):
        return self.analysis(x, chunk_size=chunk_size)

    def synthesis(self, coeff, *, chunk_size=None):
        z, f = self._input(coeff, True)
        if chunk_size is None: chunk_size = self.num_bands
        if not isinstance(chunk_size, int) or chunk_size <= 0: raise ValueError('invalid chunk_size')
        # The adjoint is the inverse on the range of analysis (Parseval frame).
        accum = None
        for c, b in zip(z.split(chunk_size, dim=-3), f.split(chunk_size)):
            term = (torch.fft.fft2(c) * b).sum(-3)
            accum = term if accum is None else accum + term
        return torch.fft.ifft2(accum).real

    def gate(self, x, weights, *, backend='auto'):
        """Synthesis(weights * analysis(x)), without materializing coefficients.

        weights: (..., K), broadcastable over x's leading axes without expanding x.
        Explicit CUDA/Triton response kernels currently accept shared (K,) FP32 weights.
        FFTs use torch.fft (cuFFT on CUDA), not custom FFT kernels.
        """
        z, f = self._input(x)
        if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
            raise TypeError('weights must be a floating tensor')
        if weights.device != z.device or weights.ndim < 1 or weights.shape[-1] != self.num_bands:
            raise ValueError('weights must be on input device with last dimension num_bands')
        if torch.broadcast_shapes(z.shape[:-2], weights.shape[:-1]) != z.shape[:-2]:
            raise ValueError('weights must not expand input leading dimensions')
        dtype = torch.promote_types(z.dtype, weights.dtype)
        z, f, weights = z.to(dtype), f.to(dtype), weights.to(dtype)
        if backend in ('auto', 'torch'):
            response = torch.einsum('...k,khw->...hw', weights, f.square())
        elif backend in ('cuda', 'triton'):
            if not z.is_cuda or torch.version.hip is not None or dtype != torch.float32 or weights.ndim != 1:
                raise ValueError('custom response kernels need NVIDIA CUDA, FP32 computation, shared (K,) weights')
            from ._shearlet_gpu import response_gpu
            response = response_gpu(f.contiguous(), weights.contiguous(), backend)
        else:
            raise ValueError(f'unknown backend {backend!r}')
        return torch.fft.ifft2(torch.fft.fft2(z) * response).real

    def extra_repr(self):
        return f'shape={self.shape}, scales={self.scales}, shear_levels={self.shear_levels}, bands={self.num_bands}'


def shearlet2d(x, system, *, chunk_size=None):
    return system.analysis(x, chunk_size=chunk_size)


def ishearlet2d(coeff, system, *, chunk_size=None):
    return system.synthesis(coeff, chunk_size=chunk_size)


def shearlet_gate(x, weights, system, *, backend='auto'):
    return system.gate(x, weights, backend=backend)


class ShearletGate(nn.Module):
    """Identity-initialized shared or channel-wise directional spectral filter.

    For channels=C, input must have shape (..., C, H, W). Use channels=None
    for one shared K-vector and compatibility with custom response kernels.
    """
    def __init__(self, shape, scales=3, *, shear_levels=None, channels=None, backend='auto', device=None, dtype=torch.float32):
        super().__init__()
        self.system = ShearletSystem(shape, scales, shear_levels=shear_levels, device=device, dtype=dtype)
        if channels is not None and (not isinstance(channels, int) or channels < 1):
            raise ValueError('channels must be a positive integer')
        self.channels, self.backend = channels, backend
        k = self.system.num_bands
        self.weight = nn.Parameter(torch.ones((k,) if channels is None else (channels, k), device=device, dtype=dtype))

    def forward(self, x):
        if self.channels is not None and (x.ndim < 3 or x.shape[-3] != self.channels):
            raise ValueError('input channel count does not match gate')
        return self.system.gate(x, self.weight, backend=self.backend)
