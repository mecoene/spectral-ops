# SpectralOps

**Differentiable orthogonal transforms and fused spectral neural network operators.**

Version 1.0.2 · research prototype · MIT

SpectralOps provides a single PyTorch API for orthonormal transforms and learnable spectral filters, plus a JAX-native implementation and experimental custom GPU kernels. Its primary optimization target is the complete Hadamard spectral gate, not merely a standalone transform:

$$y = H^T\mathrm{diag}(w)Hx$$

The CUDA and Triton forward kernels execute both butterflies and spectral weighting in one kernel for each batch of rows, keeping intermediates in on-chip storage where the implementation and compiler permit. This is a design objective, not a measured GPU speedup. Backpropagation for input gradients reuses the same fused gate; weight gradients currently use PyTorch reference operations and a reduction.

## Installation

From the extracted repository root, preferably inside a virtual environment:

```bash
python -m pip install -e '.[dev]'
```

Install a PyTorch build appropriate to your platform first if you need CUDA. Optional dependencies:

```bash
python -m pip install -e '.[jax]'     # JAX CPU by default; see JAX docs for GPU
python -m pip install -e '.[triton]'  # NVIDIA GPU, compatible Linux Triton
python -m pip install -e '.[cuda]'    # ninja; also needs CUDA toolkit and C++ compiler
```

The CUDA backend lazily builds its extension on first use. No CUDA compiler is needed for ordinary PyTorch operations. Do not install this project from a public package index until you have verified its identity and published it yourself; the name is provisional.

## Quick start

```python
import torch
import spectralops as so

x = torch.randn(32, 128, requires_grad=True)
h = so.hadamard(x)
assert torch.allclose(so.hadamard(h), x, atol=1e-6)
d = so.dct(x)
x_hat = so.idct(d)
a = so.haar(x, levels=3)
x_hat = so.ihaar(a, levels=3)

layer = so.SpectralGate(128, transform='hadamard')
y = layer(x)
y.square().mean().backward()

image = torch.randn(4, 3, 128, 128)
coeff = so.haar2d(image, levels=2)
reconstructed = so.ihaar2d(coeff, levels=2)
```

### Experimental GPU backends

```python
x = torch.randn(512, 256, device='cuda', requires_grad=True)
w = torch.randn(256, device='cuda', requires_grad=True)
y = so.spectral_gate(x, w, backend='triton')  # or backend='cuda'
y.square().mean().backward()
```

`backend='auto'` deliberately uses the PyTorch reference implementation, even on GPU, until the experimental kernels are validated on hardware. Explicit backend requests fail rather than silently falling back. The GPU kernels support power-of-two transform lengths 1–1024, NVIDIA CUDA devices, FP32/FP16/BF16 (FP32 internal arithmetic), and a same-dtype 1D gate weight vector. Inputs are moved to the last axis and made contiguous as necessary, which can introduce extra copy costs. Accelerated functions use custom autograd and have not been validated with `torch.compile`, CUDA graphs or `torch.func` transforms. Use `backend='torch'` for those experiments.

## Operator support

| Operator | PyTorch native | Custom CUDA | Triton | JAX native |
|---|---|---|---|---|
| Hadamard (WHT) | Yes | Experimental | Experimental | Yes |
| Fused Hadamard gate | Yes (unfused reference) | Experimental | Experimental | Yes (JIT composition) |
| DCT-II / inverse | Yes | No | No | Yes |
| Haar / inverse | Yes | No | No | Yes |
| Separable 2D DCT/Haar | Yes | No | No | Compose axis transforms |
| Trainable gate module | Yes | Experimental | Experimental | Functional API |

All transforms are orthonormal. Hadamard uses natural Sylvester ordering (not sequency). DCT uses type II and its inverse; any positive transform length is supported. Haar requires a power-of-two length and packs coefficients as `[approximation, coarsest details, ..., finest details]`. The 2D Haar is separable full-axis decomposition, not the conventional recursively LL-only wavelet pyramid. All operators accept an arbitrary transform axis. Gate weights can be scalar, a vector matching the transform axis, or a higher-rank tensor broadcastable to the original input without expanding it (reference backend only for scalar/higher rank).

### JAX

```python
import jax
import jax.numpy as jnp
from spectralops import jax as sj

x = jnp.ones((32, 128))
w = jnp.ones(128)
f = jax.jit(lambda a, b: sj.spectral_gate(a, b, transform='hadamard'))
y = f(x, w)
dw = jax.grad(lambda b: jnp.mean(f(x, b)**2))(w)
```

The JAX integration uses JAX primitives, not FFI bindings to the custom CUDA kernels. Transform and axis options should be static under `jit`. The project currently depends on PyTorch even if you only use its JAX submodule.

Dense WHT uses matrix multiplication (typically cuBLAS/cuBLASLt on NVIDIA GPU).

Related implementations and documentation: [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform), [HadaCore](https://pytorch.org/blog/hadacore/), [Triton](https://triton-lang.org/), [PyTorch C++ extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html), [JAX FFT](https://docs.jax.dev/en/latest/jax.scipy.fft.html). These are prior work and references, not code bundled into this library. Research novelty and package-name availability have not been established.

V1.0.2 update:

I just added 2D cone-adapted shearlets (Shearlet Wavelets) along with Shearlet gate.

```python
import torch
import spectralops as so

system = so.ShearletSystem((128, 128), scales=4)
x = torch.randn(2, 3, 128, 128, requires_grad=True)
c = so.shearlet2d(x, system, chunk_size=8)  # (2, 3, 37, 128, 128)
x_hat = so.ishearlet2d(c, system, chunk_size=8)
filters = system.filters                     # (37, 128, 128), FFT order
kernels = system.spatial_kernels()           # centered spatial impulse responses
layer = so.ShearletGate((128,128), scales=4, channels=3)
layer(x).square().mean().backward()
# On an NVIDIA GPU: shared weights, FP32 filter bank, experimental response kernels
# layer = so.ShearletGate((128,128), scales=4, backend='triton').cuda()
# y = layer(x.cuda())  # or backend='cuda'
```

The system is real, band-limited, undecimated and Parseval-normalized, not Gaussian directional masks or a compact-support ShearLab port. The fused gate avoids materializing the full coefficient tensor by reducing squared frequency masks with learned band weights; FFTs still use PyTorch/cuFFT.



