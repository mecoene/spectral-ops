"""SpectralOps: differentiable spectral layers."""
from .ops import hadamard, dct, idct, dct2d, idct2d, haar, ihaar, haar2d, ihaar2d, spectral_gate
from .nn import SpectralGate
__version__ = "0.1.0"
__all__ = ["hadamard", "dct", "idct", "dct2d", "idct2d", "haar", "ihaar", "haar2d", "ihaar2d", "spectral_gate", "SpectralGate"]
