"""Learnable spectral layers."""
import torch
from torch import nn
from .ops import spectral_gate


class SpectralGate(nn.Module):
    """Identity-initialized learnable diagonal operator in a spectral basis."""
    def __init__(self, features, transform="hadamard", dim=-1, *, levels=None, backend="auto", device=None, dtype=None):
        super().__init__()
        if not isinstance(features, int) or features <= 0:
            raise ValueError("features must be a positive integer")
        if transform not in {"hadamard", "dct", "haar"}:
            raise ValueError("unknown transform")
        if transform in {"hadamard", "haar"} and features & (features-1):
            raise ValueError("features must be a power of two")
        self.features, self.transform, self.dim = features, transform, dim
        self.levels, self.backend = levels, backend
        self.weight = nn.Parameter(torch.ones(features, device=device, dtype=dtype))

    def forward(self, x):
        return spectral_gate(x, self.weight, transform=self.transform, dim=self.dim, levels=self.levels, backend=self.backend)

    def extra_repr(self):
        return f"features={self.features}, transform={self.transform!r}, dim={self.dim}, backend={self.backend!r}"
