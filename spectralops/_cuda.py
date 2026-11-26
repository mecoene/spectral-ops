"""Lazy JIT build of the optional CUDA extension. No compiler needed for torch backend."""
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    src = Path(__file__).parent / "csrc"
    return load(name="spectralops_cuda_v010", sources=[str(src/"bindings.cpp"), str(src/"hadamard.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


def launch(x, w=None):
    if x.numel() == 0:
        return torch.empty_like(x)
    return _extension().hadamard(x, x.new_empty(0) if w is None else w)
