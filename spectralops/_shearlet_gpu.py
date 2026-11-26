"""Optional custom filter reduction kernels. FFTs remain torch.fft/cuFFT."""
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    src = Path(__file__).parent / 'csrc'
    return load(name='spectralops_shearlet_v020', sources=[str(src/'shearlet_bindings.cpp'), str(src/'shearlet.cu')], extra_cflags=['-O3'], extra_cuda_cflags=['-O3'])


class _Response(torch.autograd.Function):
    @staticmethod
    def forward(ctx, filters, weights, backend):
        ctx.save_for_backward(filters, weights)
        if backend == 'cuda':
            return _extension().response(filters, weights)
        from ._shearlet_triton import launch
        return launch(filters, weights)

    @staticmethod
    def backward(ctx, grad):
        f, w = ctx.saved_tensors
        df = 2*f*w[:, None, None]*grad if ctx.needs_input_grad[0] else None
        dw = torch.einsum('khw,hw->k', f.square(), grad) if ctx.needs_input_grad[1] else None
        return df, dw, None


def response_gpu(filters, weights, backend):
    return _Response.apply(filters, weights, backend)
