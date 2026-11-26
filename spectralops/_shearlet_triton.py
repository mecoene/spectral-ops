"""One GPU kernel reduces squared shearlet filters with learned band weights."""
import torch
import triton
import triton.language as tl


@triton.jit
def _response(F, W, Y, P: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0., tl.float32)
    for k in range(K):
        f = tl.load(F+k*P+i, i<P, other=0.)
        w = tl.load(W+k)
        accum += f*f*w
    tl.store(Y+i, accum, i<P)


def launch(filters, weights):
    k,h,w = filters.shape
    out = torch.empty((h,w), device=filters.device, dtype=filters.dtype)
    with torch.cuda.device(filters.device):
        _response[(triton.cdiv(h*w, 256),)](filters, weights, out, h*w, k, 256)
    return out
