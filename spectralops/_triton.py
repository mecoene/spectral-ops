"""Experimental row-local fused WHT kernels; requires an NVIDIA GPU and Triton."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _butterfly(v, idx, LOG_N: tl.constexpr):
    for stage in tl.static_range(LOG_N):
        bit = 1 << stage
        peer = tl.gather(v, idx ^ bit, axis=0)
        v = tl.where((idx & bit) == 0, v + peer, peer - v)
    return v


@triton.jit
def _kernel(X, W, Y, N: tl.constexpr, LOG_N: tl.constexpr,
            SCALE: tl.constexpr, GATE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, N)
    v = tl.load(X + row*N + i).to(tl.float32)
    v = _butterfly(v, i, LOG_N) * SCALE
    if GATE:
        w = tl.load(W + i).to(tl.float32)
        v = _butterfly(v*w, i, LOG_N) * SCALE
    tl.store(Y + row*N + i, v)


def launch(x, w=None):
    n = x.shape[-1]
    y = torch.empty_like(x)
    rows = x.numel() // n
    if rows == 0:
        return y
    with torch.cuda.device(x.device):
        _kernel[(rows,)](x, x if w is None else w, y, n,
                         n.bit_length()-1, 1/math.sqrt(n), w is not None,
                         num_warps=4 if n >= 128 else 1)
    return y
