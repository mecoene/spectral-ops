"""Optional JAX-native implementations (JIT/autodiff, not custom GPU kernels)."""
import math
import jax.numpy as jnp
from jax.scipy.fft import dct as _dct, idct as _idct


def _input(x, axis):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("expected real floating input")
    if not -x.ndim <= axis < x.ndim or x.shape[axis] == 0:
        raise ValueError("invalid or empty transform axis")
    return x


def _power(n):
    if n <= 0 or n & (n-1):
        raise ValueError("transform length must be a positive power of two")


def _work(x):
    return x.astype(jnp.float32) if x.dtype in (jnp.float16,jnp.bfloat16) else x


def hadamard(x, axis=-1):
    x = _input(x,axis)
    z = _work(jnp.moveaxis(x,axis,-1))
    shape = z.shape
    n = shape[-1]
    _power(n)
    h = 1
    while h < n:
        a = z.reshape(*shape[:-1], n//(2*h),2,h)
        u,v = a[...,0,:], a[...,1,:]
        z = jnp.concatenate((u+v,u-v),axis=-1).reshape(shape)
        h *= 2
    return jnp.moveaxis((z/math.sqrt(n)).astype(x.dtype),-1,axis)


def dct(x,axis=-1):
    x = _input(x,axis)
    return _dct(_work(x),type=2,axis=axis,norm="ortho").astype(x.dtype)


def idct(x,axis=-1):
    x = _input(x,axis)
    return _idct(_work(x),type=2,axis=axis,norm="ortho").astype(x.dtype)


def _haar(x,axis,levels,inverse):
    x = _input(x,axis)
    z = _work(jnp.moveaxis(x,axis,-1))
    n = z.shape[-1]
    _power(n)
    if levels is None: levels = n.bit_length()-1
    if not isinstance(levels,int) or not 0 <= levels <= n.bit_length()-1:
        raise ValueError("invalid levels")
    s = 1/math.sqrt(2)
    for step in (reversed(range(levels)) if inverse else range(levels)):
        width=n>>step
        if inverse:
            a,b=z[...,:width//2],z[...,width//2:width]
            q=jnp.stack(((a+b)*s,(a-b)*s),axis=-1).reshape(*z.shape[:-1],width)
            z=jnp.concatenate((q,z[...,width:]),axis=-1)
        else:
            a,b=z[...,:width:2],z[...,1:width:2]
            z=jnp.concatenate(((a+b)*s,(a-b)*s,z[...,width:]),axis=-1)
    return jnp.moveaxis(z.astype(x.dtype),-1,axis)


def haar(x,axis=-1,*,levels=None): return _haar(x,axis,levels,False)
def ihaar(x,axis=-1,*,levels=None): return _haar(x,axis,levels,True)


def spectral_gate(x,weights,*,transform="hadamard",axis=-1,levels=None):
    x = _input(x,axis)
    w = jnp.asarray(weights)
    if not jnp.issubdtype(w.dtype,jnp.floating): raise TypeError("expected floating weights")
    if levels is not None and transform != "haar": raise ValueError("levels applies only to Haar")
    axis %= x.ndim
    z = jnp.moveaxis(x,axis,-1)
    if w.ndim==1:
        if w.shape[0]!=z.shape[-1]: raise ValueError("weight length mismatch")
    elif w.ndim>1:
        if jnp.broadcast_shapes(x.shape,w.shape)!=x.shape: raise ValueError("weights must not expand input")
        w=jnp.moveaxis(w.reshape((1,)*(x.ndim-w.ndim)+w.shape),axis,-1)
    dtype=jnp.result_type(z,w)
    z,w=_work(z.astype(dtype)),_work(w.astype(dtype))
    if transform=="hadamard": y=hadamard(hadamard(z)*w)
    elif transform=="dct": y=idct(dct(z)*w)
    elif transform=="haar": y=ihaar(haar(z,levels=levels)*w,levels=levels)
    else: raise ValueError("unknown transform")
    return jnp.moveaxis(y.astype(dtype),-1,axis)
