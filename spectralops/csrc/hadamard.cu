#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

__device__ void butterfly(float* s, int n) {
    for (int h=1; h<n; h*=2) {
        for (int j=threadIdx.x; j<n/2; j+=blockDim.x) {
            int a=(j/h)*(2*h)+(j%h), b=a+h;
            float u=s[a], v=s[b];
            s[a]=u+v; s[b]=u-v;
        }
        __syncthreads();
    }
}

template<typename scalar_t, bool Gate>
__global__ void kernel(const scalar_t* x, const scalar_t* w, scalar_t* y, int n, float scale) {
    extern __shared__ float s[];
    const int64_t base=static_cast<int64_t>(blockIdx.x)*n;
    for(int i=threadIdx.x; i<n; i+=blockDim.x) s[i]=static_cast<float>(x[base+i]);
    __syncthreads();
    butterfly(s,n);
    if (Gate) {
        for(int i=threadIdx.x; i<n; i+=blockDim.x) s[i]=s[i]*scale*static_cast<float>(w[i]);
        __syncthreads();
        butterfly(s,n);
    }
    for(int i=threadIdx.x; i<n; i+=blockDim.x) y[base+i]=static_cast<scalar_t>(s[i]*scale);
}

torch::Tensor hadamard_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim()>0, "expected contiguous CUDA input");
    TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16, "unsupported dtype");
    const auto n64=x.size(-1);
    TORCH_CHECK(n64>0 && n64<=1024 && (n64 & (n64-1))==0, "invalid transform length");
    const int n=static_cast<int>(n64);
    const bool gate=w.numel()!=0;
    if(gate) TORCH_CHECK(w.device()==x.device() && w.scalar_type()==x.scalar_type() && w.is_contiguous() && w.dim()==1 && w.numel()==n, "invalid weight vector");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y=torch::empty_like(x);
    const auto rows=x.numel()/n;
    if(rows==0) return y;
    TORCH_CHECK(rows<=2147483647LL, "too many rows");
    const int threads=std::min(256,std::max(32,n));
    const float scale=1.0f/std::sqrt(static_cast<float>(n));
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "spectralops_wht", [&] {
        if(gate) kernel<scalar_t,true><<<rows,threads,n*sizeof(float),stream>>>(x.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),y.data_ptr<scalar_t>(),n,scale);
        else kernel<scalar_t,false><<<rows,threads,n*sizeof(float),stream>>>(x.data_ptr<scalar_t>(),nullptr,y.data_ptr<scalar_t>(),n,scale);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
