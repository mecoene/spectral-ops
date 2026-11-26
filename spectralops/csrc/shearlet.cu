#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void response_kernel(const float* f, const float* w, float* y, int64_t p, int64_t k) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= p) return;
    float sum = 0.f;
    for (int64_t b=0; b<k; ++b) {
        float v = f[b*p+i];
        sum = fmaf(v*v,w[b],sum);
    }
    y[i] = sum;
}

torch::Tensor shearlet_response_cuda(torch::Tensor filters, torch::Tensor weights) {
    TORCH_CHECK(filters.is_cuda() && weights.device()==filters.device(), "expected same CUDA device");
    TORCH_CHECK(filters.scalar_type()==at::kFloat && weights.scalar_type()==at::kFloat, "expected float32");
    TORCH_CHECK(filters.is_contiguous() && weights.is_contiguous(), "expected contiguous tensors");
    TORCH_CHECK(filters.dim()==3 && weights.dim()==1 && filters.size(0)==weights.size(0), "invalid shapes");
    const c10::cuda::CUDAGuard guard(filters.device());
    auto h=filters.size(1), w=filters.size(2), p=h*w;
    auto out=torch::empty({h,w},filters.options());
    if (!p) return out;
    TORCH_CHECK((p+255)/256<=2147483647LL, "grid too large");
    auto stream=at::cuda::getCurrentCUDAStream(filters.get_device());
    response_kernel<<<(p+255)/256,256,0,stream>>>(filters.data_ptr<float>(),weights.data_ptr<float>(),out.data_ptr<float>(),p,filters.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
