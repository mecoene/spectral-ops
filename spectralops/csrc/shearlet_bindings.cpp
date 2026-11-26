#include <torch/extension.h>
torch::Tensor shearlet_response_cuda(torch::Tensor filters, torch::Tensor weights);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("response", &shearlet_response_cuda, "Weighted squared shearlet response (CUDA)");
}
