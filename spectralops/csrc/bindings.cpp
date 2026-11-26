#include <torch/extension.h>

torch::Tensor hadamard_cuda(torch::Tensor x, torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard", &hadamard_cuda, "Orthonormal WHT or fused spectral gate (CUDA)");
}
