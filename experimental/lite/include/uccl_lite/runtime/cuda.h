#pragma once

#include <stdexcept>
#include <cuda_runtime_api.h>

namespace uccl_lite::runtime {
inline void checkCuda(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
}  // namespace uccl_lite::runtime
