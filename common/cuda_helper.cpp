#include "cuda_helper.h"

#include <iostream>

// cuda api
#include <cublas.h>
// #include <cublasLt.h>
#include <cuda_runtime.h>

namespace CudaHelper {

/**
 * @note use template check error code types
 */
template <typename T>
void Check(T err, const char* const func, const char* const file,
           const int line);

template <>
void Check(cudaError_t err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

template <>
void Check(cublasStatus_t err, const char* const func, const char* const file,
           const int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS error at: " << file << ":" << func << ":" << line
              << std::endl;
    // comments to avoid "undefined reference to `cublasGetStatusString'
    // std::cerr << cublasGetStatusString(err) << " " << func << std::endl;
    exit(1);
  }
}

}  // namespace CudaHelper
