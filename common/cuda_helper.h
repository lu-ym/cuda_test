#pragma once

#include <iostream>
#include <string>

// cuda api
#include <cuda_runtime.h>

#define CheckCudaError(val) CudaHelper::Check((val), #val, __FILE__, __LINE__)

namespace CudaHelper {

/**
 * @note use template check error code types
 */
template <typename T>
void Check(T err, const char* const func, const char* const file,
           const int line);

template <typename T>
class DeviceVector {
 public:
  DeviceVector(std::size_t size) {
    if (size == 0) {
      std::cout << __func__ << ":Vector size must large than 0" << std::endl;
      return;
    }
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * size);
    if (status != cudaSuccess || ptr == nullptr) {
      std::cout << __func__ << "malloc fail with error code " << status
                << std::endl;
    }
#ifdef __CUDACC_DEBUG__
    else {
      std::cout << __func__ << "malloced ptr is " << ptr << std::endl;
    }
#endif
  }

  ~DeviceVector() {
    if (ptr != nullptr) {
      cudaError_t status = cudaFree(ptr);
      if (status != cudaSuccess) {
        std::cout << __func__ << "free fail with error code " << status
                  << std::endl;
      }
    } else {
      std::cout << __func__ << "ptr is nullptr" << std::endl;
    }
  }

  DeviceVector(DeviceVector&) = delete;
  DeviceVector& operator=(DeviceVector&) = delete;
  DeviceVector(DeviceVector&&) = delete;
  DeviceVector& operator=(DeviceVector&&) = delete;

  T* data() { return ptr; };

 private:
  T* ptr{nullptr};
};

}  // namespace CudaHelper
