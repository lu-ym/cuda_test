
#include <functional>
#include <vector>

// cuda headers
#include <cublas_v2.h>
// #include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"
#include "reduction.h"
#include "utils.h"
#include "utilsm.h"

/**
 * cudaFuncCachePreferNone = 0
 *   Default function cache configuration, no preference
 * cudaFuncCachePreferShared = 1
 *   Prefer larger shared memory and smaller L1 cache
 * cudaFuncCachePreferL1 = 2
 *   Prefer larger L1 cache and smaller shared memory
 * cudaFuncCachePreferEqual = 3
 *   Prefer equal size L1 cache and shared memory
 */
void getSharedMemoryCfg(void) {
  cudaFuncCache pCacheCfg;
  CheckCudaError(cudaDeviceGetCacheConfig(&pCacheCfg));
  std::cout << "Device CacheConfig is: " << pCacheCfg << std::endl;
}
/**
 * @brief Cblas asum,
 * @param in input values. must be device pointer
 * @param out result value. must be host pointer
 */
inline void cublasAsum(const float *in, float *out, std::size_t elems) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  // output must be host result
  CheckCudaError(cublasCreate(&cublasH));
  CheckCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CheckCudaError(cublasSetStream(cublasH, stream));
  CheckCudaError(cublasSasum(cublasH, static_cast<int>(elems), in, 1, out));
  CheckCudaError(cudaStreamSynchronize(stream));
  CheckCudaError(cublasDestroy(cublasH));
  CheckCudaError(cudaStreamDestroy(stream));

  // &out = cublasSasum(static_cast<int>(elems), in, 1);
  return;
}

const std::vector<std::function<void(const float *, float *, std::size_t)>>
    functions{cublasAsum, test_asum0, test_asum1, test_asum3, test_asum8};

int main() {
  int cold_iters = 5;
  int hot_iters = 10;
  bool status = true;

  std::function<void(const float *, float *, std::size_t)> asum_func =
      test_asum8;

  getSharedMemoryCfg();
  // CheckCudaError(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  // getSharedMemoryCfg();

  // all range for simple validataion.
  // for (uint32_t power = 2; power <= 25; ++power) {
  // specfic test case which could passed kernel-7
  // for (uint32_t power = 15; power <= 15; ++power) {
  // reference power number which is 4M elements as Mark Harris did.
  for (uint32_t power = 22; power <= 22; ++power) {
    std::cout << "---------------------------------"
              << " Power is:" << power << " "
              << "---------------------------------" << std::endl;
    size_t numElements = 1 << power;
    std::cout << "Elements count is:";
    if (power >= 20) {
      std::cout << (1 << (power % 20)) << "M";
    } else if (power >= 10) {
      std::cout << (1 << (power % 10)) << "K";
    } else {
      std::cout << numElements;
    }
    std::cout << std::endl;

    /**************** Prepare inputs ********************/
    // malloc and set intput vector
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(1);
    float golden{0.f};

    // fake inputs
    GenRandomVec(h_A);

    CudaHelper::DeviceVector<float> d_A(numElements);
    // including some workspace
    CudaHelper::DeviceVector<float> d_B(numElements);
    CheckCudaError(cudaMemcpy(d_A.data(), h_A.data(),
                              numElements * sizeof(float), cudaMemcpyDefault));

    /**************** Test Compute and Compare ********************/
    // golden compute
    cublasAsum(d_A.data(), &golden, numElements);

    // test
    asum_func(d_A.data(), d_B.data(), numElements);
    CheckCudaError(cudaDeviceSynchronize());
    // copy back
    CheckCudaError(
        cudaMemcpy(h_B.data(), d_B.data(), sizeof(float), cudaMemcpyDefault));
    if (pointEqual(h_B[0], golden)) {
      status = true;
      std::cout << "[PASS] ";
    } else {
      status = false;
      std::cout << "[FAIL] Test is " << h_B[0] << ". Golden is" << golden
                << std::endl;
    }

    /**************** Benchmark ********************/
    if (status) {
      cudaEvent_t start_g, stop_g;
      CheckCudaError(cudaEventCreate(&start_g));
      CheckCudaError(cudaEventCreate(&stop_g));

      float duration_golden{0.0};
      for (int i = 0; i < cold_iters; ++i) {
        cublasAsum(d_A.data(), &golden, numElements);
      }
      CheckCudaError(cudaEventRecord(start_g));
      for (int i = 0; i < hot_iters; ++i) {
        cublasAsum(d_A.data(), &golden, numElements);
      }
      CheckCudaError(cudaEventRecord(stop_g));
      CheckCudaError(cudaEventSynchronize(stop_g));
      CheckCudaError(cudaEventElapsedTime(&duration_golden, start_g, stop_g));

      float duration_test{0.0};
      for (int i = 0; i < cold_iters; ++i) {
        asum_func(static_cast<float *>(d_A.data()),
                  static_cast<float *>(d_B.data()), numElements);
      }
      CheckCudaError(cudaEventRecord(start_g));
      for (int i = 0; i < hot_iters; ++i) {
        asum_func(d_A.data(), d_B.data(), numElements);
      }
      CheckCudaError(cudaEventRecord(stop_g));
      CheckCudaError(cudaEventSynchronize(stop_g));
      CheckCudaError(cudaEventElapsedTime(&duration_test, start_g, stop_g));

      std::cout << "percentage is: " << duration_test / duration_golden * 100;
      std::cout << "%. Test is " << (duration_test * 1000 / hot_iters) << " us";
      std::cout << " Golden is " << (duration_golden * 1000 / hot_iters)
                << " us." << std::endl;

      CheckCudaError(cudaEventDestroy(start_g));
      CheckCudaError(cudaEventDestroy(stop_g));
    }
  }

  return 0;
}
