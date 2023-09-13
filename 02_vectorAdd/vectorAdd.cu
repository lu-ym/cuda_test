#include <ctime>
#include <iomanip>
#include <iostream>

// success after add this comment -- should be VS issue.
// #pragma comment(lib,"cuda.lib")

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// #include "math_functions.h"
#include "cuda_helper.h"
#include "utils.h"
#include "utilsm.h"

#define MAX_THREADS 1024 * 1024 * 64
// #define MAX_THREADS 32 * 3072
// #define MAX_THREADS 64 * 1024

// #pragma comment(lib,"cudart.lib")
void generate_input(float *input, unsigned int input_len) {
  std::srand(std::time(nullptr));
  for (unsigned int i = 0; i < input_len; ++i) {
    input[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

// golden test
void cpu_vector_add(float *const vector1, float *const vector2,
                    float *vector_sum, unsigned int len) {
  for (unsigned int i = 0; i < len; ++i) {
    vector_sum[i] = vector1[i] + vector2[i];
    // vector1[i] += vector2[i];
  }
}

/**
 * @brief
 */
__global__ void gpu_vector_add(float *const vector1, float *const vector2,
                               float *vector_sum, const unsigned int length) {
  // unsigned int tid;
  // for (unsigned int i = 0; i < length;) {
  //     tid = threadIdx.x + blockDim.x * blockIdx.x + i;
  //     i += MAX_THREADS;
  //     if ((tid < length) && (tid < i)) {
  //         vector_sum[tid] = vector1[tid] + vector2[tid];
  //         // vector1[tid] += vector2[tid];
  //     }
  // }
  // single cycle version
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // if (tid < length) {
  vector_sum[tid] = vector1[tid] + vector2[tid];
  // vector1[tid] += vector2[tid];
  // }
}

__global__ void gpu_vector_add4(float4 *vector1, float4 *const vector2,
                                float4 *vector_sum, const unsigned int length) {
  // unsigned int tid;
  // for (unsigned int i = 0; i < length;) {
  //     tid = threadIdx.x + blockDim.x * blockIdx.x + i;
  //     i += MAX_THREADS;
  //     if ((tid < length) && (tid < i)) {
  //         vector_sum[tid].x = vector1[tid].x + vector2[tid].x;
  //         vector_sum[tid].y = vector1[tid].y + vector2[tid].y;
  //         vector_sum[tid].z = vector1[tid].z + vector2[tid].z;
  //         vector_sum[tid].w = vector1[tid].w + vector2[tid].w;
  //     }
  // }
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // if (tid < length) {
  vector_sum[tid].x = vector1[tid].x + vector2[tid].x;
  vector_sum[tid].y = vector1[tid].y + vector2[tid].y;
  vector_sum[tid].z = vector1[tid].z + vector2[tid].z;
  vector_sum[tid].w = vector1[tid].w + vector2[tid].w;
  // }
}
__global__ void gpu_vector_add5(float4 *vector1, float4 *const vector2,
                                float4 *vector_sum, const unsigned int length) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // float4 plus is not allowed on new cuda
  // vector_sum[tid] = vector1[tid] + vector2[tid];
  vector_sum[tid].x = vector1[tid].x + vector2[tid].x;
  vector_sum[tid].y = vector1[tid].y + vector2[tid].y;
  vector_sum[tid].z = vector1[tid].z + vector2[tid].z;
  vector_sum[tid].w = vector1[tid].w + vector2[tid].w;
}

int main() {
  // Set up clock for timing comparisons
  std::clock_t start;
  float duration_cpu, duration_gpu;
  unsigned int len_pow = 24;
  // std::cout << "CLOCKS_PER_SEC is " << CLOCKS_PER_SEC << std::endl;//10^6
  // for (; len_pow < 25; len_pow++) {
  std::cout << "---------------- len_pow is " << len_pow << " ----------------"
            << std::endl;
  unsigned int vector_len = (unsigned int)1 << len_pow;
  unsigned int length_bytes = vector_len * sizeof(int);
  std::cout << "vector length is " << vector_len << std::endl;
  std::cout << "vector size is "
            << static_cast<double>(length_bytes / 1024) / 1024 * sizeof(int)
            << " GB." << std::endl;

  float *vector1 = new float[vector_len];  // on host
  float *vector2 = new float[vector_len];
  float *vector_sum = new float[vector_len];
  float *vector_sum2 = new float[vector_len];
  float *vector1_g, *vector2_g, *vector_sum_g;  // in GPU

  // generate input
  generate_input(vector1, vector_len);
  generate_input(vector2, vector_len);
  start = std::clock();
  cpu_vector_add(vector1, vector2, vector_sum, vector_len);
  duration_cpu = std::clock() - start;
  std::cout << "CPU computation:" << std::endl;
  std::cout << "  duration: ";
  print_time_us(duration_cpu);

  // Set up device-side memory for input
  CheckCudaError(cudaMalloc(&vector1_g, length_bytes));
  CheckCudaError(cudaMalloc(&vector2_g, length_bytes));
  CheckCudaError(cudaMalloc(&vector_sum_g, length_bytes));
  CheckCudaError(
      cudaMemcpy(vector1_g, vector1, length_bytes, cudaMemcpyHostToDevice));
  CheckCudaError(
      cudaMemcpy(vector2_g, vector2, length_bytes, cudaMemcpyHostToDevice));

  // generate CUDA parameters and call CUDA device functions
  // std::cout << "Basic GPU computing version" << std::endl;
  // unsigned int max_threads_used;
  // if (vector_len > MAX_THREADS) {
  //   max_threads_used = MAX_THREADS;
  //   std::cout << "MAX_THREADS is: " << MAX_THREADS << std::endl
  //             << "while vector_len is: " << vector_len << std::endl;
  // } else {
  //   max_threads_used = vector_len;
  // }
  // unsigned int block_sz = 1024;
  // unsigned int grid_sz = max_threads_used / block_sz;
  // if (max_threads_used % block_sz) grid_sz++;
  // cudaEvent_t start_g, stop_g;
  // CheckCudaError(cudaEventCreate(&start_g));
  // CheckCudaError(cudaEventCreate(&stop_g));
  // CheckCudaError(cudaEventRecord(start_g));
  // gpu_vector_add<<<grid_sz, block_sz>>>(vector1_g, vector2_g, vector_sum_g,
  //                                       vector_len);
  // CheckCudaError(cudaEventRecord(stop_g));
  // CheckCudaError(cudaEventSynchronize(stop_g));
  // CheckCudaError(cudaEventElapsedTime(&duration_gpu, start_g,
  // stop_g)); std::cout << "GPU computation:" << std::endl; std::cout << "
  // duration: " << duration_gpu << "ms" << std::endl;
  // CheckCudaError(cudaEventDestroy(start_g));
  // CheckCudaError(cudaEventDestroy(stop_g));

  // float4 version
  std::cout << "GPU computing float4 version" << std::endl;
  unsigned int max_threads_used = vector_len / 4;
  if (max_threads_used > MAX_THREADS) {
    max_threads_used = MAX_THREADS;
    std::cout << "MAX_THREADS is: " << MAX_THREADS << std::endl
              << "while vector_len is: " << vector_len << std::endl;
  }
  unsigned int block_sz = 1024;
  unsigned int grid_sz = (max_threads_used + block_sz - 1) / block_sz;
  cudaEvent_t start_g, stop_g;
  CheckCudaError(cudaEventCreate(&start_g));
  CheckCudaError(cudaEventCreate(&stop_g));
  CheckCudaError(cudaEventRecord(start_g));
  // gpu_vector_add4<<<grid_sz, block_sz>>>(
  gpu_vector_add5<<<grid_sz, block_sz>>>(
      (float4 *)vector1_g, (float4 *)vector2_g, (float4 *)vector_sum_g,
      vector_len / 4);
  CheckCudaError(cudaEventRecord(stop_g));
  CheckCudaError(cudaEventSynchronize(stop_g));
  CheckCudaError(cudaEventElapsedTime(&duration_gpu, start_g, stop_g));
  std::cout << "GPU computation:" << std::endl;
  std::cout << "  duration: " << duration_gpu << "ms" << std::endl;
  CheckCudaError(cudaEventDestroy(start_g));
  CheckCudaError(cudaEventDestroy(stop_g));

  // copy back
  CheckCudaError(cudaMemcpy(vector_sum2, vector_sum_g, length_bytes,
                            cudaMemcpyDeviceToHost));
  // CheckCudaError(cudaMemcpy(vector_sum2, vector1_g, length_bytes,
  //                            cudaMemcpyDeviceToHost));
  // acceleration rate
  std::cout << "Calculation Result -- ";
  bool result = true;
  for (unsigned int i = 0; i < vector_len; i++) {
    if (!pointEqual(vector_sum[i], vector_sum2[i])) {
      // if (vector_sum[i] != vector_sum2[i]) {
      std::cout << "Data mismatch at " << i << ": cpu_sum is " << vector_sum[i]
                << " while gpu sum is " << vector_sum2[i] << std::endl;
      result = false;
      print_vector(vector_sum, 0, 16);
      print_vector(vector_sum, vector_len - 16, vector_len);
      print_vector(vector_sum2, 0, 16);
      print_vector(vector_sum2, vector_len - 16, vector_len);
      break;
    }
  }
  std::cout << std::boolalpha << result << std::endl;
  std::cout << "Acceleration Rate: " << std::resetiosflags(std::ios::fixed)
            << duration_cpu / 1000 / duration_gpu << " times" << std::endl;
  std::cout << std::endl;

  delete[] vector1;
  delete[] vector2;
  delete[] vector_sum;
  delete[] vector_sum2;
  // }
  return 0;
}
