#include <stdlib.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

// #include "cublas_v2.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "helper_cuda.h"

#include "utils.h"
#include "utilsm.h"

#define WARP_SIZE 32
// #define MAX_BLOCK_THREADS 128
#define MAX_BLOCK_THREADS 1024
#define MAX_SHARED_MEMORY_SIZE (1024 * 48)

/**
 * @brief Kernel 1 -- interleaved addressing
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                              float *result) {
//   // statement needs constanct value. So need template.
//   // extern __shared__ float sdata[block_size];
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   // shared memory is only used in a block
//   uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
//   if (id_global < n) sdata[threadIdx.x] = x[id_global];
//   __syncthreads();
//   // do reduction in shared mem
//   for (uint32_t s = 1; s < blockDim.x; s *= 2) {
//     if (threadIdx.x % (2 * s) == 0) {
//       sdata[threadIdx.x] += sdata[threadIdx.x + s];
//     }
//     __syncthreads();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }

/**
 * @brief Kernel 2 -- interleaved addressing without divergence
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
//   if (id_global < n) sdata[threadIdx.x] = x[id_global];
//   __syncthreads();
//   // do reduction in shared mem
//   for (uint32_t s = 1; s < blockDim.x; s *= 2) {
//     uint32_t index = 2 * s * threadIdx.x;
//     if (index < blockDim.x) {
//       sdata[index] += sdata[index + s];
//     }
//     __syncthreads();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }

/**
 * @brief Kernel 3 -- sequential addressing
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
//   if (id_global < n) sdata[threadIdx.x] = x[id_global];
//   __syncthreads();
//   // do reduction in shared mem
//   for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//     if (threadIdx.x < s) {
//       sdata[threadIdx.x] += sdata[threadIdx.x + s];
//     }
//     __syncthreads();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }
/**
 * @brief test frame for kernel 1-3
 * @param power
 */
// void test_asum(uint32_t power) {
//   size_t numElements = 1 << power;
//   size_t size_total = numElements * sizeof(float);
//   std::cout << "Elements count is:";
//   if (power >= 20) {
//     std::cout << (1 << (power % 20)) << "M";
//   } else if (power >= 10) {
//     std::cout << (1 << (power % 10)) << "K";
//   } else {
//     std::cout << numElements;
//   }
//   std::cout << std::endl;
//   // malloc and set intput vector
//   float *h_A = (float *)malloc(size_total);
//   for (int i = 0; i < numElements; ++i) {
//     h_A[i] = 1.0;
//   }
//   float *d_A = NULL;
//   checkCudaErrors(cudaMalloc((void **)&d_A, size_total));
//   checkCudaErrors(cudaMemcpy(d_A, h_A, size_total, cudaMemcpyHostToDevice));
//   // block/grid size and malloc output vector
//   uint32_t block_dim_x = MAX_BLOCK_THREADS;
//   if (numElements < block_dim_x) {
//     block_dim_x = ((numElements + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
//   }
//   uint32_t grid_dim_x =
//       (numElements + MAX_BLOCK_THREADS - 1) / MAX_BLOCK_THREADS;
//   float *h_B = (float *)malloc(sizeof(float));
//   float *d_B = NULL;
//   // TODO: actually need large size. While small size could pass_also.
//   checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float) * grid_dim_x)));
//   // checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float))));
//   // bench mark
//   float duration_temp, duration_total{0.0};
//   cudaEvent_t start_g, stop_g;
//   checkCudaErrors(cudaEventCreate(&start_g));
//   checkCudaErrors(cudaEventCreate(&stop_g));
//   // kernel launch
//   uint32_t shared_mem_size = block_dim_x * sizeof(float);
//   dim3 blockDim(block_dim_x);
//   dim3 gridDim(grid_dim_x);
//   std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
//             << std::endl;
//   checkCudaErrors(cudaEventRecord(start_g));
//   asum_kernel<<<gridDim, blockDim, shared_mem_size>>>(numElements, d_A, 0, 0,
//                                                       d_B);
//   checkCudaErrors(cudaEventRecord(stop_g));
//   checkCudaErrors(cudaEventSynchronize(stop_g));
//   checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//   duration_total += duration_temp;
//   if (grid_dim_x > MAX_BLOCK_THREADS) {
//     // checkCudaErrors(cudaStreamSynchronize(NULL));
//     // checkCudaErrors(cudaDeviceSynchronize());
//     std::cout << "Next kernel launch" << std::endl;
//     uint32_t n = grid_dim_x;
//     grid_dim_x = (grid_dim_x + MAX_BLOCK_THREADS - 1) / MAX_BLOCK_THREADS;
//     gridDim = dim3(grid_dim_x);
//     checkCudaErrors(cudaEventRecord(start_g));
//     asum_kernel<<<gridDim, blockDim, shared_mem_size>>>(n, d_B, 0, 0, d_B);
//     checkCudaErrors(cudaEventRecord(stop_g));
//     checkCudaErrors(cudaEventSynchronize(stop_g));
//     checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//     duration_total += duration_temp;
//   }
//   if (grid_dim_x > 1) {
//     // checkCudaErrors(cudaStreamSynchronize(NULL));
//     // checkCudaErrors(cudaDeviceSynchronize());
//     std::cout << "Next kernel launch" << std::endl;
//     shared_mem_size = grid_dim_x * sizeof(grid_dim_x);
//     checkCudaErrors(cudaEventRecord(start_g));
//     asum_kernel<<<1, gridDim, shared_mem_size>>>(grid_dim_x, d_B, 0, 0, d_B);
//     checkCudaErrors(cudaEventRecord(stop_g));
//     checkCudaErrors(cudaEventSynchronize(stop_g));
//     checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//     duration_total += duration_temp;
//   }
//   std::cout << "  duration: " << (duration_total * 1000) << " us" <<
//   std::endl;
//   // copye back
//   // TODO(YingmaoLu):stream sync
//   checkCudaErrors(cudaMemcpy(h_B, d_B, sizeof(float),
//   cudaMemcpyDeviceToHost));
//   // Free device and host memory
//   checkCudaErrors(cudaFree(d_A));
//   checkCudaErrors(cudaFree(d_B));
//   checkCudaErrors(cudaEventDestroy(start_g));
//   checkCudaErrors(cudaEventDestroy(stop_g));
//   float golden = 1.0 * numElements;
//   if (pointEqual(h_B[0], golden)) {
//     std::cout << "GPU compute PASS." << std::endl;
//   } else {
//     std::cout << "GPU compute FAIL. " << std::endl;
//     std::cout << "\tResult is :" << h_B[0] << ". Golden is:" << golden
//               << std::endl;
//   }
//   free(h_A);
//   free(h_B);
// }

/******************************************************************************/
#define EACH_BLOCK_SIZE_TIMES 2
/**
 * @brief Kernel 4 -- first add during load
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   // NOTE (LYM): blockDim.x*2 due to elements for a block is doubled.
//   //          And shared_memory size is also doubled.
//   uint32_t id_global =
//       (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
//   if ((id_global + blockDim.x) < n) {  // need update to the lastest value.
//     sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
//   }
//   __syncthreads();
//   // do reduction in shared mem
//   for (int s = blockDim.x / EACH_BLOCK_SIZE_TIMES; s > 0; s >>= 1) {
//     if (threadIdx.x < s) {
//       sdata[threadIdx.x] += sdata[threadIdx.x + s];
//     }
//     __syncthreads();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }
/**
 * @brief Kernel 5 -- Unroll the last warp
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   uint32_t id_global =
//       (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
//   if ((id_global + blockDim.x) < n) {
//     sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
//   }
//   __syncthreads();
//   // do reduction in shared mem
//   for (int s = blockDim.x / EACH_BLOCK_SIZE_TIMES; s > 32; s >>= 1) {
//     if (threadIdx.x < s) {
//       sdata[threadIdx.x] += sdata[threadIdx.x + s];
//     }
//     __syncthreads();
//   }
//   // unroll the last warp
//   if (threadIdx.x < 32) {
//     float tmp = sdata[threadIdx.x];
//     tmp += sdata[threadIdx.x + 32];
//     sdata[threadIdx.x] = tmp;
//     __syncwarp();
//     tmp += sdata[threadIdx.x + 16];
//     sdata[threadIdx.x] = tmp;
//     __syncwarp();
//     tmp += sdata[threadIdx.x + 8];
//     sdata[threadIdx.x] = tmp;
//     __syncwarp();
//     tmp += sdata[threadIdx.x + 4];
//     sdata[threadIdx.x] = tmp;
//     __syncwarp();
//     tmp += sdata[threadIdx.x + 2];
//     sdata[threadIdx.x] = tmp;
//     __syncwarp();
//     tmp += sdata[threadIdx.x + 1];
//     sdata[threadIdx.x] = tmp;
//     // __syncwarp();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }

/**
 * @brief Kernel 6 -- Completely unrolled
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   uint32_t id_global =
//       (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
//   if ((id_global + blockDim.x) < n) {
//     sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
//   }
//   __syncthreads();
//   // do reduction in shared mem
//   if (blockDim.x == 1024) {
//     if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512];
//     __syncthreads();
//   }
//   if (blockDim.x >= 512) {
//     if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256];
//     __syncthreads();
//   }
//   if (blockDim.x >= 256) {
//     if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
//     __syncthreads();
//   }
//   if (blockDim.x >= 128) {
//     if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
//     __syncthreads();
//   }
//   // unroll the last warp
//   if (threadIdx.x < 32) {
//     sdata[threadIdx.x] += sdata[threadIdx.x + 32];
//     __syncwarp();
//     sdata[threadIdx.x] += sdata[threadIdx.x + 16];
//     __syncwarp();
//     sdata[threadIdx.x] += sdata[threadIdx.x + 8];
//     __syncwarp();
//     sdata[threadIdx.x] += sdata[threadIdx.x + 4];
//     __syncwarp();
//     sdata[threadIdx.x] += sdata[threadIdx.x + 2];
//     __syncwarp();
//     sdata[threadIdx.x] += sdata[threadIdx.x + 1];
//     __syncwarp();
//   }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }

// void test_asum(uint32_t power) {
//   size_t numElements = 1 << power;
//   size_t size_total = numElements * sizeof(float);
//   std::cout << "Elements count is:";
//   if (power >= 20) {
//     std::cout << (1 << (power % 20)) << "M";
//   } else if (power >= 10) {
//     std::cout << (1 << (power % 10)) << "K";
//   } else {
//     std::cout << numElements;
//   }
//   std::cout << std::endl;
//   // malloc and set intput vector
//   float *h_A = (float *)malloc(size_total);
//   for (int i = 0; i < numElements; ++i) {
//     h_A[i] = 1.0;
//   }
//   float *d_A = NULL;
//   checkCudaErrors(cudaMalloc((void **)&d_A, size_total));
//   checkCudaErrors(cudaMemcpy(d_A, h_A, size_total, cudaMemcpyHostToDevice));
//   // block/grid size and malloc output vector
//   uint32_t block_dim_x = MAX_BLOCK_THREADS;
//   if (numElements < block_dim_x) {
//     block_dim_x = ((numElements + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
//   }
//   uint32_t grid_dim_x = (numElements + MAX_BLOCK_THREADS - 1) /
//                         MAX_BLOCK_THREADS / EACH_BLOCK_SIZE_TIMES;
//   float *h_B = (float *)malloc(sizeof(float));
//   float *d_B = NULL;
//   // TODO: actually need large size. While small size could pass_also.
//   checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float) * grid_dim_x)));
//   // checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float))));
//   // bench mark
//   float duration_temp, duration_total{0.0};
//   cudaEvent_t start_g, stop_g;
//   checkCudaErrors(cudaEventCreate(&start_g));
//   checkCudaErrors(cudaEventCreate(&stop_g));
//   // kernel launch -- TODO: Don't need double?
//   uint32_t shared_mem_size =
//       block_dim_x * sizeof(float) * EACH_BLOCK_SIZE_TIMES;
//   dim3 blockDim(block_dim_x);
//   dim3 gridDim(grid_dim_x);
//   std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
//             << std::endl;
//   checkCudaErrors(cudaEventRecord(start_g));
//   asum_kernel<<<gridDim, blockDim, shared_mem_size>>>(numElements, d_A, 0, 0,
//                                                       d_B);
//   checkCudaErrors(cudaEventRecord(stop_g));
//   checkCudaErrors(cudaEventSynchronize(stop_g));
//   checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//   duration_total += duration_temp;
//   // Internal launch to avoid new blockDim.x exceed CUDA limitation.
//   if (grid_dim_x > (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES)) {
//     // std::cout << "Next kernel launch" << std::endl;
//     uint32_t n = grid_dim_x;
//     grid_dim_x =
//         (grid_dim_x + (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES) - 1) /
//         (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES);
//     gridDim = dim3(grid_dim_x);
//     std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" <<
//     gridDim.x
//               << std::endl;
//     checkCudaErrors(cudaEventRecord(start_g));
//     asum_kernel<<<gridDim, blockDim, shared_mem_size>>>(n, d_B, 0, 0, d_B);
//     checkCudaErrors(cudaEventRecord(stop_g));
//     checkCudaErrors(cudaEventSynchronize(stop_g));
//     checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//     duration_total += duration_temp;
//   }
//   if (grid_dim_x > 1) {
//     uint32_t n = grid_dim_x;
//     grid_dim_x =
//         (grid_dim_x + EACH_BLOCK_SIZE_TIMES - 1) / EACH_BLOCK_SIZE_TIMES;
//     blockDim = dim3(grid_dim_x);
//     shared_mem_size = grid_dim_x * sizeof(grid_dim_x) *
//     EACH_BLOCK_SIZE_TIMES; std::cout << "Block dim is:" << blockDim.x <<
//     std::endl; checkCudaErrors(cudaEventRecord(start_g)); asum_kernel<<<1,
//     blockDim, shared_mem_size>>>(n, d_B, 0, 0, d_B);
//     checkCudaErrors(cudaEventRecord(stop_g));
//     checkCudaErrors(cudaEventSynchronize(stop_g));
//     checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//     duration_total += duration_temp;
//   }
//   std::cout << "  duration: " << (duration_total * 1000) << " us" <<
//   std::endl;
//   // copy back
//   checkCudaErrors(cudaMemcpy(h_B, d_B, sizeof(float),
//   cudaMemcpyDeviceToHost));
//   // Free device and host memory
//   checkCudaErrors(cudaFree(d_A));
//   checkCudaErrors(cudaFree(d_B));
//   checkCudaErrors(cudaEventDestroy(start_g));
//   checkCudaErrors(cudaEventDestroy(stop_g));
//   float golden = 1.0 * numElements;
//   if (pointEqual(h_B[0], golden)) {
//     std::cout << "GPU compute PASS." << std::endl;
//   } else {
//     std::cout << "GPU compute FAIL. " << std::endl;
//     std::cout << "\tResult is :" << h_B[0] << ". Golden is:" << golden
//               << std::endl;
//   }
//   free(h_A);
//   free(h_B);
// }

/******************************************************************************/
/**
 * @brief Kernel 7 -- Complete unroll
 */
// __global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
//                             float *result, int blockSize) {
//   // statement needs constanct value. So need template.
//   extern __shared__ float sdata[];
//   // each thread loads one element from global to shared mem
//   uint32_t id_global =
//       (blockIdx.x * (blockSize * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
//   // all data once per while grid(including block)
//   // uint32_t grid_size = (blockSize * EACH_BLOCK_SIZE_TIMES) * gridDim.x;
//   uint32_t grid_size = EACH_BLOCK_SIZE_TIMES * blockDim.x;
//   sdata[id_global] = 0;
//   while (id_global < n) {
//     // if (blockIdx.x == 1) {
//     if (threadIdx.x == 0)
//       printf("id_global %d is %d.\r\n", blockIdx.x, id_global);
//     // }
//     // calculation multi cycles
//     if ((id_global + blockDim.x) < n) {
//       sdata[threadIdx.x] += x[id_global] + x[id_global + blockDim.x];
//     } else {
//       sdata[threadIdx.x] += x[id_global];
//     }

//     id_global += grid_size;
//   }
//   __syncthreads();
//   // do reduction in shared mem
//   if (blockDim.x == 1024) {
//     if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512];
//     __syncthreads();
//   }
//   if (blockDim.x >= 512) {
//     if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256];
//     __syncthreads();
//   }
//   if (blockDim.x >= 256) {
//     if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
//     __syncthreads();
//   }
//   if (blockDim.x >= 128) {
//     if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
//     __syncthreads();
//   }
//   // unroll the last warp
// if (threadIdx.x < 32) {
//   float tmp = sdata[threadIdx.x];
//   tmp += sdata[threadIdx.x + 32];
//   sdata[threadIdx.x] = tmp;
//   __syncwarp();
//   tmp += sdata[threadIdx.x + 16];
//   sdata[threadIdx.x] = tmp;
//   __syncwarp();
//   tmp += sdata[threadIdx.x + 8];
//   sdata[threadIdx.x] = tmp;
//   __syncwarp();
//   tmp += sdata[threadIdx.x + 4];
//   sdata[threadIdx.x] = tmp;
//   __syncwarp();
//   tmp += sdata[threadIdx.x + 2];
//   sdata[threadIdx.x] = tmp;
//   __syncwarp();
//   tmp += sdata[threadIdx.x + 1];
//   sdata[threadIdx.x] = tmp;
//   // __syncwarp();
// }
//   // write result of eac block to global mem
//   if (threadIdx.x == 0) {
//     result[blockIdx.x] = sdata[0];
//   }
//   __syncthreads();
// }

// void test_asum(uint32_t power) {
//   size_t numElements = 1 << power;
//   size_t size_total = numElements * sizeof(float);
//   std::cout << "Elements count is:";
//   if (power >= 20) {
//     std::cout << (1 << (power % 20)) << "M";
//   } else if (power >= 10) {
//     std::cout << (1 << (power % 10)) << "K";
//   } else {
//     std::cout << numElements;
//   }
//   std::cout << std::endl;
//   // malloc and set intput vector
//   float *h_A = (float *)malloc(size_total);
//   for (int i = 0; i < numElements; ++i) {
//     h_A[i] = 1.0;
//   }
//   float *d_A = NULL;
//   checkCudaErrors(cudaMalloc((void **)&d_A, size_total));
//   checkCudaErrors(cudaMemcpy(d_A, h_A, size_total, cudaMemcpyHostToDevice));
//   // block/grid size and malloc output vector
// uint32_t total_threads = numElements / (std::log2(numElements));
// uint32_t block_dim_x = MAX_BLOCK_THREADS;
// if (total_threads < block_dim_x) {
//   // manually calculate block_dim_x if there is few elements
//   block_dim_x = ((total_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
// }
// uint32_t grid_dim_x = (total_threads + MAX_BLOCK_THREADS - 1) /
//                       MAX_BLOCK_THREADS / EACH_BLOCK_SIZE_TIMES;
// uint32_t grid_size = block_dim_x * EACH_BLOCK_SIZE_TIMES * grid_dim_x;
// uint32_t calca_each_block = ((numElements + grid_size - 1) / grid_size);
// uint32_t block_size = calca_each_block * block_dim_x * EACH_BLOCK_SIZE_TIMES;
//    std::cout << "grid_size: " << grid_size
//             << ". calca_each: " << calca_each_block
//             << ". block_size: " << block_size << std::endl;
//   float *h_B = (float *)malloc(sizeof(float));
//   float *d_B = NULL;
//   // TODO: actually need large size. While small size could pass_also.
//   checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float) * grid_dim_x)));
//   // checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float))));
//   // bench mark
//   float duration_temp, duration_total{0.0};
//   cudaEvent_t start_g, stop_g;
//   checkCudaErrors(cudaEventCreate(&start_g));
//   checkCudaErrors(cudaEventCreate(&stop_g));
//   // kernel launch
//   uint32_t shared_mem_size = block_size * sizeof(float) *
//   EACH_BLOCK_SIZE_TIMES; std::cout << "Calculated shared_mem_size is: " <<
//   shared_mem_size
//             << std::endl;
//   if (shared_mem_size > MAX_SHARED_MEMORY_SIZE)
//     shared_mem_size = MAX_SHARED_MEMORY_SIZE;
//   dim3 blockDim(block_dim_x);
//   dim3 gridDim(grid_dim_x);
//   std::cout << "Total threads is:" << total_threads << std::endl;
//   std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
//             << std::endl;
//   checkCudaErrors(cudaEventRecord(start_g));
//   asum_kernel<<<gridDim, blockDim, shared_mem_size>>>(numElements, d_A, 0, 0,
//                                                       d_B, block_size);
//   checkCudaErrors(cudaEventRecord(stop_g));
//   checkCudaErrors(cudaEventSynchronize(stop_g));
//   checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//   duration_total += duration_temp;
//   // final launch
//   if (grid_dim_x > 1) {
//     uint32_t n = grid_dim_x;
//     grid_dim_x =
//         (grid_dim_x + EACH_BLOCK_SIZE_TIMES - 1) / EACH_BLOCK_SIZE_TIMES;
//     blockDim = dim3(grid_dim_x);
//     shared_mem_size = grid_dim_x * sizeof(grid_dim_x) *
//     EACH_BLOCK_SIZE_TIMES; std::cout << "Block dim is:" << blockDim.x <<
//     std::endl; checkCudaErrors(cudaEventRecord(start_g)); asum_kernel<<<1,
//     blockDim, shared_mem_size>>>(n, d_B, 0, 0, d_B);
//     checkCudaErrors(cudaEventRecord(stop_g));
//     checkCudaErrors(cudaEventSynchronize(stop_g));
//     checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));
//     duration_total += duration_temp;
//   }
//   std::cout << "  duration: " << (duration_total * 1000) << " us" <<
//   std::endl;
//   // copy back
//   checkCudaErrors(cudaMemcpy(h_B, d_B, sizeof(float),
//   cudaMemcpyDeviceToHost));
//   // Free device and host memory
//   checkCudaErrors(cudaFree(d_A));
//   checkCudaErrors(cudaFree(d_B));
//   checkCudaErrors(cudaEventDestroy(start_g));
//   checkCudaErrors(cudaEventDestroy(stop_g));
//   float golden = 1.0 * numElements;
//   if (pointEqual(h_B[0], golden)) {
//     std::cout << "GPU compute PASS." << std::endl;
//   } else {
//     std::cout << "GPU compute FAIL. " << std::endl;
//     std::cout << "\tResult is :" << h_B[0] << ". Golden is:" << golden
//               << std::endl;
//   }
//   free(h_A);
//   free(h_B);
// }

// Nvidia version on paper/blog.
// template <int N>
// __global__ void reduce_ws(float *gdata, float *out) {
//   // 32*32 = 1024. At most has 32 warps.
//   __shared__ float sdata[32];
//   int tid = threadIdx.x;
//   int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   float val = 0.0f;  // use tlr
//   unsigned mask = 0xFFFFFFFFU;
//   int lane = threadIdx.x % WARP_SIZE;
//   int warpID = threadIdx.x / WARP_SIZE;
//   while (idx < N) {  // grid stride loop to load
//     val += gdata[idx];
//     idx += gridDim.x * blockDim.x;
//   }
//   // 1st warp-shuffle reduction -- reduction in a warp
//   for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
//     val += __shfl_down_sync(mask, val, offset);
//   if (lane == 0) sdata[warpID] = val;  // store in warp_ID location
//   __syncthreads();                     // put warp results in shared mem
//   // hereafter, just warp 0.
//   if (warpID == 0) {
//     // reload val from shared mem if warp existed
//     val = (tid < blockDim.x / WARP_SIZE) ? sdata[lane] : 0;
//     // final warp-shuffle reduction
//     for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
//       val += __shfl_down_sync(mask, val, offset);
//     if (tid == 0) atomicAdd(out, val);
//   }
// }

/**
 * @brief Kernel 8 -- warp shuffle
 */
__global__ void asum_kernel(int n, const float *x, int shiftx, int incx,
                            float *result) {
  // 32*32 = 1024. At most has 32 warps.
  extern __shared__ float sdata[32];
  // each thread loads one element from global to shared mem
  int tid = threadIdx.x;
  int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
  float val = 0.0f;
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  unsigned mask = 0xFFFFFFFFU;
  while (id_global < n) {
    val += x[id_global];
    id_global += blockDim.x * gridDim.x;  // each block size.
  }
  //
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(mask, val, offset);
  if (lane_id == 0) sdata[warp_id] = val;  // store in warp_ID location
  __syncthreads();                         // put warp results in shared mem
  if (warp_id == 0) {
    // reload val from shared mem if warp existed
    val = (tid < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0;
    // final warp-shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      val += __shfl_down_sync(mask, val, offset);
    // do reduction cross blocks.
    if (tid == 0) atomicAdd(result, val);
  }
}

void test_asum(uint32_t power) {
  size_t numElements = 1 << power;
  size_t size_total = numElements * sizeof(float);
  std::cout << "Elements count is:";
  if (power >= 20) {
    std::cout << (1 << (power % 20)) << "M";
  } else if (power >= 10) {
    std::cout << (1 << (power % 10)) << "K";
  } else {
    std::cout << numElements;
  }
  std::cout << std::endl;
  // malloc and set intput vector
  float *h_A = (float *)malloc(size_total);
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = 1.0;
  }
  float *d_A = NULL;
  checkCudaErrors(cudaMalloc((void **)&d_A, size_total));
  checkCudaErrors(cudaMemcpy(d_A, h_A, size_total, cudaMemcpyHostToDevice));
  // block/grid size and malloc output vector
  uint32_t block_dim_x = MAX_BLOCK_THREADS;
  if (numElements < block_dim_x) {
    block_dim_x = ((numElements + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  }
  uint32_t grid_dim_x =
      (numElements + MAX_BLOCK_THREADS - 1) / MAX_BLOCK_THREADS / 16;
  float *h_B = (float *)malloc(sizeof(float));
  float *d_B = NULL;
  // TODO: actually need large size. While small size could pass_also.
  checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float) * grid_dim_x)));
  // checkCudaErrors(cudaMalloc((void **)&d_B, (sizeof(float))));
  // bench mark
  float duration_temp, duration_total{0.0};
  cudaEvent_t start_g, stop_g;
  checkCudaErrors(cudaEventCreate(&start_g));
  checkCudaErrors(cudaEventCreate(&stop_g));
  // kernel launch -- TODO: Don't need double?
  dim3 blockDim(block_dim_x);
  dim3 gridDim(grid_dim_x);
  std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
            << std::endl;
  checkCudaErrors(cudaEventRecord(start_g));
  asum_kernel<<<gridDim, blockDim>>>(numElements, d_A, 0, 0, d_B);
  checkCudaErrors(cudaEventRecord(stop_g));
  checkCudaErrors(cudaEventSynchronize(stop_g));
  checkCudaErrors(cudaEventElapsedTime(&duration_temp, start_g, stop_g));

  duration_total += duration_temp;
  std::cout << "  duration: " << (duration_total * 1000) << " us" << std::endl;
  // copy back
  checkCudaErrors(cudaMemcpy(h_B, d_B, sizeof(float), cudaMemcpyDeviceToHost));
  // Free device and host memory
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaEventDestroy(start_g));
  checkCudaErrors(cudaEventDestroy(stop_g));
  float golden = 1.0 * numElements;
  if (pointEqual(h_B[0], golden)) {
    std::cout << "GPU compute PASS." << std::endl;
  } else {
    std::cout << "GPU compute FAIL. " << std::endl;
    std::cout << "\tResult is :" << h_B[0] << ". Golden is:" << golden
              << std::endl;
  }
  free(h_A);
  free(h_B);
}

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
  checkCudaErrors(cudaDeviceGetCacheConfig(&pCacheCfg));
  std::cout << "Device CacheConfig is: " << pCacheCfg << std::endl;
}
int main() {
  getSharedMemoryCfg();
  // checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  // getSharedMemoryCfg();
  // all range for simple validataion.
  // for (uint32_t power = 10; power <= 25; ++power) {
  // specfic test case which could passed kernel-7
  // for (uint32_t power = 15; power <= 15; ++power) {
  // reference power number which is 4M elements as Mark Harris did.
  for (uint32_t power = 22; power <= 22; ++power) {
    std::cout << "---------------------------------"
              << "Power is:" << power << "  "
              << "---------------------------------" << std::endl;
    test_asum(power);
  }
  // getSharedMemoryCfg();

  return 0;
}
