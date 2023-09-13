

#include <stdlib.h>

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

// cuda headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// internel
#include "reduction.h"
#include "utils.h"
#include "utilsm.h"

/**
 * @brief Kernel 1 -- interleaved addressing
 */
__global__ void asum_kernel1(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  // extern __shared__ float sdata[block_size];
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  // shared memory is only used in a block
  uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (id_global < n) sdata[threadIdx.x] = x[id_global];
  __syncthreads();
  // do reduction in shared mem
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if (threadIdx.x % (2 * s) == 0) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}

/**
 * @brief Kernel 2 -- interleaved addressing without divergence
 */
__global__ void asum_kernel2(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (id_global < n) sdata[threadIdx.x] = x[id_global];
  __syncthreads();
  // do reduction in shared mem
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    uint32_t index = 2 * s * threadIdx.x;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}

/**
 * @brief Kernel 3 -- sequential addressing
 */
__global__ void asum_kernel3(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  uint32_t id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (id_global < n) sdata[threadIdx.x] = x[id_global];
  __syncthreads();
  // do reduction in shared mem
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}
/**
 * @brief test frame for kernel 1-3
 * @param power
 */
void test_asum0(const float *in, float *out, std::size_t elems) {
  // block/grid size and malloc output vector
  uint32_t block_dim_x = MAX_BLOCK_THREADS;
  if (elems < block_dim_x) {
    block_dim_x = CeilDiv<uint32_t>(elems, warp_size) * warp_size;
  }
  uint32_t grid_dim_x = CeilDiv<uint32_t>(elems, MAX_BLOCK_THREADS);
  // kernel launch
  uint32_t shared_mem_size = block_dim_x * sizeof(float);
  dim3 blockDim(block_dim_x);
  dim3 gridDim(grid_dim_x);
  // std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
  //           << std::endl;
  asum_kernel3<<<gridDim, blockDim, shared_mem_size>>>(elems, in, 0, 0, out);
  if (grid_dim_x > MAX_BLOCK_THREADS) {
    // std::cout << "Next kernel launch" << std::endl;
    uint32_t n = grid_dim_x;
    grid_dim_x = CeilDiv<uint32_t>(grid_dim_x, MAX_BLOCK_THREADS);
    gridDim = dim3(grid_dim_x);
    asum_kernel3<<<gridDim, blockDim, shared_mem_size>>>(n, out, 0, 0, out);
  }
  if (grid_dim_x > 1) {
    // std::cout << "Next kernel launch" << std::endl;
    shared_mem_size = grid_dim_x * sizeof(grid_dim_x);
    asum_kernel3<<<1, gridDim, shared_mem_size>>>(grid_dim_x, out, 0, 0, out);
  }
  return;
}

/******************************************************************************/
constexpr int EACH_BLOCK_SIZE_TIMES = 2;
/**
 * @brief Kernel 4 -- first add during load
 */
__global__ void asum_kernel4(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  // NOTE (LYM): blockDim.x*2 due to elements for a block is doubled.
  //          And shared_memory size is also doubled.
  uint32_t id_global =
      (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
  if ((id_global + blockDim.x) < n) {  // need update to the lastest value.
    sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
  }
  __syncthreads();
  // do reduction in shared mem
  for (int s = blockDim.x / EACH_BLOCK_SIZE_TIMES; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}
/**
 * @brief Kernel 5 -- Unroll the last warp
 */
__global__ void asum_kernel5(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  uint32_t id_global =
      (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
  if ((id_global + blockDim.x) < n) {
    sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
  }
  __syncthreads();
  // do reduction in shared mem
  for (int s = blockDim.x / EACH_BLOCK_SIZE_TIMES; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  // unroll the last warp
  if (threadIdx.x < 32) {
    float tmp = sdata[threadIdx.x];
    tmp += sdata[threadIdx.x + 32];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 16];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 8];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 4];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 2];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 1];
    sdata[threadIdx.x] = tmp;
    // __syncwarp();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}

/**
 * @brief Kernel 6 -- Completely unrolled
 */
__global__ void asum_kernel6(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  uint32_t id_global =
      (blockIdx.x * (blockDim.x * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
  if ((id_global + blockDim.x) < n) {
    sdata[threadIdx.x] = x[id_global] + x[id_global + blockDim.x];
  }
  __syncthreads();
  // do reduction in shared mem
  if (blockDim.x == 1024) {
    if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512];
    __syncthreads();
  }
  if (blockDim.x >= 512) {
    if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256];
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    __syncthreads();
  }
  // unroll the last warp
  if (threadIdx.x < 32) {
    sdata[threadIdx.x] += sdata[threadIdx.x + 32];
    __syncwarp();
    sdata[threadIdx.x] += sdata[threadIdx.x + 16];
    __syncwarp();
    sdata[threadIdx.x] += sdata[threadIdx.x + 8];
    __syncwarp();
    sdata[threadIdx.x] += sdata[threadIdx.x + 4];
    __syncwarp();
    sdata[threadIdx.x] += sdata[threadIdx.x + 2];
    __syncwarp();
    sdata[threadIdx.x] += sdata[threadIdx.x + 1];
    __syncwarp();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}

void test_asum1(const float *in, float *out, std::size_t elems) {
  // block/grid size and malloc output vector
  uint32_t block_dim_x = MAX_BLOCK_THREADS;
  if (elems < block_dim_x) {
    block_dim_x = CeilDiv<uint32_t>(elems, warp_size) * warp_size;
  }
  uint32_t grid_dim_x =
      CeilDiv<uint32_t>(elems, MAX_BLOCK_THREADS) / EACH_BLOCK_SIZE_TIMES;
  // kernel launch -- TODO: Don't need double?
  uint32_t shared_mem_size =
      block_dim_x * sizeof(float) * EACH_BLOCK_SIZE_TIMES;
  dim3 blockDim(block_dim_x);
  dim3 gridDim(grid_dim_x);
  // std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
  //           << std::endl;
  asum_kernel6<<<gridDim, blockDim, shared_mem_size>>>(elems, in, 0, 0, out);
  // Internal launch to avoid new blockDim.x exceed CUDA limitation.
  if (grid_dim_x > (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES)) {
    // std::cout << "Next kernel launch" << std::endl;
    uint32_t n = grid_dim_x;
    grid_dim_x =
        (grid_dim_x + (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES) - 1) /
        (MAX_BLOCK_THREADS * EACH_BLOCK_SIZE_TIMES);
    gridDim = dim3(grid_dim_x);
    // std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" <<
    // gridDim.x << std::endl;
    asum_kernel6<<<gridDim, blockDim, shared_mem_size>>>(n, out, 0, 0, out);
  }
  if (grid_dim_x > 1) {
    uint32_t n = grid_dim_x;
    grid_dim_x =
        (grid_dim_x + EACH_BLOCK_SIZE_TIMES - 1) / EACH_BLOCK_SIZE_TIMES;
    blockDim = dim3(grid_dim_x);
    shared_mem_size = grid_dim_x * sizeof(grid_dim_x) * EACH_BLOCK_SIZE_TIMES;
    // std::cout << "Block dim is:" << blockDim.x << std::endl;
    asum_kernel6<<<1, blockDim, shared_mem_size>>>(n, out, 0, 0, out);
  }
}

/******************************************************************************/
/**
 * @brief Kernel 7 -- Complete unroll
 */
__global__ void asum_kernel7(int n, const float *x, int shiftx, int incx,
                             float *result, int blockSize) {
  // statement needs constanct value. So need template.
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  uint32_t id_global =
      (blockIdx.x * (blockSize * EACH_BLOCK_SIZE_TIMES)) + threadIdx.x;
  // all data once per while grid(including block)
  // uint32_t grid_size = (blockSize * EACH_BLOCK_SIZE_TIMES) * gridDim.x;
  uint32_t grid_size = EACH_BLOCK_SIZE_TIMES * blockDim.x;
  sdata[id_global] = 0;
  while (id_global < n) {
    // if (blockIdx.x == 1) {
    // if (threadIdx.x == 0)
    //   printf("id_global %d is %d.\r\n", blockIdx.x, id_global);
    // }
    // calculation multi cycles
    if ((id_global + blockDim.x) < n) {
      sdata[threadIdx.x] += x[id_global] + x[id_global + blockDim.x];
    } else {
      sdata[threadIdx.x] += x[id_global];
    }

    id_global += grid_size;
  }
  __syncthreads();
  // do reduction in shared mem
  if (blockDim.x == 1024) {
    if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512];
    __syncthreads();
  }
  if (blockDim.x >= 512) {
    if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256];
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    __syncthreads();
  }
  // unroll the last warp
  if (threadIdx.x < 32) {
    float tmp = sdata[threadIdx.x];
    tmp += sdata[threadIdx.x + 32];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 16];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 8];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 4];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 2];
    sdata[threadIdx.x] = tmp;
    __syncwarp();
    tmp += sdata[threadIdx.x + 1];
    sdata[threadIdx.x] = tmp;
    // __syncwarp();
  }
  // write result of eac block to global mem
  if (threadIdx.x == 0) {
    result[blockIdx.x] = sdata[0];
  }
  __syncthreads();
}

void test_asum3(const float *in, float *out, std::size_t elems) {
  // block/grid size and malloc output vector
  uint32_t total_threads = elems / (std::log2(elems));
  uint32_t block_dim_x = MAX_BLOCK_THREADS;
  if (total_threads < block_dim_x) {
    // manually calculate block_dim_x if there is few elements
    block_dim_x = (CeilDiv<uint32_t>(total_threads, warp_size)) * warp_size;
  }
  uint32_t grid_dim_x = CeilDiv<uint32_t>(total_threads, MAX_BLOCK_THREADS) /
                        EACH_BLOCK_SIZE_TIMES;
  grid_dim_x = std::max<uint32_t>(grid_dim_x, 1);
  uint32_t grid_size = block_dim_x * EACH_BLOCK_SIZE_TIMES * grid_dim_x;
  uint32_t calca_each_block = CeilDiv<uint32_t>(elems, grid_size);
  uint32_t block_size = calca_each_block * block_dim_x * EACH_BLOCK_SIZE_TIMES;
  // std::cout << "grid_size: " << grid_size
  //           << ". calca_each: " << calca_each_block
  //           << ". block_size: " << block_size << std::endl;
  // kernel launch
  uint32_t shared_mem_size = block_size * sizeof(float) * EACH_BLOCK_SIZE_TIMES;
  // std::cout << "Calculated shared_mem_size is: " << shared_mem_size
  //           << std::endl;
  if (shared_mem_size > MAX_SHARED_MEMORY_SIZE) {
    shared_mem_size = MAX_SHARED_MEMORY_SIZE;
  }
  dim3 blockDim(block_dim_x);
  dim3 gridDim(grid_dim_x);
  // std::cout << "Total threads is:" << total_threads << std::endl;
  // std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
  //           << std::endl;
  asum_kernel7<<<gridDim, blockDim, shared_mem_size>>>(elems, in, 0, 0, out,
                                                       block_size);
  // final launch
  if (grid_dim_x > 1) {
    uint32_t n = grid_dim_x;
    grid_dim_x =
        (grid_dim_x + EACH_BLOCK_SIZE_TIMES - 1) / EACH_BLOCK_SIZE_TIMES;
    blockDim = dim3(grid_dim_x);
    shared_mem_size = grid_dim_x * sizeof(grid_dim_x) * EACH_BLOCK_SIZE_TIMES;
    // std::cout << "Block dim is:" << blockDim.x << std::endl;
    // TODO: parameter may be incorrect
    asum_kernel7<<<1, blockDim, shared_mem_size>>>(n, out, 0, 0, out,
                                                   block_size);
  }
}

// Nvidia version on paper/blog.
template <int N>
__global__ void reduce_ws(float *gdata, float *out) {
  // 32*32 = 1024. At most has 32 warps.
  __shared__ float sdata[32];
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  float val = 0.0f;  // use tlr
  unsigned mask = 0xFFFFFFFFU;
  int lane = threadIdx.x % warp_size;
  int warpID = threadIdx.x / warp_size;
  while (idx < N) {  // grid stride loop to load
    val += gdata[idx];
    idx += gridDim.x * blockDim.x;
  }
  // 1st warp-shuffle reduction -- reduction in a warp
  for (int offset = warp_size / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(mask, val, offset);
  if (lane == 0) sdata[warpID] = val;  // store in warp_ID location
  __syncthreads();                     // put warp results in shared mem
  // hereafter, just warp 0.
  if (warpID == 0) {
    // reload val from shared mem if warp existed
    val = (tid < blockDim.x / warp_size) ? sdata[lane] : 0;
    // final warp-shuffle reduction
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
      val += __shfl_down_sync(mask, val, offset);
    if (tid == 0) atomicAdd(out, val);
  }
}

constexpr uint32_t serial_kernel8 = 1;
/**
 * @brief Kernel 8 -- warp shuffle
 * @param n total elements
 */
__global__ void asum_kernel8(int n, const float *x, int shiftx, int incx,
                             float *result) {
  // 32*32 = 1024. At most has 32 warps.
  extern __shared__ float sdata[32];
  // each thread loads one element from global to shared mem
  uint32_t tid = threadIdx.x;
  uint32_t id_global = (blockIdx.x * blockDim.x) + tid;
  uint32_t lane_id = tid % warp_size;
  uint32_t warp_id = tid / warp_size;
  // uint32_t lane_id = tid & warp_size_mask;
  // uint32_t warp_id = tid >> warp_size_width;

  constexpr uint32_t mask = 0xFFFFFFFFU;

  // serial compute while load
  // float val = 0.f;
  // while (id_global < n) {
  //   val += x[id_global];
  //   id_global += blockDim.x * gridDim.x;  // each block size.
  // }
  // no serial compute
  float val = x[id_global];

  //
  for (int offset = warp_size / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(mask, val, offset);
  if (lane_id == 0) sdata[warp_id] = val;  // store in warp_ID location
  __syncthreads();                         // put warp results in shared mem
  if (warp_id == 0) {
    // reload val from shared mem if warp existed
    // 1024/32 = 32. max counts are in a warp size
    val = (tid < blockDim.x / warp_size) ? sdata[lane_id] : 0.f;
    // final warp-shuffle reduction
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
      val += __shfl_down_sync(mask, val, offset);
    // do reduction cross blocks.
    if (tid == 0) atomicAdd(result, val);
  }
}

void test_asum8(const float *in, float *out, std::size_t elems) {
  // block/grid size and malloc output vector
  uint32_t block_dim_x = MAX_BLOCK_THREADS;
  if (elems < block_dim_x) {
    block_dim_x = CeilDiv<std::size_t>(elems, warp_size) * warp_size;
  }
  uint32_t grid_dim_x =
      CeilDiv<std::size_t>(elems, MAX_BLOCK_THREADS) / serial_kernel8;
  // kernel launch
  dim3 blockDim(block_dim_x);
  dim3 gridDim(grid_dim_x);
#ifndef NDEBUG
  std::cout << "Block dim is:" << blockDim.x << ". Grid dim is:" << gridDim.x
            << std::endl;
#endif
  asum_kernel8<<<gridDim, blockDim>>>(elems, in, 0, 0, out);
  return;
}
