#include <cmath>
#include <ctime>
#include <iostream>

#include "cuda.h"
#include <device_launch_parameters.h>
#include "simpletest.h"
#include "utils.h"

/**
 * \brief test memory and thread id
 * \param globalMem -- GPU memory address
 * \return none
 */
__global__ void thread_id_log(struct IDS* const globalMem,
                              const unsigned short len) {
  unsigned short id =
      (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
      threadIdx.y * blockDim.x + threadIdx.x;
  if (id < len) {
    globalMem[id].tIdx = threadIdx.x;
    globalMem[id].bIdx = blockIdx.x;
  }
  return;
}

void gpu_test(int block_size, int grid_sz, struct IDS* const globalMem,
              const unsigned short len) {
  thread_id_log<<<grid_sz, block_size, len * sizeof(struct IDS)>>>(globalMem,
                                                                   len);
}
