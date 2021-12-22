#ifndef REDUCE_H__
#define REDUCE_H__

// RTX2060
#define MAX_BLOCK_SZ    1024     // total max per block.
#define MAX_BLOCK_DIM_X 1024
#define MAX_BLOCK_DIM_Y 1024
#define MAX_BLOCK_DIM_Z 64
#define MAX_GRID_DIM_X  2147483647  // 1<<31 -1. 0x7f ff ff ff
#define MAX_GRID_DIM_Y  65535
#define MAX_GRID_DIM_Z  65535

unsigned int gpu_sum_reduce(unsigned int* d_in, unsigned int d_in_len, unsigned long long* time_total);

#endif // !REDUCE_H__


