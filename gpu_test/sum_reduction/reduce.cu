#include "reduce.h"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"



__global__ void reduce_base(unsigned int* d_in, unsigned int d_in_len){
	while(threadIdx.x < d_in_len){
		d_in_len = d_in_len/2;
		unsigned int id_0 = threadIdx.x; 
		unsigned int id_1 = threadIdx.x + d_in_len;
		d_in[id_0] += d_in[id_1];
		__syncthreads();
		if(d_in_len == 1) return;
	}
	// return;
}

/**
 * \brief test reduce algrithm.
 * \param d_in 
 * \param d_in_len must be the power of 2. not a generate function.
 * \return void 
 */
unsigned int gpu_sum_reduce(unsigned int* globalMem, unsigned int d_in_len){
	unsigned int block_sz = MAX_BLOCK_SZ;
	unsigned int grid_sz = MAX_BLOCK_SZ;
	// TODO failed. may be could not return GPU memory???
	reduce_base<<<block_sz,grid_sz,2*sizeof(unsigned int)*d_in_len>>>(globalMem,d_in_len);
	return d_in[0];
}

