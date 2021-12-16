#include "simpletest.h"

#include <cmath>
#include <iostream>
#include <ctime>

#include "cuda.h"
#include "device_launch_parameters.h"

#include "utils.h"

/**
 * \brief test memory and thread id
 * \param globalMem -- GPU memory address
 * \return none 
 */
__global__ void thread_id_log(struct IDS* const globalMem,const unsigned short len){
	unsigned short id = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) 
		+ threadIdx.y * blockDim.x + threadIdx.x ;
	if(id < len){
		globalMem[id].tIdx = threadIdx.x;
		globalMem[id].bIdx = blockIdx.x;
	}
	return;
}

void gpu_test(int block_size,int grid_sz,struct IDS* const globalMem,const unsigned short len){
	thread_id_log<<<block_size, grid_sz, len * sizeof(struct IDS) >>>(globalMem,len);
}