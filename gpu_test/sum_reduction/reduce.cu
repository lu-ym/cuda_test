#include "reduce.h"

#include <cmath>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "utilsm.h"

__global__ void reduce_base(unsigned int* d_in, unsigned short d_in_len){
	d_in_len = d_in_len / 2;
	while(d_in_len > 0){
		if (threadIdx.x < d_in_len){
			//unsigned int id_0 = threadIdx.x; 
			unsigned int id_1 = threadIdx.x + d_in_len;
			d_in[threadIdx.x] += d_in[id_1];
			// d_in[threadIdx.x] = d_in[threadIdx.x] + d_in[id_1];
		}
		__syncthreads();
		d_in_len = d_in_len / 2;
	}
}

/**
 * \brief test reduce algorithm.
 * \param d_in 
 * \param d_in_len must be the power of 2. not a generate purpose function.
 *				should <= MAX_BLOCK_SZ
 * \return 
 */
unsigned int gpu_sum_reduce(unsigned int* hostMem, unsigned int vector_len){
	unsigned int* globalMem;
	unsigned long long time_start, duration;

	// copy data to device
	time_start = get_time_us();
	checkCudaErrors(cudaMalloc(&globalMem, sizeof(unsigned int) * vector_len));
	checkCudaErrors(cudaMemcpy(globalMem,hostMem , sizeof(unsigned int) * vector_len, 
		cudaMemcpyHostToDevice));
	duration = get_time_us() - time_start;
	bandwidth_print(duration, *hostMem, vector_len);
	// calculation
	unsigned int block_sz = vector_len / 2;
	unsigned int grid_sz = 1;
	time_start = get_time_us();
	reduce_base <<<grid_sz , block_sz, 2 * sizeof(unsigned int) * vector_len >>>
							(globalMem, vector_len);
	duration = get_time_us() - time_start;
	std::cout << "  calculation duration is:";
  print_time_us(duration);
  // copy back to host -- ignore due to we only need sum
	checkCudaErrors(cudaMemcpy(hostMem, globalMem, sizeof(unsigned int),
		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(globalMem));
	
	// return globalMem[0];	// Must not use host memory rather than Device memory!!!
	return hostMem[0];
}

