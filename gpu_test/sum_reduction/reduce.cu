#include "reduce.h"

#include <cmath>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "utilsm.h"

/**
 * @brief rough implementation
 *				all data W/R operation are on global memory
 * @param d_in 
 * @param d_in_len 
 * @return  
 */
__global__ void reduce_base(unsigned int* d_in, unsigned short d_in_len){
	d_in_len = d_in_len / 2;
  unsigned int tid;
	while(d_in_len > 0){
    tid = threadIdx.x + blockDim.x * blockIdx.x;
		if (tid < d_in_len){
			d_in[tid] += d_in[tid + d_in_len];
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
unsigned int gpu_sum_reduce(unsigned int* hostMem, unsigned int vector_len,unsigned long long *time_total ){
	unsigned int* globalMem;
	unsigned long long time_start;
  unsigned long long duration_mmalc,duration_cp1, duration_cal, duration_cp2, duration_mfree;

	// copy data to device -- includes memory malloc time
	time_start = get_time_us();
	checkCudaErrors(cudaMalloc(&globalMem, sizeof(unsigned int) * vector_len));
	duration_mmalc = get_time_us() - time_start;
	std::cout << "cudaMalloc time is: ";
  print_time_us(duration_mmalc);
  time_start = get_time_us();
	checkCudaErrors(cudaMemcpy(globalMem,hostMem , sizeof(unsigned int) * vector_len, 
		cudaMemcpyHostToDevice));
	duration_cp1 = get_time_us() - time_start;
	bandwidth_print(duration_cp1, *hostMem, vector_len);
	// calculation
  unsigned int block_sz = vector_len / 2;
  unsigned int grid_sz = 1;
	if (vector_len > MAX_BLOCK_SZ) {
		block_sz = MAX_BLOCK_SZ;
		grid_sz = vector_len / MAX_BLOCK_SZ;
		if (vector_len % MAX_BLOCK_SZ) grid_sz += 1;
	}
	std::cout << "block_sz is " << block_sz << "  grid_sz is " << grid_sz << std::endl;
	time_start = get_time_us();
	reduce_base <<<grid_sz , block_sz, 2 * sizeof(unsigned int) * vector_len >>>
							(globalMem, vector_len);
	duration_cal = get_time_us() - time_start;
	std::cout << "  calculation duration is:";
  print_time_us(duration_cal);
  // copy back to host -- ignore due to we only need sum
	time_start = get_time_us();
	checkCudaErrors(cudaMemcpy(hostMem, globalMem, sizeof(unsigned int),
		cudaMemcpyDeviceToHost));
	duration_cp2 = get_time_us() - time_start;
  std::cout << "  Copy back " << std::endl;
  bandwidth_print(duration_cp2, *hostMem, sizeof(unsigned int)); 
	time_start = get_time_us();
  checkCudaErrors(cudaFree(globalMem));
	duration_mfree = get_time_us() - time_start;
  std::cout << "  free time is: ";
  print_time_us(duration_mfree);
	// return globalMem[0];	// Must not use host memory rather than Device memory!!!
	*time_total = duration_mmalc + duration_cp1 + duration_cal + duration_cp2 + duration_mfree;
	return hostMem[0];
}

