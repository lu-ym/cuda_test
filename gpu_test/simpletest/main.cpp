#include <iostream>
#include <ctime>
#include <iomanip>

// success after add this comment -- should be VS issue.
#pragma comment(lib,"cuda.lib")

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
//#include "reduce.h"
#include "simpletest.h"


// #pragma comment(lib,"cudart.lib")
void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i;
	}
}

// #define TOTAL_THREADS	1000
#define TOTAL_THREADS	80

/**
 *  
 * \ref struct IDS
 * \code  struct IDS{
 *	unsigned short tIdx;
 * 	// unsigned short tIdy;
 *  unsigned short bIdx;
 *	// unsigned short bIdy;
 * };
 * \endcode 
 */
int main(){
	// Set up clock for timing comparisons
	std::clock_t start;
	double duration;

	struct IDS *ids = new struct IDS[TOTAL_THREADS];				// on host
	// unsigned int* addr_in;	// in GPU
	struct IDS* addr_in;	// in GPU
	//unsigned short TOTAL_THREADS;
	// ids = malloc(TOTAL_THREADS * sizeof(struct IDS));
	// if(ids == nullptr){
	// 	std::cout << "There is no enough memory on host. TOTAL_THREADS size is: "
	// 		<< (TOTAL_THREADS * sizeof(struct IDS)) << std::endl;
	// }

	// Set up device-side memory for input
	checkCudaErrors(cudaMalloc(&addr_in, TOTAL_THREADS * sizeof(struct IDS)));
	// generate CUDA parameters and call cuda device functions
	// unsigned int block_sz = MAX_BLOCK_SZ; 
	// unsigned int grid_sz = MAX_GRID_SZ;
	gpu_test(64,64,addr_in,TOTAL_THREADS);	// generate ID lists

	checkCudaErrors(cudaMemcpy(ids, addr_in, TOTAL_THREADS * sizeof(struct IDS), cudaMemcpyDeviceToHost));
	// std::cout << "Threads No:{blockIdX;bIdY;threadIdx,tIdY}" << std::flush;
	std::cout << "Threads No:{blockIdX;threadIdx}" << std::flush;
	for(int a = 0; a < TOTAL_THREADS; a++){
		if(!(a%8)) {
			std::cout << std::endl << std::setw(4) << a << ": " << std::flush;
		}
		// std::cout << "{" <<setw(4) << ids[a].bIdx << ";" << setw(4) << ids[a].bIdy 
			// << ";"<<setw(4) << ids[a].tIdx << ";" << setw(4) << ids[a].tIdy << "} "<< std::flush;
		std::cout << "{" << std::setw(4) << ids[a].bIdx << ";" << std::setw(4) << ids[a].tIdx <<
			"} "<< std::flush;
	}
	std::cout << std::endl;
	// free(ids);
	checkCudaErrors(cudaFree(addr_in));
}

