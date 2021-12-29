#include <iostream>
#include <iomanip>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "reduce.h"
#include "utilsm.h"

// success after add this comment -- should be VS issue.
#pragma comment(lib,"cuda.lib")
// #pragma comment(lib,"cudart.lib")
void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i % 255;
	}
}

unsigned int cpu_simple_sum(unsigned int* h_in, unsigned int h_in_len)
{
	unsigned int total_sum = 0;

	for (unsigned int i = 0; i < h_in_len; ++i)
	{
		total_sum = total_sum + h_in[i];
	}

	return total_sum;
}

/**
 * \brief 
 * \param len_pow the power of data length. base is 2. 2^(len_pow)
 * \return int 
 */
int main(void)
{
  //cuInit(0);		// must have this init
	unsigned long long time_start,duration_cpu,duration_gpu;
	unsigned int len_pow = 11;
	for (;len_pow < 17 ;len_pow++){
    std::cout << "---------------- len_pow is "<< len_pow <<" ----------------" << std::endl;
    // generate test data
    unsigned int vector_len = (unsigned int)1 << len_pow;
    std::cout << "vector length is " << vector_len << std::endl;
    unsigned int* input_vec = new unsigned int[vector_len];
    generate_input(input_vec, vector_len);

    // Do CPU sum for reference
    std::cout << "CPU computation:" << std::endl;
    time_start = get_time_us();
    unsigned int cpu_total_sum = cpu_simple_sum(input_vec, vector_len);
    duration_cpu = get_time_us() - time_start;
    std::cout << "  duration: ";
    print_time_us(duration_cpu);
    std::cout << "  result: " << std::hex <<cpu_total_sum
			<< std::resetiosflags(std::ios::hex) << std::endl;
    std::cout << std::endl;

    // GPU calculation
    std::cout << "GPU computation:" << std::endl;
    unsigned int gpu_total_sum = gpu_sum_reduce(input_vec, vector_len,&duration_gpu);
    std::cout << "  total duration: ";
    print_time_us(duration_gpu);
    std::cout << "  result: "  << std::hex << gpu_total_sum 
			<< std::resetiosflags(std::ios::hex) << std::endl;
    std::cout  << std::endl;

    delete[] input_vec;
    // acceleration rate
    std::cout << "Calculation Result -- ";
    if (cpu_total_sum == gpu_total_sum) {
      std::cout << "Pass" << std::endl;
    }
    else {
      std::cout << "Fail" << std::endl;
    }
    std::cout << "Acceleration Rate: " << std::resetiosflags(std::ios::fixed)
      << (double)duration_cpu / duration_gpu << " times" << std::endl;
    std::cout << std::endl;
	}
	return 0;
}
