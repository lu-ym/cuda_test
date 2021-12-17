#include <iostream>
#include <ctime>

#include "cuda.h"

#include "device_launch_parameters.h"
#include "utils.h"

#include "reduce.h"

// success after add this comment -- should be VS issue.
#pragma comment(lib,"cuda.lib")
// #pragma comment(lib,"cudart.lib")
void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i;
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
 * @brief sum reduction
 * 
 * @return int 
 */
int main()
{
	// Set up clock for timing comparisons
	std::clock_t start;
	double duration_cpu;
	double duration_gpu;
	// generate test data
	unsigned int vector_len = 1024;
	unsigned int *input_vec= new unsigned int[vector_len];
	generate_input(input_vec,vector_len);

	// Do CPU sum for reference
	start = std::clock();
	unsigned int cpu_total_sum = cpu_simple_sum(input_vec, vector_len);
	// duration_cpu = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	duration_cpu = (std::clock() - start);
	std::cout << cpu_total_sum << std::endl;
	// std::cout << "CPU time: " << duration_cpu << " s" << std::endl;
	std::cout << "CPU time: " << duration_cpu << " ms" << std::endl;

	// Set up device-side memory for input
	unsigned int* d_in;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * vector_len));
	checkCudaErrors(cudaMemcpy(d_in, input_vec, sizeof(unsigned int) * vector_len, cudaMemcpyHostToDevice));
	start = std::clock();
	unsigned int gpu_total_sum = gpu_sum_reduce(d_in,vector_len);
	duration_gpu = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	// duration_gpu = (std::clock() - start);
	std::cout << cpu_total_sum << std::endl;
	std::cout << "GPU time: " << duration_gpu << " s" << std::endl;
	// std::cout << "CPU time: " << duration_gpu << " ms" << std::endl;

	checkCudaErrors(cudaFree(d_in));
	delete[] input_vec;
	// acceleration rate
	// std::cout << "Calculation result is" << lambda(cpu_total_sum == gpu_total_sum) ? ("Pass") : ("Fail") << std::endl;
	std::cout << "Calculation result is"; 
	if(cpu_total_sum == gpu_total_sum) 	std::cout << "Pass"<< std::endl;
	else 	std::cout << "Fail" << std::endl;
	std::cout << "Accelaration Rate is " << duration_gpu / duration_cpu << " times" << std::endl;
	return 0;
}
