#include <iostream>
#include <ctime>

#include "cuda.h"

#include "device_launch_parameters.h"
#include "utils.h"

#include "reduce.h"
#include "deviceinfo.h"
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
	double duration;

	// generate test
	gpu_test();
		
}
