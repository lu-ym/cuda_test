#include <ctime>
#include <iomanip>
#include <iostream>

// success after add this comment -- should be VS issue.
// #pragma comment(lib,"cuda.lib")

#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utils.h"
#include "utilsm.h"

// #define MAX_THREADS 8192

void gen_matrix(float **mat, const unsigned int row_num,
                const unsigned int column_num) {
    std::srand(std::time(nullptr));
    for (unsigned int i = 0; i < row_num; ++i) {
        for (unsigned int j = 0; j < column_num; ++j) {
            mat[i][j] =
                static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }
    }
}
void reset_matrix(float **mat, const unsigned int row_num,
                  const unsigned int column_num) {
    for (unsigned int i = 0; i < row_num; ++i) {
        memset(mat[i], 0, column_num * sizeof(float));
        // for (unsigned int j = 0; j < column_num; ++j) {
        // mat[i][j] = 0.0;
        // }
    }
}
bool compare_matrix(float **const mat1, float **const mat2,
                    const unsigned int row_num, const unsigned int column_num) {
    bool result = true;
    for (unsigned int i = 0; i < row_num; ++i) {
        for (unsigned int j = 0; j < column_num; ++j) {
            if (!pointEqual(mat1[i][j], mat2[i][j])) {
                result = false;
                std::cout << "Data mismatch at " << i << "/" << j << ":";
                std::cout << "\t while data: " << std::fixed << mat1[i][j]
                          << " vs " << std::fixed << mat2[i][j] << std::endl;
                break;
            }
        }
        if (!result) break;
    }
    return result;
}
void print_matrix(float **const matrix, const unsigned int row_start,
                  const unsigned int row_end, const unsigned int col_start,
                  const unsigned int col_end) {
    for (unsigned int i = row_start; i < row_end; ++i) {
        std::cout << "Row/Column " << std::setw(4) << i << "/" << col_start
                  << ":";
        for (unsigned int j = col_start; j < col_end; ++j) {
            std::cout << " " << std::fixed << matrix[i][j];
        }
        std::cout << std::endl;
    }
}

/**
 * @brief basic revision. no optimization.
 */
void cpu_gemm_0(float **const mat1, float **const mat2, float **mat_prod,
                const unsigned int m, const unsigned int n,
                const unsigned int k) {
    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int b = 0; b < n; ++b) {
            for (unsigned int c = 0; c < k; ++c) {
                mat_prod[a][c] += mat1[a][b] * mat2[b][c];
            }
        }
    }
}
/**
 * @brief CPU revision 1.  mat1 and mat_prd get good spatial locality.
 *          mat_prod gets good temporal locality.
 */
void cpu_gemm_1(float **const mat1, float **const mat2, float **mat_prod,
                const unsigned int m, const unsigned int n,
                const unsigned int k) {
    float temp_prod;
    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int c = 0; c < k; ++c) {
            temp_prod = 0.0;
            for (unsigned int b = 0; b < n; ++b) {
                temp_prod += mat1[a][b] * mat2[b][c];
            }
            mat_prod[a][c] = temp_prod;
        }
    }
}

#define TILE_LENGTH 16
/**
 * @brief CPU revision 2. Tiling.
 *          Compiler should already implemente this policy?
 *      Reference:_Using Blocking to Increase Temporal Locality.pdf
 */
void cpu_gemm_2(float **const mat1, float **const mat2, float **mat_prod,
                const unsigned int m, const unsigned int n,
                const unsigned int k) {
    float temp_prod;
    for (unsigned int b = 0; b < n; b += TILE_LENGTH) {
        for (unsigned int c = 0; c < k; c += TILE_LENGTH) {
            for (unsigned int a = 0; a < m; ++a) {
                for (unsigned int c1 = c; c1 < (c + TILE_LENGTH); ++c1) {
                    temp_prod = mat_prod[a][c1];
                    for (unsigned int b1 = b; b1 < (b + TILE_LENGTH); ++b1) {
                        temp_prod += mat1[a][b1] * mat2[b1][c1];
                    }
                    mat_prod[a][c1] = temp_prod;
                }
            }
        }
    }
}
/**
 * Basic version: use 2d block
 */
__global__ void gpu_gemm(float **const mat1, float **const mat2,
                         float **mat_prod, const unsigned int m,
                         const unsigned int n, const unsigned int k) {
    float temp_prod = 0.0f;
    unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    for (unsigned int b = 0; b < n; ++b) {
        temp_prod += mat1[tidx][b] * mat2[b][tidy];
    }
    mat_prod[tidx][tidy] = temp_prod;
}
/**
 * Profiling method: 1) Tiling; 2) Shared memory 3) Row/column major memory?
 */
__global__ void gpu_gemm_2(float **const mat1, float **const mat2,
                           float **mat_prod, const unsigned int m,
                           const unsigned int n, const unsigned int k) {
    float temp_prod;
}
int main_old() {
    // Set up clock for timing comparisons
    std::clock_t start;
    float duration_cpu, duration_cpu2, duration_cpu3, duration_gpu;
    unsigned int len_pow = 4;
    bool result;
    // const unsigned int max_length_per_block = 32;

    for (; len_pow < 10; len_pow++) {
        std::cout << "---------------- len_pow is " << len_pow
                  << " ----------------" << std::endl;
        unsigned int m = (unsigned int)1 << len_pow;
        unsigned int n = m;
        unsigned int k = n;
        std::cout << "matrix size M/N/K are all " << m << std::endl;

        float **mat1 = new float *[m]; // on host
        float **mat2 = new float *[n];
        float **mat_prod1 = new float *[m];
        float **mat_prod2 = new float *[m];
        for (unsigned int i = 0; i < m; ++i) {
            mat1[i] = new float[n];
            mat_prod1[i] = new float[k];
            mat_prod2[i] = new float[k];
        }
        for (unsigned int i = 0; i < n; ++i) {
            mat2[i] = new float[k];
        }

        // generate input
        gen_matrix(mat1, m, n);
        gen_matrix(mat2, n, k);
        // print_matrix(mat1, 0, m, 0, n);
        // print_matrix(mat2, 0, n, 0, k);

        // prepare data on GPU
        float **mat1_g, **mat2_g, **mat_pro_g;
        float **temp_p1 = new float *[m];
        float **temp_p2 = new float *[m];
        float **temp_pp = new float *[m];
        unsigned int row_size = m * sizeof(float);
        unsigned int row_pointer_size = m * sizeof(float *);
        checkCudaErrors(cudaMalloc((void ***)&mat1_g, row_pointer_size));
        for (unsigned int i = 0; i < m; ++i) {
            checkCudaErrors(cudaMalloc((void **)&(temp_p1[i]), row_size));
            checkCudaErrors(cudaMemcpy(temp_p1[i], mat1[i], row_size,
                                       cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMemcpy(mat1_g, temp_p1, row_pointer_size,
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void ***)&mat2_g, row_pointer_size));
        for (unsigned int i = 0; i < m; ++i) {
            checkCudaErrors(cudaMalloc((void **)&(temp_p2[i]), row_size));
            checkCudaErrors(cudaMemcpy(temp_p2[i], mat2[i], row_size,
                                       cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMemcpy(mat2_g, temp_p2, row_pointer_size,
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((float ***)&mat_pro_g, row_pointer_size));
        for (unsigned int i = 0; i < m; ++i) {
            checkCudaErrors(cudaMalloc((void **)&(temp_pp[i]), row_size));
        }
        checkCudaErrors(cudaMemcpy(mat_pro_g, temp_pp, row_pointer_size,
                                   cudaMemcpyHostToDevice));

        // CPU computation
        reset_matrix(mat_prod1, m, k);
        start = std::clock();
        cpu_gemm_0(mat1, mat2, mat_prod1, m, n, k);
        // std::cout << "CLOCK_PER_MILLSEC is " << CLOCK_PER_MILLSEC <<
        // std::endl;
        duration_cpu = static_cast<float>(std::clock() - start) /
                       static_cast<float>(CLOCK_PER_MILLSEC);
        std::cout << "CPU computation:" << std::endl;
        std::cout << "  duration: " << duration_cpu << " ms" << std::endl;
        print_matrix(mat_prod1, 0, 1, 0, 10);

        // // CPU computation 2
        // reset_matrix(mat_prod2, m, k);
        // start = std::clock();
        // cpu_gemm_1(mat1, mat2, mat_prod2, m, n, k);
        // duration_cpu2 = static_cast<float>(std::clock() - start) /
        //                 static_cast<float>(CLOCK_PER_MILLSEC);
        // std::cout << "CPU computation:" << std::endl;
        // std::cout << "  duration: " << duration_cpu2 << " ms" << std::endl;
        // print_matrix(mat_prod2, 0, 1, 0, 10);
        // result = compare_matrix(mat_prod1, mat_prod2, m, k);
        // std::cout << "Calculation Result -- " << std::boolalpha << result
        //           << std::endl;
        // std::cout << "Acceleration Rate: " <<
        // std::resetiosflags(std::ios::fixed)
        //           << duration_cpu / duration_cpu2 << " times" << std::endl;

        // // CPU version 3
        // reset_matrix(mat_prod2, m, k);
        // start = std::clock();
        // cpu_gemm_2(mat1, mat2, mat_prod2, m, n, k);
        // duration_cpu3 = static_cast<float>(std::clock() - start) /
        //                 static_cast<float>(CLOCK_PER_MILLSEC);
        // std::cout << "CPU computation:" << std::endl;
        // std::cout << "  duration: " << duration_cpu3 << " ms" << std::endl;
        // print_matrix(mat_prod2, 0, 1, 0, 10);
        // result = compare_matrix(mat_prod1, mat_prod2, m, k);
        // std::cout << "Calculation Result -- " << std::boolalpha << result
        //           << std::endl;
        // std::cout << "Acceleration Rate: " <<
        // std::resetiosflags(std::ios::fixed)
        //           << duration_cpu / duration_cpu3 << " times" << std::endl;

        // GPU part
        reset_matrix(mat_prod2, m, k);
        dim3 block_dim;
        dim3 grid_dim;
        if (m > 32) {
            block_dim = dim3(32, 32, 1);
            grid_dim = dim3((m + 31) / 32, (m + 31) / 32, 1);
        } else {
            block_dim = dim3(m, m, 1);
            grid_dim = dim3(1, 1, 1);
        }
        cudaEvent_t start_g, stop_g;
        checkCudaErrors(cudaEventCreate(&start_g));
        checkCudaErrors(cudaEventCreate(&stop_g));
        checkCudaErrors(cudaEventRecord(start_g));
        gpu_gemm<<<grid_dim, block_dim>>>(mat1_g, mat2_g, mat_pro_g, m, n, k);
        checkCudaErrors(cudaEventRecord(stop_g));
        checkCudaErrors(cudaEventSynchronize(stop_g));
        checkCudaErrors(cudaEventElapsedTime(&duration_gpu, start_g, stop_g));
        std::cout << "GPU computation:" << std::endl;
        std::cout << "  duration: " << duration_gpu << "ms" << std::endl;
        checkCudaErrors(cudaEventDestroy(start_g));
        checkCudaErrors(cudaEventDestroy(stop_g));
        for (unsigned int i = 0; i < m; ++i) {
            checkCudaErrors(cudaMemcpy(mat_prod2[i], temp_pp[i], row_size,
                                       cudaMemcpyDeviceToHost));
        }
        print_matrix(mat_prod2, 0, 1, 0, 10);
        result = compare_matrix(mat_prod1, mat_prod2, m, k);
        std::cout << "Calculation Result -- " << std::boolalpha << result
                  << std::endl;
        std::cout << "Acceleration Rate: "
                  << std::resetiosflags(std::ios::fixed)
                  << duration_cpu / duration_gpu << " times" << std::endl;

        for (unsigned int i = 0; i < m; ++i) {
            delete[] mat1[i];
            delete[] mat_prod1[i];
            delete[] mat_prod2[i];
            delete[] mat2[i];
            checkCudaErrors(cudaFree(temp_p1[i]));
            checkCudaErrors(cudaFree(temp_p2[i]));
            checkCudaErrors(cudaFree(temp_pp[i]));
        }
        delete[] mat1;
        delete[] mat2;
        delete[] mat_prod1;
        delete[] mat_prod2;
        delete temp_p1;
        delete temp_p2;
        delete temp_pp;
        checkCudaErrors(cudaFree(mat1_g));
        checkCudaErrors(cudaFree(mat2_g));
        checkCudaErrors(cudaFree(mat_pro_g));
    }

    return 0;
}
