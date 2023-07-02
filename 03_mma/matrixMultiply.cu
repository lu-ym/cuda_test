#include <stdlib.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

// #include "cublas_v2.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "helper_cuda.h"

#include "utils.h"
#include "utilsm.h"

bool compare_matrix(float *const mat1, float *const mat2,
                    const unsigned int row_num, const unsigned int column_num) {
  for (unsigned int i = 0; i < row_num; ++i) {
    for (unsigned int j = 0; j < column_num; ++j) {
      if (!pointEqual(mat1[i * column_num + j], mat2[i * column_num + j])) {
        std::cout << "Data mismatch at " << i << "/" << j << ":";
        std::cout << "while data: " << std::fixed << mat1[i * column_num + j]
                  << " vs " << std::fixed << mat2[i * column_num + j]
                  << std::endl;
        std::cout << "\t diff: " << std::setprecision(2)
                  << (100 *
                      std::abs(mat1[i * column_num + j] -
                               mat2[i * column_num + j]) /
                      std::abs(mat1[i * column_num + j]))
                  << " %" << std::endl;
        return false;
      }
    }
  }
  return true;
}

void transposeMatrix(const float *orig, float *reshaped,
                     unsigned int side_len) {
  for (unsigned int i = 0; i < side_len; i++) {
    for (unsigned int j = 0; j < side_len; ++j) {
      reshaped[j * side_len + i] = orig[i * side_len + j];
    }
  }
}

void print_matrix(float *const matrix, const unsigned int row_start,
                  const unsigned int row_end, const unsigned int col_start,
                  const unsigned int col_end, const unsigned int column_num) {
  for (unsigned int i = row_start; i < row_end; ++i) {
    std::cout << "Row/Column " << std::setw(4) << i << "/" << col_start << ":";
    for (unsigned int j = col_start; j < col_end; ++j) {
      std::cout << " " << std::fixed << matrix[i * column_num + j];
    }
    std::cout << std::endl;
  }
}

/** Copy from simpleCUBLAS.cpp of cuda samples. **/
// Column major? -- Yes. CuBlas use column major.
/* Host implementation of a simple version of sgemm */
// static void simple_sgemm(int n, float alpha, const float *A, const float *B,
//                          float beta, float *C) {
//   int i;
//   int j;
//   int k;

//   for (i = 0; i < n; ++i) {
//     for (j = 0; j < n; ++j) {
//       float prod = 0;
//       for (k = 0; k < n; ++k) {
//         prod += A[k * n + i] * B[j * n + k];
//       }
//       C[j * n + i] = alpha * prod + beta * C[j * n + i];
//     }
//   }
// }

/**
 * @brief basic revision. no optimization.
 */
// void cpu_gemm_0(const unsigned int m, const unsigned int n,
//                 const unsigned int k, const float alpha, float *const mat1,
//                 float *const mat2, const float beta, float *mat3) {
//   for (unsigned int a = 0; a < m; ++a) {
//     for (unsigned int b = 0; b < n; ++b) {
//       for (unsigned int c = 0; c < k; ++c) {
//         mat_prod[a][c] += mat1[a][b] * mat2[b][c];
//       }
//     }
//   }
// }
/**
 * @brief CPU revision 1.  mat1 and mat_prd get good spatial locality.
 *          mat3 gets good temporal locality.
 */
void cpu_gemm_1(const unsigned int m, const unsigned int n,
                const unsigned int k, const float alpha, float *const mat1,
                float *const mat2, const float beta, float *mat3) {
  float temp_prod;
  for (unsigned int a = 0; a < m; ++a) {
    for (unsigned int c = 0; c < k; ++c) {
      temp_prod = 0.0;
      for (unsigned int b = 0; b < n; ++b) {
        temp_prod += mat1[b * m + a] * mat2[c * n + b];
      }
      mat3[c * m + a] = alpha * temp_prod + beta * mat3[c * m + a];
    }
  }
}

#define TILE_LENGTH 16
/**
 * @brief CPU revision 2. Tiling.
 *          Compiler should already implemente this policy?
 *      Reference:_Using Blocking to Increase Temporal Locality.pdf
 */
// void cpu_gemm_2(float **const mat1, float **const mat2, float **mat_prod,
//                 const unsigned int m, const unsigned int n,
//                 const unsigned int k) {
//   float temp_prod;
//   for (unsigned int b = 0; b < n; b += TILE_LENGTH) {
//     for (unsigned int c = 0; c < k; c += TILE_LENGTH) {
//       for (unsigned int a = 0; a < m; ++a) {
//         for (unsigned int c1 = c; c1 < (c + TILE_LENGTH); ++c1) {
//           temp_prod = mat_prod[a][c1];
//           for (unsigned int b1 = b; b1 < (b + TILE_LENGTH); ++b1) {
//             temp_prod += mat1[a][b1] * mat2[b1][c1];
//           }
//           mat_prod[a][c1] = temp_prod;
//         }
//       }
//     }
//   }
// }
/**
 * @brief CPU revision 3. Adapt to new pointer allocation mode.
 */
// void cpu_gemm_3(const unsigned int m, const unsigned int n,
//                 const unsigned int k, const float alpha, float *const mat1,
//                 float *const mat2, const float beta, float *mat3) {
//   float temp_prod;
//   for (unsigned int b = 0; b < n; b += TILE_LENGTH) {
//     for (unsigned int c = 0; c < k; c += TILE_LENGTH) {
//       for (unsigned int a = 0; a < m; ++a) {
//         for (unsigned int c1 = c; c1 < (c + TILE_LENGTH); ++c1) {
//           temp_prod = mat3[a * m + c1];
//           for (unsigned int b1 = b; b1 < (b + TILE_LENGTH); ++b1) {
//             temp_prod += mat1[a * m + b1] * mat2[b1 * n + c1];
//           }
//           // TODO: Due to tiling, below calculation is wrong
//           mat3[a * m + c1] = alpha * temp_prod + beta * mat3[a * m + c1];
//         }
//       }
//     }
//   }
// }
/**
 * Basic version: use 2d block. 1D grid dim.
 */
__global__ void gpu_gemm(const unsigned int m, const unsigned int n,
                         const unsigned int k, const float alpha,
                         const float *mat1, const float *mat2, const float beta,
                         float *mat3) {
  float temp_prod = 0.0f;
  unsigned int tidx = threadIdx.x * n + blockIdx.x;
  for (unsigned int b = 0; b < n; ++b) {
    temp_prod += mat1[b * n + blockIdx.x] * mat2[b + threadIdx.x * n];
  }
  mat3[tidx] = alpha * temp_prod + beta * mat3[tidx];
  // mat3[tidx] = temp_prod;
}
/**
 * Profiling method:
 * 1) Tiling -- take full use of L1/Shared memory
 * 2) Shared memory -- Turing L1/Shared_Memory is configurable. How?
 * 3) Row/column major memory? -- should use 2D matrix
 * 4) use fma instruction? -- fmaf()
 * 5) Double buffer(Data prefetch). Take full use of pipeline.
 * 6) register -- tiling shared memory.
 * 7) use template. -- avoid if/else .etc control logics.
 * 8) use 2D matrix to avoid thread id/control id calculation?
 */
/**
 * This implementation is based on both Matrix 1 and 2 are row-major.
 **/
#define TILE_LENGTH_0 8
#define TILE_SIZE_0 TILE_LENGTH_0 *TILE_LENGTH_0
#define SM_BYTES_PER_T sizeof(float) * 2
__global__ void gpu_gemm_2(const unsigned int m, const unsigned int n,
                           const unsigned int k, const float alpha,
                           const float *mat1, const float *mat2,
                           const float beta, float *mat3) {
  __shared__ float temp_1[TILE_LENGTH_0][TILE_LENGTH_0];
  __shared__ float temp_2[TILE_LENGTH_0][TILE_LENGTH_0];
  unsigned int id_col_m =
      ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.y + blockIdx.x) *
          blockDim.x +
      threadIdx.x;
  float temp_prod = mat3[id_col_m];
#pragma unroll
  for (unsigned int tile_num = 0; tile_num < gridDim.x; tile_num++) {
    // load Matrix from globalMem to shared memory.
    unsigned int id_1 =
        ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.y + tile_num) *
            blockDim.x +
        threadIdx.x;
    unsigned int id_2 =
        ((tile_num * blockDim.y + threadIdx.y) * gridDim.y + blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    temp_1[threadIdx.x][threadIdx.y] = mat1[id_1];
    temp_2[threadIdx.x][threadIdx.y] = mat2[id_2];
    __syncthreads();
// calculatation
#pragma unroll
    for (unsigned int i = 0; i < TILE_LENGTH_0; i++) {
      // temp_prod = fmaf(temp_1[i][blockIdx.y], temp_2[blockIdx.x][i],
      // temp_prod);
      temp_prod += temp_1[i][threadIdx.y] * temp_2[threadIdx.x][i];
      // temp_prod += temp_1[threadIdx.x][i] * temp_2[i][threadIdx.y];
    }
    __syncthreads();
  }
  mat3[id_col_m] = temp_prod;
}
int main() {
  // Set up clock for timing comparisons
  // std::clock_t start;
  // float duration_cpu;
  // float duration_cb, duration_gpu_1, duration_gpu_2;
  float duration_gpu_2;
  unsigned int len_pow = 10;
  // const unsigned int len_limit = len_pow + 1;
  const unsigned int len_limit = 11;
  // bool result;
  float alpha = 1.0f;
  float beta = 0.0f;
  dim3 block_dim;
  dim3 grid_dim;

  // cublasStatus_t status;
  // cublasHandle_t handle;
  // status = cublasCreate(&handle);
  // if (status != CUBLAS_STATUS_SUCCESS) {
  //   fprintf(stderr, "!!!! kernel execution error.\n");
  //   return EXIT_FAILURE;
  // }

  cudaEvent_t start_g, stop_g;

  for (; len_pow < len_limit; len_pow++) {
    std::cout << "---------------- len_pow is " << len_pow
              << " ----------------" << std::endl;
    unsigned int m = (unsigned int)1 << len_pow;
    std::cout << "matrix side length is all " << m << std::endl;
    unsigned int matrix_size = m * m;

    float *mat1 = new float[matrix_size];  // on host
    float *mat2 = new float[matrix_size];
    float *mat3 = new float[matrix_size];
    float *mat3_h = new float[matrix_size];
    float *mat3_d1 = new float[matrix_size];
    float *mat3_d2 = new float[matrix_size];

    // generate input
    /* Fill the matrices with test data */
    for (unsigned int i = 0; i < matrix_size; i++) {
      mat1[i] = std::rand() / static_cast<float>(RAND_MAX);
      mat2[i] = std::rand() / static_cast<float>(RAND_MAX);
      // mat3[i] = rand() / static_cast<float>(RAND_MAX);
      mat3[i] = 0.0f;
    }
    memcpy(mat3_h, mat3, matrix_size * sizeof(float));
    // prepare data on GPU
    float *mat1_g, *mat2_g, *mat3_g;
    checkCudaErrors(cudaMalloc((void **)&mat1_g, matrix_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&mat2_g, matrix_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&mat3_g, matrix_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(mat1_g, mat1, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat2_g, mat2, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat3_g, mat3, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));

    // CPU computation  -- ignore this part. Use cublas Sgemm as golden sample
    // start = std::clock();
    // cpu_gemm_1(m, m, m, alpha, mat1, mat2, beta, mat3_h);
    // // simple_sgemm(m, alpha, mat1, mat2, beta, mat3_h);
    // duration_cpu = static_cast<float>(std::clock() - start) /
    //                static_cast<float>(CLOCK_PER_MILLSEC);
    // std::cout << "CPU computation: duration: " << duration_cpu << " ms"
    //           << std::endl;
    // // print_matrix(mat3_h, 0, 4, 0, 8, m);
    // print_matrix(mat3_h, 0, 8, 0, 8, m);

    // GPU part, golden. cuda Sgemm
    // checkCudaErrors(cudaEventCreate(&start_g));
    // checkCudaErrors(cudaEventCreate(&stop_g));
    // checkCudaErrors(cudaEventRecord(start_g));
    // status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha,
    //                      mat1_g, m, mat2_g, m, &beta, mat3_g, m);
    // checkCudaErrors(cudaEventRecord(stop_g));
    // if (status != CUBLAS_STATUS_SUCCESS) {
    //   fprintf(stderr, "!!!! kernel execution error.\n");
    //   return EXIT_FAILURE;
    // }
    // checkCudaErrors(cudaEventSynchronize(stop_g));
    // checkCudaErrors(cudaEventElapsedTime(&duration_cb, start_g, stop_g));
    // std::cout << "cuBlas computation duration: " << duration_cb << "ms"
    //           << std::endl;
    // checkCudaErrors(cudaEventDestroy(start_g));
    // checkCudaErrors(cudaEventDestroy(stop_g));
    // checkCudaErrors(cudaMemcpy(mat3_d1, mat3_g, matrix_size * sizeof(float),
    //                            cudaMemcpyDeviceToHost));
    // print_matrix(mat3_d1, 0, 4, 0, 8, m);
    // print_matrix(mat3_d1, 0, m, 0, m, m);

    // GPU cuda manual implemention. -- basic
    // checkCudaErrors(cudaMemcpy(mat3_g, mat3, matrix_size * sizeof(float),
    //                            cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaEventCreate(&start_g));
    // checkCudaErrors(cudaEventCreate(&stop_g));
    // checkCudaErrors(cudaEventRecord(start_g));
    // gpu_gemm<<<m, m>>>(m, m, m, alpha, mat1_g, mat2_g, beta, mat3_g);
    // checkCudaErrors(cudaEventRecord(stop_g));
    // checkCudaErrors(cudaEventSynchronize(stop_g));
    // checkCudaErrors(cudaEventElapsedTime(&duration_gpu_1, start_g, stop_g));
    // std::cout << "GPU manual calculation duration: " << duration_gpu_1 <<
    // "ms"
    //           << std::endl;
    // checkCudaErrors(cudaEventDestroy(start_g));
    // checkCudaErrors(cudaEventDestroy(stop_g));
    // checkCudaErrors(cudaMemcpy(mat3_d2, mat3_g, matrix_size * sizeof(float),
    //                            cudaMemcpyDeviceToHost));
    // print_matrix(mat3_d2, 0, 4, 0, 8, m);
    // print_matrix(mat3_d2, 0, 4, 0, 4, m);

    // TODO: Transpose matrix B to row major also?
    // thranpose matrix to row-major order
    float *mat1_r = new float[matrix_size];
    float *mat2_r = new float[matrix_size];
    float *mat3_r = new float[matrix_size];
    transposeMatrix(mat1, mat1_r, m);
    transposeMatrix(mat2, mat2_r, m);
    transposeMatrix(mat3, mat3_r, m);
    checkCudaErrors(cudaMemcpy(mat1_g, mat1_r, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat2_g, mat2_r, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat3_g, mat3_r, matrix_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    delete mat1_r;
    delete mat2_r;
    delete mat3_r;

    // GPU cuda manual implemention.2
    block_dim = dim3(TILE_LENGTH_0, TILE_LENGTH_0, 1);
    grid_dim = dim3(m / TILE_LENGTH_0, m / TILE_LENGTH_0, 1);
    std::cout << "grid size is: " << grid_dim.x << " " << grid_dim.y << " "
              << grid_dim.z << std::endl;
    std::cout << "Block size is: " << block_dim.x << " " << block_dim.y << " "
              << block_dim.z << std::endl;
    checkCudaErrors(cudaEventCreate(&start_g));
    checkCudaErrors(cudaEventCreate(&stop_g));
    checkCudaErrors(cudaEventRecord(start_g));
    gpu_gemm_2<<<grid_dim, block_dim, SM_BYTES_PER_T>>>(m, m, m, alpha, mat1_g,
                                                        mat2_g, beta, mat3_g);
    checkCudaErrors(cudaEventRecord(stop_g));
    // check Synchronous error(at kernel lanuch)
    checkCudaErrors(cudaGetLastError());
    // check Asynchronous error(during kernel execution) -- Notes: do sync
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventSynchronize(stop_g));
    checkCudaErrors(cudaEventElapsedTime(&duration_gpu_2, start_g, stop_g));
    std::cout << "GPU manual calculation(Tiling) duration: " << duration_gpu_2
              << "ms" << std::endl;
    checkCudaErrors(cudaEventDestroy(start_g));
    checkCudaErrors(cudaEventDestroy(stop_g));
    checkCudaErrors(cudaMemcpy(mat3_h, mat3_g, matrix_size * sizeof(float),
                               cudaMemcpyDeviceToHost));
    transposeMatrix(mat3_h, mat3_d2, m);
    // print_matrix(mat3_d2, 0, 4, 0, 8, m);
    // print_matrix(mat3_d2, 0, m, 0, m, m);

    // result = compare_matrix(mat3_d1, mat3_d2, m, m);
    // std::cout << "Calculation Result -- " << std::boolalpha << result
    //           << std::endl;
    // if (result) {
    //   std::cout << "Comparative Rate: " << (duration_cb / duration_gpu_2) *
    //   100
    //             << " %" << std::endl;
    // }
    delete mat1;
    delete mat2;
    delete mat3;

    delete mat3_h;
    delete mat3_d1;
    delete mat3_d2;
    checkCudaErrors(cudaFree(mat1_g));
    checkCudaErrors(cudaFree(mat2_g));
    checkCudaErrors(cudaFree(mat3_g));
  }
  /* Shutdown */
  // status = cublasDestroy(handle);
  // if (status != CUBLAS_STATUS_SUCCESS) {
  //   fprintf(stderr, "!!!! shutdown error (A)\n");
  //   return EXIT_FAILURE;
  // }
  return 0;
}
