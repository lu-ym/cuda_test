#
cmake_minimum_required (VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (CudaTest)

set(CUDA_PATH "/usr/local/cuda")

set(CUDA_LIBRARY cuda)
set(CUDART_LIBRARY cudart)
set(CUBLAS_LIBRARY cublas)
set(CUBLASLT_LIBRARY cublasLt)

if(NOT WIN32)
  find_library(CUDA_LIBRARY cuda ${CUDA_PATH}/lib64)
  find_library(CUDART_LIBRARY cudart ${CUDA_PATH}/lib64)
  find_library(CUBLAS_LIBRARY cublas ${CUDA_PATH}/lib64)
  find_library(CUBLASLT_LIBRARY cublasLt ${CUDA_PATH}/lib64)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(00_deviceinfo)
add_subdirectory(01_simpletest)
add_subdirectory(02_vectorAdd)
add_subdirectory(03_mma)
add_subdirectory(04_reduction)
add_subdirectory(05_conv)


