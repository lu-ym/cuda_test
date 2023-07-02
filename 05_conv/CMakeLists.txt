# CopyRight
#
#
cmake_minimum_required (VERSION 3.18)

project(conv LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME})

file(GLOB LIB_SRC
  "${CMAKE_SOURCE_DIR}/common/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

include_directories(
  "${CUDA_PATH}/include"
  "${CMAKE_SOURCE_DIR}/common"
  "${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_sources(${PROJECT_NAME} PUBLIC ${LIB_SRC})

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${CUDA_LIBRARY}
    ${CUDART_LIBRARY}
)

target_link_directories(${PROJECT_NAME}
  PUBLIC
    "${CUDA_PATH}/lib64"
)


