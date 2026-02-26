cmake_minimum_required(VERSION 3.10)
project(ts_infer)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

list(APPEND CUDA_NVCC_FLAGS -std=c++14;-O3)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
list(APPEND CUDA_NVCC_FLAGS -gencode;arch=compute_86,code=sm_86)


set(CMAKE_PREFIX_PATH "/home/vigi/c++_packages/libtorch-cxx11-abi-shared-with-deps-1.11.0+cu115/libtorch")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
