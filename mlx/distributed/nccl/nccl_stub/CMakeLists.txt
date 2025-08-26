cmake_minimum_required(VERSION 3.25)

project(nccl LANGUAGES C CXX)

file(
  DOWNLOAD
  "https://raw.githubusercontent.com/NVIDIA/nccl/refs/tags/v2.27.5-1/src/nccl.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/nccl.h")

add_library(nccl SHARED nccl_stubs.cpp)
find_package(CUDAToolkit REQUIRED)
target_include_directories(nccl PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(nccl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
