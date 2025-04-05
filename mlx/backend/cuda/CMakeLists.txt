# Filename rules in CUDA backend:
#
# * Use .cu/.cuh if code contains device code, and .cpp/.h if not.
# * Device-only kernel code should be put in kernels/ subdir.
# * Files in kernels/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

target_compile_definitions(mlx PUBLIC MLX_USE_CUDA)

# Enable defining device lambda functions.
target_compile_options(mlx
                       PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

# Compute capability 7 is required for synchronization between CPU/GPU with
# managed memory. TODO: Add more architectures for potential performance gain.
set(MLX_CUDA_ARCHITECTURES
    "70;80"
    CACHE STRING "CUDA architectures")
message(STATUS "CUDA architectures: ${MLX_CUDA_ARCHITECTURES}")
set_target_properties(mlx PROPERTIES CUDA_ARCHITECTURES
                                     "${MLX_CUDA_ARCHITECTURES}")

# Use fixed version of CCCL.
FetchContent_Declare(
  cccl
  URL "https://github.com/NVIDIA/cccl/releases/download/v2.8.1/cccl-v2.8.1.zip")
FetchContent_MakeAvailable(cccl)
target_include_directories(mlx PRIVATE BEFORE "${cccl_SOURCE_DIR}/include")

# Make CUDA APIs visible in .cpp files.
find_package(CUDAToolkit REQUIRED)
target_include_directories(mlx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# We are using cublasLt.
target_link_libraries(mlx PRIVATE CUDA::cublasLt_static)

# Suppress nvcc warnings on MLX headers.
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                                   --diag_suppress=997>)
