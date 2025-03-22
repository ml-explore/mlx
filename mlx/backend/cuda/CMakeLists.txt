target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

set_target_properties(
  mlx
  PROPERTIES CUDA_STANDARD 17
             CUDA_SEPARABLE_COMPILATION ON
             # Compute capability 7 is required for synchronization between
             # CPU/GPU with managed memory.
             # TODO: Add more architectures for release build.
             CUDA_ARCHITECTURES "70;80")

# Make CUDA APIs visible in .cpp files.
find_package(CUDAToolkit REQUIRED)
target_include_directories(mlx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Suppress nvcc warnings on C++ headers.
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                                   --diag_suppress=997>)
