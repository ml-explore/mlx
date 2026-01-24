# Filename rules in ROCm backend:
#
# * Use .hip/.hpp if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/random.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.hip)

target_compile_definitions(mlx PRIVATE MLX_USE_ROCM)

# Set HIP compiler flags
target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>")

# Set GPU architectures for ROCm
if(NOT DEFINED MLX_ROCM_ARCHITECTURES)
  set(MLX_ROCM_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100")
endif()
message(STATUS "ROCm architectures: ${MLX_ROCM_ARCHITECTURES}")

foreach(arch ${MLX_ROCM_ARCHITECTURES})
  target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${arch}>")
endforeach()

# Find ROCm packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocthrust REQUIRED)
find_package(rocprim REQUIRED)

# Link ROCm libraries
target_link_libraries(mlx PRIVATE hip::host roc::rocblas roc::rocthrust roc::rocprim)

# Include ROCm headers
target_include_directories(mlx PRIVATE ${HIP_INCLUDE_DIRS})
