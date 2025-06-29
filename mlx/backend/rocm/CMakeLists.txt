# Filename rules in ROCm backend:
#
# * Use .hip/.hpp if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/random.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_ROCM)

# Embed kernel sources in binary for JIT compilation.
file(
  GLOB MLX_JIT_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.hpp")
string(JOIN ":" MLX_JIT_SOURCES_ARG ${MLX_JIT_SOURCES})
add_custom_command(
  OUTPUT gen/rocm_jit_sources.h
  COMMAND
    ${CMAKE_COMMAND} -DMLX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
    -DMLX_JIT_SOURCES=${MLX_JIT_SOURCES_ARG} -P
    "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
  DEPENDS bin2h.cmake ${MLX_JIT_SOURCES})
add_custom_target(rocm_jit_sources DEPENDS gen/rocm_jit_sources.h)
add_dependencies(mlx rocm_jit_sources)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Find ROCm installation
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

# Link with ROCm libraries
target_link_libraries(mlx PRIVATE hip::device roc::rocblas)

# Set GPU architectures for ROCm Common ROCm architectures: gfx900, gfx906,
# gfx908, gfx90a, gfx1030, gfx1100
set(MLX_ROCM_ARCHITECTURES
    "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
    CACHE STRING "ROCm GPU architectures")
message(STATUS "ROCm GPU architectures: ${MLX_ROCM_ARCHITECTURES}")

# Set GPU targets for HIP compilation
set_property(TARGET mlx PROPERTY HIP_ARCHITECTURES "${MLX_ROCM_ARCHITECTURES}")

# Enable HIP language support
enable_language(HIP)

# Set HIP compiler flags
target_compile_options(
  mlx
  PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>"
          "$<$<COMPILE_LANGUAGE:HIP>:-Xcompiler=-Wall>"
          "$<$<COMPILE_LANGUAGE:HIP>:-Xcompiler=-Wextra>")

# Add ROCm include directories
target_include_directories(mlx PRIVATE ${hip_INCLUDE_DIRS})
target_include_directories(mlx PRIVATE ${rocblas_INCLUDE_DIRS})
