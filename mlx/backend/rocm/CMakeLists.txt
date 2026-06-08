# Filename rules in ROCm backend:
#
# * Use .hip/.hpp if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.

# Find ROCm packages
find_package(hip REQUIRED CONFIG)
find_package(rocblas REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)
find_package(rocprim REQUIRED CONFIG)
find_package(hiprand REQUIRED CONFIG)
# Ensure HIP architectures are set - respect user-provided value from command
# line The user can set this via -DCMAKE_HIP_ARCHITECTURES=gfx1011
#
# Supported architectures from ROCm 6.4.0 - 7.2.0 compatibility matrix: CDNA:
# gfx908 (MI100), gfx90a (MI200), gfx942 (MI300) RDNA2:  gfx1030 (RX 6000
# series) RDNA3:  gfx1100 (RX 7900), gfx1101 (RX 7600) RDNA3.5: gfx1150,
# gfx1151, gfx1152 (Ryzen AI / Radeon 8060S) RDNA4:  gfx1200, gfx1201 (RX 9000
# series)
if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES
      "gfx908;gfx90a;gfx942;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201"
      CACHE STRING "HIP architectures" FORCE)
endif()
message(
  STATUS "ROCm backend using HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

# Check if any target architecture supports WMMA (RDNA 3 / gfx11xx and RDNA 4 /
# gfx12xx)
set(MLX_HAS_ROCM_WMMA OFF)
foreach(arch ${CMAKE_HIP_ARCHITECTURES})
  if(arch MATCHES "^gfx1[12]")
    set(MLX_HAS_ROCM_WMMA ON)
    break()
  endif()
endforeach()
message(STATUS "ROCm WMMA support: ${MLX_HAS_ROCM_WMMA}")

if(MLX_HAS_ROCM_WMMA)
  find_package(rocwmma REQUIRED CONFIG)
endif()

# Build architecture flags
set(HIP_ARCH_FLAGS "")
foreach(arch ${CMAKE_HIP_ARCHITECTURES})
  list(APPEND HIP_ARCH_FLAGS "--offload-arch=${arch}")
endforeach()

# Get HIP include directories
get_target_property(HIP_DEVICE_INCLUDES hip::device
                    INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(ROCTHRUST_INCLUDES roc::rocthrust
                    INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(ROCPRIM_INCLUDES roc::rocprim INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(HIPRAND_INCLUDES hip::hiprand INTERFACE_INCLUDE_DIRECTORIES)
if(MLX_HAS_ROCM_WMMA)
  get_target_property(ROCWMMA_INCLUDES roc::rocwmma
                      INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find GCC installation for C++ standard library headers ROCm's clang needs to
# know where to find libstdc++ headers
execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include/c++
  OUTPUT_VARIABLE GCC_CXX_INCLUDE_BASE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(GCC_CXX_INCLUDE_BASE "${GCC_CXX_INCLUDE_BASE}" DIRECTORY)

# Get GCC version for the target-specific include directory
execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
  OUTPUT_VARIABLE GCC_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "^[0-9]+" GCC_MAJOR_VERSION "${GCC_VERSION}")

# Build include flags - use PROJECT_SOURCE_DIR for correct path
set(HIP_INCLUDE_FLAGS "-I${PROJECT_SOURCE_DIR}" "-I${HIP_INCLUDE_DIRS}")

# Add C++ standard library include paths for HIP compiler
if(EXISTS "${GCC_CXX_INCLUDE_BASE}/c++/${GCC_MAJOR_VERSION}")
  list(APPEND HIP_INCLUDE_FLAGS
       "-I${GCC_CXX_INCLUDE_BASE}/c++/${GCC_MAJOR_VERSION}")
  list(APPEND HIP_INCLUDE_FLAGS
       "-I${GCC_CXX_INCLUDE_BASE}/c++/${GCC_MAJOR_VERSION}/x86_64-linux-gnu")
  list(APPEND HIP_INCLUDE_FLAGS
       "-I${GCC_CXX_INCLUDE_BASE}/c++/${GCC_MAJOR_VERSION}/backward")
endif()

# Also try to find system include directories
if(EXISTS "/usr/include/c++/${GCC_MAJOR_VERSION}")
  list(APPEND HIP_INCLUDE_FLAGS "-I/usr/include/c++/${GCC_MAJOR_VERSION}")
  list(APPEND HIP_INCLUDE_FLAGS
       "-I/usr/include/x86_64-linux-gnu/c++/${GCC_MAJOR_VERSION}")
  list(APPEND HIP_INCLUDE_FLAGS
       "-I/usr/include/c++/${GCC_MAJOR_VERSION}/backward")
endif()

# Add standard system include paths
list(APPEND HIP_INCLUDE_FLAGS "-I/usr/include/x86_64-linux-gnu")
list(APPEND HIP_INCLUDE_FLAGS "-I/usr/include")

foreach(inc ${HIP_DEVICE_INCLUDES})
  if(inc)
    list(APPEND HIP_INCLUDE_FLAGS "-I${inc}")
  endif()
endforeach()
foreach(inc ${ROCTHRUST_INCLUDES})
  if(inc)
    list(APPEND HIP_INCLUDE_FLAGS "-I${inc}")
  endif()
endforeach()
foreach(inc ${ROCPRIM_INCLUDES})
  if(inc)
    list(APPEND HIP_INCLUDE_FLAGS "-I${inc}")
  endif()
endforeach()
foreach(inc ${HIPRAND_INCLUDES})
  if(inc)
    list(APPEND HIP_INCLUDE_FLAGS "-I${inc}")
  endif()
endforeach()
if(MLX_HAS_ROCM_WMMA)
  foreach(inc ${ROCWMMA_INCLUDES})
    if(inc)
      list(APPEND HIP_INCLUDE_FLAGS "-I${inc}")
    endif()
  endforeach()
endif()

message(STATUS "HIP include flags: ${HIP_INCLUDE_FLAGS}")

# HIP source files
set(HIP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/event.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/arange.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/binary.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/indexing.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/random.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rope.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/scan.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/sort.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/unary.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gemv.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/gemms/naive_gemm.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/affine_quantize.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/fp_quantize.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/convert_fp8.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmv_tiled_kernel.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/conv/gemm_conv.hip)

if(MLX_HAS_ROCM_WMMA)
  list(APPEND HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_wmma.hip)
endif()

# Create output directory for compiled objects
set(HIP_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip_objs")
file(MAKE_DIRECTORY ${HIP_OBJ_DIR})

# Detect CPU count for parallel HIP offload compilation Use half of available
# CPUs for parallel HIP offload compilation per file (Ninja already parallelizes
# across files, so this avoids oversubscription)
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 4)
else()
  math(EXPR NPROC "${NPROC} / 2")
  if(NPROC LESS 2)
    set(NPROC 2)
  endif()
endif()

# Compile each HIP file to object file using custom commands Use -fno-gpu-rdc to
# avoid needing device link step
set(HIP_OBJECTS "")
foreach(hip_src ${HIP_SOURCES})
  get_filename_component(hip_name ${hip_src} NAME_WE)
  get_filename_component(hip_dir ${hip_src} DIRECTORY)
  file(RELATIVE_PATH rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${hip_dir})

  # Create subdirectory for object if needed
  if(rel_dir)
    set(obj_subdir "${HIP_OBJ_DIR}/${rel_dir}")
    file(MAKE_DIRECTORY ${obj_subdir})
    set(hip_obj "${obj_subdir}/${hip_name}.o")
  else()
    set(hip_obj "${HIP_OBJ_DIR}/${hip_name}.o")
  endif()

  add_custom_command(
    OUTPUT ${hip_obj}
    COMMAND
      ${CMAKE_HIP_COMPILER} -c ${hip_src} -o ${hip_obj} -fPIC -DMLX_USE_ROCM
      ${HIP_ARCH_FLAGS} ${HIP_INCLUDE_FLAGS} -std=c++17 -parallel-jobs=${NPROC}
    DEPENDS ${hip_src}
    COMMENT "Compiling HIP source ${hip_src}"
    VERBATIM)

  list(APPEND HIP_OBJECTS ${hip_obj})
endforeach()

# Create a custom target for all HIP objects
add_custom_target(mlx_hip_objects DEPENDS ${HIP_OBJECTS})

# Create static library from all objects (no device link needed without
# -fgpu-rdc)
set(HIP_STATIC_LIB "${CMAKE_CURRENT_BINARY_DIR}/libmlx_rocm_kernels.a")
add_custom_command(
  OUTPUT ${HIP_STATIC_LIB}
  COMMAND ${CMAKE_AR} rcs ${HIP_STATIC_LIB} ${HIP_OBJECTS}
  DEPENDS ${HIP_OBJECTS}
  COMMENT "Creating static library from HIP objects"
  VERBATIM)

add_custom_target(mlx_rocm_kernels_lib DEPENDS ${HIP_STATIC_LIB})

# Add C++ sources directly to mlx target
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/rocblas_gemm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/hipblaslt_gemm.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_ROCM)
if(MLX_HAS_ROCM_WMMA)
  target_compile_definitions(mlx PRIVATE MLX_HAS_ROCM_WMMA)
endif()

# Make mlx depend on the HIP kernels library
add_dependencies(mlx mlx_rocm_kernels_lib)

# Get the library paths from the imported targets (without propagating compile
# options)
get_target_property(ROCBLAS_LIB roc::rocblas IMPORTED_LOCATION)
if(NOT ROCBLAS_LIB)
  get_target_property(ROCBLAS_LIB roc::rocblas IMPORTED_LOCATION_RELEASE)
endif()
if(NOT ROCBLAS_LIB)
  # Fallback to finding the library directly
  find_library(ROCBLAS_LIB rocblas PATHS ${ROCM_PATH}/lib /opt/rocm/lib
                                         /opt/rocm-6.0.0/lib)
endif()

get_target_property(HIPRAND_LIB hip::hiprand IMPORTED_LOCATION)
if(NOT HIPRAND_LIB)
  get_target_property(HIPRAND_LIB hip::hiprand IMPORTED_LOCATION_RELEASE)
endif()
if(NOT HIPRAND_LIB)
  find_library(HIPRAND_LIB hiprand PATHS ${ROCM_PATH}/lib /opt/rocm/lib
                                         /opt/rocm-6.0.0/lib)
endif()

# Find amdhip64 library
find_library(AMDHIP64_LIB amdhip64 PATHS ${ROCM_PATH}/lib /opt/rocm/lib
                                         /opt/rocm-6.0.0/lib)

# Find hiprtc library (needed for JIT compilation)
find_library(HIPRTC_LIB hiprtc PATHS ${ROCM_PATH}/lib /opt/rocm/lib
                                     /opt/rocm-6.0.0/lib)

# Find hipBLASLt library (optimized GEMM for half-precision)
find_library(HIPBLASLT_LIB hipblaslt PATHS ${ROCM_PATH}/lib /opt/rocm/lib
                                           /opt/rocm-6.0.0/lib)

message(
  STATUS
    "ROCm libraries: rocblas=${ROCBLAS_LIB}, hiprand=${HIPRAND_LIB}, amdhip64=${AMDHIP64_LIB}, hiprtc=${HIPRTC_LIB}, hipblaslt=${HIPBLASLT_LIB}"
)

# Link the static library and ROCm libraries to mlx We link directly to the .so
# files instead of using CMake targets to avoid propagating compile options like
# -x hip
target_link_libraries(
  mlx PRIVATE ${HIP_STATIC_LIB} ${AMDHIP64_LIB} ${ROCBLAS_LIB} ${HIPRAND_LIB}
              ${HIPRTC_LIB} ${HIPBLASLT_LIB})

# Include ROCm headers for mlx C++ files Get the HIP include directory from the
# hip package
get_target_property(HIP_HOST_INCLUDES hip::host INTERFACE_INCLUDE_DIRECTORIES)
if(HIP_HOST_INCLUDES)
  target_include_directories(mlx PRIVATE ${HIP_HOST_INCLUDES})
endif()
target_include_directories(mlx PRIVATE ${HIP_INCLUDE_DIRS})

# Add HIP platform define for C++ files
target_compile_definitions(mlx PRIVATE __HIP_PLATFORM_AMD__=1)
