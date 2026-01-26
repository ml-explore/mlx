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

# Try to find MIOpen (optional but recommended)
find_package(miopen CONFIG QUIET)
if(miopen_FOUND)
  message(STATUS "MIOpen found - enabling MIOpen support")
  set(MLX_USE_MIOPEN ON)
else()
  # Try to find MIOpen library directly
  find_library(MIOPEN_LIB MIOpen PATHS ${ROCM_PATH}/lib /opt/rocm/lib /opt/rocm-6.0.0/lib)
  find_path(MIOPEN_INCLUDE_DIR miopen/miopen.h PATHS ${ROCM_PATH}/include /opt/rocm/include /opt/rocm-6.0.0/include)
  if(MIOPEN_LIB AND MIOPEN_INCLUDE_DIR)
    message(STATUS "MIOpen found at ${MIOPEN_LIB} - enabling MIOpen support")
    set(MLX_USE_MIOPEN ON)
  else()
    message(STATUS "MIOpen not found - convolution and SDPA will use fallback implementations")
    set(MLX_USE_MIOPEN OFF)
  endif()
endif()

# Ensure HIP architectures are set - respect user-provided value
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES
      "gfx906;gfx908;gfx90a;gfx1030;gfx1100"
      CACHE STRING "HIP architectures" FORCE)
endif()
message(
  STATUS "ROCm backend using HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

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

# Build include flags
set(HIP_INCLUDE_FLAGS "-I${CMAKE_SOURCE_DIR}" "-I${HIP_INCLUDE_DIRS}")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/scan.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/sort.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/unary.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gemv.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/affine_quantize.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/fp_quantize.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/convert_fp8.hip)

# Create output directory for compiled objects
set(HIP_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip_objs")
file(MAKE_DIRECTORY ${HIP_OBJ_DIR})

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
    COMMAND ${CMAKE_HIP_COMPILER} -c ${hip_src} -o ${hip_obj} -fPIC
            -DMLX_USE_ROCM ${HIP_ARCH_FLAGS} ${HIP_INCLUDE_FLAGS} -std=c++17
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
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/rocblas_gemm.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_ROCM)

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

message(
  STATUS
    "ROCm libraries: rocblas=${ROCBLAS_LIB}, hiprand=${HIPRAND_LIB}, amdhip64=${AMDHIP64_LIB}"
)

# Link the static library and ROCm libraries to mlx We link directly to the .so
# files instead of using CMake targets to avoid propagating compile options like
# -x hip
target_link_libraries(mlx PRIVATE ${HIP_STATIC_LIB} ${AMDHIP64_LIB}
                                  ${ROCBLAS_LIB} ${HIPRAND_LIB})

# Include ROCm headers for mlx C++ files Get the HIP include directory from the
# hip package
get_target_property(HIP_HOST_INCLUDES hip::host INTERFACE_INCLUDE_DIRECTORIES)
if(HIP_HOST_INCLUDES)
  target_include_directories(mlx PRIVATE ${HIP_HOST_INCLUDES})
endif()
target_include_directories(mlx PRIVATE ${HIP_INCLUDE_DIRS})

# Add HIP platform define for C++ files
target_compile_definitions(mlx PRIVATE __HIP_PLATFORM_AMD__=1)
