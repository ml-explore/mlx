function(make_jit_source SRC_FILE)
  # This function takes a metal header file,
  # runs the C preprocessesor on it, and makes
  # the processed contents available as a string in a C++ function
  # mlx::core::metal::${SRC_NAME}()
  #
  # To use the function, declare it in jit/includes.h and
  # include jit/includes.h.
  #
  # Additional arguments to this function are treated as dependencies
  # in the Cmake build system.
  get_filename_component(SRC_NAME ${SRC_FILE} NAME)
  add_custom_command(
    OUTPUT  jit/${SRC_NAME}.cpp
    COMMAND /bin/bash
              ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.sh
              ${CMAKE_CURRENT_BINARY_DIR}/jit
              ${CMAKE_C_COMPILER}
              ${PROJECT_SOURCE_DIR}
              ${SRC_FILE}
              "-D${MLX_METAL_VERSION}"
    DEPENDS make_compiled_preamble.sh
            kernels/${SRC_FILE}.h
            ${ARGN}
  )
  add_custom_target(${SRC_NAME} DEPENDS jit/${SRC_NAME}.cpp)
  add_dependencies(mlx ${SRC_NAME})
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/jit/${SRC_NAME}.cpp
  )
endfunction(make_jit_source)

make_jit_source(
  utils
  kernels/bf16.h
  kernels/complex.h
  kernels/defines.h
)
make_jit_source(
  unary_ops
  kernels/erf.h
  kernels/expm1f.h
)
make_jit_source(binary_ops)
make_jit_source(ternary_ops)
make_jit_source(
  reduce_utils
  kernels/atomic.h
  kernels/reduction/ops.h
)
make_jit_source(scatter)
make_jit_source(gather)

if (MLX_METAL_JIT) 
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_kernels.cpp
  )
  make_jit_source(arange)
  make_jit_source(copy)
  make_jit_source(unary)
  make_jit_source(binary)
  make_jit_source(binary_two)
  make_jit_source(
    fft
    kernels/fft/radix.h
    kernels/fft/readwrite.h
  )
  make_jit_source(ternary)
  make_jit_source(softmax)
  make_jit_source(scan)
  make_jit_source(sort)
  make_jit_source(
    reduce
    kernels/reduction/reduce_all.h
    kernels/reduction/reduce_col.h
    kernels/reduction/reduce_row.h
  )
  make_jit_source(
    steel/gemm/gemm
    kernels/steel/utils.h
    kernels/steel/gemm/loader.h
    kernels/steel/gemm/mma.h
    kernels/steel/gemm/params.h
    kernels/steel/gemm/transforms.h
  )
  make_jit_source(steel/gemm/kernels/steel_gemm_fused)
  make_jit_source(
    steel/gemm/kernels/steel_gemm_masked
    kernels/steel/defines.h
  )
  make_jit_source(steel/gemm/kernels/steel_gemm_splitk)
  make_jit_source(
    steel/conv/conv
    kernels/steel/utils.h
    kernels/steel/defines.h
    kernels/steel/gemm/mma.h
    kernels/steel/gemm/transforms.h
    kernels/steel/conv/params.h
    kernels/steel/conv/loader.h
    kernels/steel/conv/loaders/loader_channel_l.h
    kernels/steel/conv/loaders/loader_channel_n.h
  )
  make_jit_source(
    steel/conv/kernels/steel_conv
  )
  make_jit_source(
    steel/conv/kernels/steel_conv_general
    kernels/steel/defines.h
    kernels/steel/conv/loaders/loader_general.h
  )
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nojit_kernels.cpp
  )
endif()

target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
)

if (NOT MLX_METAL_PATH)
  set(MLX_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR}/kernels/)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

target_compile_definitions(
  mlx PRIVATE METAL_PATH="${MLX_METAL_PATH}/mlx.metallib")
