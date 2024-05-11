function(make_jit_source SRC_NAME)
  add_custom_command(
    OUTPUT  jit/${SRC_NAME}.cpp
    COMMAND /bin/bash
              ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.sh
              ${CMAKE_CURRENT_BINARY_DIR}/jit
              ${CMAKE_C_COMPILER}
              ${PROJECT_SOURCE_DIR}
              ${SRC_NAME}
              "-D${MLX_METAL_VERSION}"
    DEPENDS make_compiled_preamble.sh
            kernels/${SRC_NAME}.h
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

make_jit_source(utils
  kernels/bf16.h
  kernels/complex.h
)
make_jit_source(unary
  kernels/erf.h
  kernels/expm1f.h
)
make_jit_source(binary)
make_jit_source(ternary)
make_jit_source(reduction
  kernels/atomic.h
  kernels/reduction/ops.h
)
make_jit_source(scatter)
make_jit_source(gather)

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
