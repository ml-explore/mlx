add_custom_command(
    OUTPUT  compiled_preamble.cpp
    COMMAND /bin/bash
              ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.sh
              ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp
              ${CMAKE_C_COMPILER}
              ${PROJECT_SOURCE_DIR}
    DEPENDS make_compiled_preamble.sh
            kernels/compiled_preamble.h
            kernels/unary.h
            kernels/binary.h
)

add_custom_target(
  compiled_preamble
  DEPENDS compiled_preamble.cpp
)

add_dependencies(mlx compiled_preamble)

target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
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
  ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp
)

if (NOT MLX_METAL_PATH)
  set(MLX_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR}/kernels/)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

target_compile_definitions(
  mlx PRIVATE METAL_PATH="${MLX_METAL_PATH}/mlx.metallib")
