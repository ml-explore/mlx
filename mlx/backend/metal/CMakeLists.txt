target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
)

if (NOT MLX_METAL_PATH)
  set(MLX_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR}/kernels/)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

target_compile_definitions(
  mlx PRIVATE METAL_PATH="${MLX_METAL_PATH}/mlx.metallib")
