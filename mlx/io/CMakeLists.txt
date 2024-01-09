target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/safetensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gguf.cpp
)
