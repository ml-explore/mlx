target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/encoder.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp)
