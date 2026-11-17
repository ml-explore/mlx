target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compile.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/trace.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/simplify.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/recompute.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fuse.cpp)
