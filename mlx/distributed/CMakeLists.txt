target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
