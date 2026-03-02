target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp)

if(MLX_BUILD_CPU AND NOT WIN32)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ring)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nccl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jaccl)
