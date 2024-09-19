target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp)

if(MPI_FOUND AND MLX_BUILD_CPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_distributed.cpp)
endif()
