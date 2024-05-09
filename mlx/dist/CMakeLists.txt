target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
)

if (MPI_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
endif()
