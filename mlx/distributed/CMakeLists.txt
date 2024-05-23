target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
)

if (MPI_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/no_distributed.cpp
  )
endif()

if (NOT MLX_BUILD_CPU)
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/no_cpu.cpp
  )
endif()
