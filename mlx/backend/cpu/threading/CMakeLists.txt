# Threading backend selection
#
# Uses platform-aligned thread pools: - macOS: Grand Central Dispatch (GCD)
# - Linux/Windows: OpenBLAS-coordinated threading

set(THREADING_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

# Select platform-specific implementation
if(APPLE)
  message(STATUS "Threading: Using Accelerate/GCD for macOS")
  list(APPEND THREADING_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/accelerate/thread_pool.mm)
  target_sources(mlx PRIVATE ${THREADING_SOURCES})
  target_compile_definitions(mlx PRIVATE MLX_THREADING_ACCELERATE)
else()
  message(STATUS "Threading: Using OpenBLAS-coordinated threading")
  list(APPEND THREADING_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/openblas/thread_pool.cpp)
  target_sources(mlx PRIVATE ${THREADING_SOURCES})
  target_compile_definitions(mlx PRIVATE MLX_THREADING_OPENBLAS)
endif()
