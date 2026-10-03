set(THREADING_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

if(APPLE)
  message(STATUS "Threading: Using Accelerate/GCD for macOS")
  list(APPEND THREADING_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/accelerate/thread_pool.mm)
else()
  message(STATUS "Threading: Using persistent CPU threading")
  list(APPEND THREADING_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/openblas/thread_pool.cpp)
endif()

target_sources(mlx PRIVATE ${THREADING_SOURCES})
