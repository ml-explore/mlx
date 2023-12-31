target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dtype.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/metal.h
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/io)
if (MLX_BUILD_ACCELERATE) 
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/accelerate)
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/common/default_primitives.cpp
  )
endif()

if (MLX_BUILD_METAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/metal)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_metal)
endif()
