target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compile.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/dtype.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/einsum.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/graph_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/metal.h)

if(MSVC)
  # Disable some MSVC warnings to speed up compilation.
  target_compile_options(mlx PUBLIC /wd4068 /wd4244 /wd4267 /wd4804)
endif()

if(WIN32)
  # Export symbols by default to behave like macOS/linux.
  set_target_properties(mlx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/common)

if(MLX_BUILD_CPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_cpu)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/distributed)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/io)

if(MLX_BUILD_METAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/metal)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_metal)
endif()
