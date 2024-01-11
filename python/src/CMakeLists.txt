pybind11_add_module(
  core
  ${CMAKE_CURRENT_SOURCE_DIR}/mlx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
)

if (NOT MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY)
  set(MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

set_target_properties(
  core 
  PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY 
  ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}
)

target_link_libraries(core PRIVATE mlx)
target_compile_definitions(core PRIVATE _VERSION_=${MLX_VERSION})

if(BUILD_SHARED_LIBS)
  target_link_options(core PRIVATE -Wl,-rpath,@loader_path/lib)
endif()
