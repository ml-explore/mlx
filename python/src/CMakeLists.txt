nanobind_add_module(
  core
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_SOURCE_DIR}/mlx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlx_func.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trees.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

if(NOT MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY)
  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  else()
    set(MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  endif()
endif()

set_target_properties(
  core
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}
             # Do not append a sub-dir for multi-config generators like MSVC
             # and XCode.
             LIBRARY_OUTPUT_DIRECTORY_RELEASE
             ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}
             LIBRARY_OUTPUT_DIRECTORY_DEBUG
             ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}
             LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
             ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}
             LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL
             ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY})

target_link_libraries(core PRIVATE mlx)

if(BUILD_SHARED_LIBS)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_options(core PRIVATE -Wl,-rpath,@loader_path/lib)
  else()
    target_link_options(core PRIVATE -Wl,-rpath,\$ORIGIN/lib)
  endif()
endif()
