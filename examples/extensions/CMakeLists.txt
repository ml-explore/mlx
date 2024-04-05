cmake_minimum_required(VERSION 3.27)

project(mlx_sample_extensions LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(MLX CONFIG REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ----------------------------- Extensions -----------------------------

# Add library
add_library(mlx_ext)

# Add sources
target_sources(
  mlx_ext
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp
)

# Add include headers
target_include_directories(
  mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}
)

# Link to mlx
target_link_libraries(mlx_ext PUBLIC mlx)

# ----------------------------- Metal -----------------------------

# Build metallib
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET mlx_ext_metallib
    TITLE mlx_ext
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  )

  add_dependencies(
    mlx_ext
    mlx_ext_metallib
  )

endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  mlx_sample_extensions
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN mlx 
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)
target_link_libraries(mlx_sample_extensions PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(mlx_sample_extensions PRIVATE -Wl,-rpath,@loader_path)
endif()
