cmake_minimum_required(VERSION 3.27)

project(mlx_sample_extensions LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(MLX CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

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

# ----------------------------- Pybind -----------------------------
pybind11_add_module(
  mlx_sample_extensions
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)
target_link_libraries(mlx_sample_extensions PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(mlx_sample_extensions PRIVATE -Wl,-rpath,@loader_path)
endif()
