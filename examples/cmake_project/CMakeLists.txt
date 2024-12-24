cmake_minimum_required(VERSION 3.27)

project(example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Comment the following two commands only the MLX C++ library is installed and
# set(MLX_ROOT "/path/to/mlx") directly if needed.
find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)

find_package(MLX CONFIG REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE mlx)
