cmake_minimum_required(VERSION 3.27)

project(import_mlx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_executable(eval_mlp eval_mlp.cpp)
target_link_libraries(eval_mlp PRIVATE mlx)

add_executable(train_mlp train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE mlx)
