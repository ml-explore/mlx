cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pip show mlx
  COMMAND grep location
  COMMAND awk "{print $4 \"/mlx\"}"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_executable(eval_mlp eval_mlp.cpp)
target_link_libraries(eval_mlp PRIVATE mlx)

add_executable(train_mlp train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE mlx)
