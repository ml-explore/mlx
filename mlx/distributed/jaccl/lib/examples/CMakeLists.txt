cmake_minimum_required(VERSION 3.24)
project(jaccl_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# cmake-format: off
#
# Treating the nearby directory like a fetched dependency to simulate what a
# user would do in their own C++ project. Ideally you would write something like
# the following:
#
# FetchContent_Declare(
#   jaccl
#   GIT_REPOSITORY https://github.com/ml-explore/mlx.git
#   GIT_TAG main
#   SOURCE_SUBDIR mlx/distributed/jaccl/lib
# )
#
# cmake-format: on
FetchContent_Declare(jaccl SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
FetchContent_MakeAvailable(jaccl)

# Helper function to build examples
function(build_example SRCFILE)
  get_filename_component(example_name ${SRCFILE} NAME_WE)
  set(target "jaccl_${example_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(${target} PRIVATE jaccl)
  message(STATUS "Building JACCL example: ${target}")
endfunction()

# Examples
build_example(minimal_env.cpp)
build_example(minimal_cfg.cpp)

# Benchmarks
build_example(allreduce_bench.cpp)
