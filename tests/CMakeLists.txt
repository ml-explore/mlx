FetchContent_Declare(
  doctest
  GIT_REPOSITORY "https://github.com/onqtam/doctest"
  GIT_TAG "b7c21ec5ceeadb4951b00396fc1e4642dd347e5f"
)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${PROJECT_SOURCE_DIR}/tests/tests.cpp)

if (MLX_BUILD_METAL)
  set(
    METAL_TEST_SOURCES
    metal_tests.cpp
  )
endif()

target_sources(tests PRIVATE 
  allocator_tests.cpp
  array_tests.cpp
  arg_reduce_tests.cpp
  autograd_tests.cpp
  blas_tests.cpp
  creations_tests.cpp
  device_tests.cpp
  eval_tests.cpp
  fft_tests.cpp
  #  graph_optimize_tests.cpp
  load_tests.cpp
  ops_tests.cpp
  random_tests.cpp
  scheduler_tests.cpp
  utils_tests.cpp
  vmap_tests.cpp
  linalg_tests.cpp
  ${METAL_TEST_SOURCES}
)

target_link_libraries(tests PRIVATE mlx doctest)
add_test(NAME tests COMMAND tests)
