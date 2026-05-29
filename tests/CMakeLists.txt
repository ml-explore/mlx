FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${PROJECT_SOURCE_DIR}/tests/tests.cpp)

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  set(METAL_TEST_SOURCES gpu_tests.cpp)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

target_sources(
  tests
  PRIVATE allocator_tests.cpp
          array_tests.cpp
          arg_reduce_tests.cpp
          autograd_tests.cpp
          blas_tests.cpp
          compile_tests.cpp
          custom_vjp_tests.cpp
          creations_tests.cpp
          device_tests.cpp
          einsum_tests.cpp
          export_import_tests.cpp
          eval_tests.cpp
          fast_tests.cpp
          fft_tests.cpp
          load_tests.cpp
          ops_tests.cpp
          random_tests.cpp
          scheduler_tests.cpp
          utils_tests.cpp
          vmap_tests.cpp
          linalg_tests.cpp
          thread_pool_tests.cpp
          ${METAL_TEST_SOURCES})

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})

set(MLX_TEST_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
  # OpenBLAS is copied to the build root on Windows. Run test discovery from
  # there so the test executable can find libopenblas.dll at startup.
  set(MLX_TEST_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

doctest_discover_tests(tests WORKING_DIRECTORY ${MLX_TEST_WORKING_DIRECTORY})
add_test(NAME tests COMMAND tests)
set_tests_properties(tests PROPERTIES WORKING_DIRECTORY
                                      ${MLX_TEST_WORKING_DIRECTORY})

# Standalone test: verify clean exit when GPU work is in-flight during teardown.
# (Cannot be a doctest case because the crash occurs during static destruction.)
add_executable(test_teardown test_teardown.cpp)
target_link_libraries(test_teardown PRIVATE mlx)
target_compile_options(test_teardown PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(test_teardown PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME teardown COMMAND test_teardown)
