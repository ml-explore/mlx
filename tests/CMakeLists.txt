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
          fft_tests.cpp
          load_tests.cpp
          ops_tests.cpp
          random_tests.cpp
          scheduler_tests.cpp
          utils_tests.cpp
          vmap_tests.cpp
          linalg_tests.cpp
          ${METAL_TEST_SOURCES})

if(MLX_BUILD_CUDA)
  # C++ tests are always built from source, so we have to specify where to find
  # CCCL headers for JIT as they are not installed in system.
  get_target_property(MLX_CCCL_DIR mlx CCCL_DIR)
  target_compile_definitions(mlx PRIVATE MLX_CCCL_DIR="${MLX_CCCL_DIR}")
  message(STATUS MLX_CCCL_DIR="${MLX_CCCL_DIR}")
endif()

if(USE_ASAN AND USE_TSAN)
  message(
    FATAL_ERROR
      "AddressSanitizer (ASan) and ThreadSanitizer (TSan) are mutually exclusive and cannot be enabled at the same time."
  )
endif()

set(SANITIZER_COMPILE_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

if(USE_ASAN)
  if(WIN32 AND MSVC)
    list(APPEND SANITIZER_COMPILE_FLAGS /fsanitize=address)
    list(APPEND SANITIZER_LINK_FLAGS /fsanitize=address)
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=address)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=address)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND SANITIZER_LINK_FLAGS -lpthread)
    endif()
  endif()
endif()

if(USE_UBSAN)
  if(WIN32 AND MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=undefined)
      list(APPEND SANITIZER_LINK_FLAGS -fsanitize=undefined)
    else()
      message(
        WARNING
          "UndefinedBehaviorSanitizer (UBSan) is not directly supported via a simple flag in MSVC."
      )
    endif()
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=undefined)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=undefined)
  endif()
endif()

if(USE_TSAN)
  if(WIN32 AND MSVC)
    message(
      FATAL_ERROR
        "ThreadSanitizer (TSan) is not supported by the MSVC compiler. Please use Clang or GCC."
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "ThreadSanitizer (TSan) is not supported on macOS.")
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=thread)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=thread)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND SANITIZER_LINK_FLAGS -lpthread)
    endif()
  endif()
endif()

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)
