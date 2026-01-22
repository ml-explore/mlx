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

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})

# On Windows with MSVC and static libraries, the linker will discard object
# files that are not referenced, which removes CUDA static initializers that
# register kernels with the runtime. /WHOLEARCHIVE forces all object files to be
# included. Also disable ICF/REF optimizations which can merge or remove CUDA
# registration code.
if(WIN32
   AND MSVC
   AND NOT BUILD_SHARED_LIBS
   AND MLX_BUILD_CUDA)
  target_link_options(
    tests PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:mlx>"
    "$<$<CONFIG:Release>:/OPT:NOICF,NOREF>"
    "$<$<CONFIG:RelWithDebInfo>:/OPT:NOICF,NOREF>")
  # Use dynamic CUDA runtime for tests executable. This must match the runtime
  # used by mlx.lib. CMake's CUDA support defaults to static runtime when
  # linking executables with CUDA code, so we must explicitly override it.
  set_target_properties(tests PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  # Explicitly link against the dynamic cudart import library using the full
  # path. CUDAToolkit_LIBRARY_DIR should be set by find_package(CUDAToolkit) in
  # the main CMakeLists.
  target_link_libraries(tests PRIVATE "${CUDAToolkit_LIBRARY_DIR}/cudart.lib")
endif()

# On Windows, copy the mlx DLL to the test executable directory for runtime
# loading. This POST_BUILD command must be added BEFORE doctest_discover_tests
# is called, because doctest_discover_tests also adds a POST_BUILD command that
# runs the test executable to enumerate tests. CMake executes POST_BUILD
# commands in the order they are added, so our copy command runs first.
if(WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mlx>
            $<TARGET_FILE_DIR:tests>)

  # Copy cuDNN DLL (find_library finds .lib but DLLs are in different location)
  if(MLX_BUILD_CUDA AND CUDNN_FOUND)
    # cuDNN DLLs are in bin/ directory parallel to lib/ where .lib files are
    # e.g., C:/Program Files/NVIDIA/CUDNN/v9.15/lib/13.0/x64/cudnn.lib ->
    # C:/Program Files/NVIDIA/CUDNN/v9.15/bin/13.0/cudnn64_9.dll
    get_filename_component(CUDNN_LIB_DIR "${cudnn_LIBRARY}" DIRECTORY)
    # Go up from lib/<version>/x64 to the cudnn root
    get_filename_component(CUDNN_ROOT "${CUDNN_LIB_DIR}/../../.." ABSOLUTE)
    # Extract version directory (e.g., "13.0") from parent of x64
    get_filename_component(CUDNN_VERSION_DIR "${CUDNN_LIB_DIR}/.." ABSOLUTE)
    get_filename_component(CUDNN_VERSION_DIR "${CUDNN_VERSION_DIR}" NAME)
    set(CUDNN_BIN_DIR "${CUDNN_ROOT}/bin/${CUDNN_VERSION_DIR}")
    # file(GLOB) runs at configure time, which is fine since cuDNN location
    # doesn't change between configure and build
    file(GLOB CUDNN_DLLS "${CUDNN_BIN_DIR}/cudnn*.dll")
    if(CUDNN_DLLS)
      add_custom_command(
        TARGET tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDNN_DLLS}
                $<TARGET_FILE_DIR:tests>
        COMMENT "Copying cuDNN DLLs for tests")
      message(STATUS "Will copy cuDNN DLLs from ${CUDNN_BIN_DIR} for tests")
    else()
      message(WARNING "cuDNN DLLs not found in ${CUDNN_BIN_DIR}")
    endif()
  endif()
endif()

doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)

if(WIN32 AND BUILD_SHARED_LIBS)
  # Also set PATH for test runtime via ENVIRONMENT_MODIFICATION so ctest can
  # find any DLLs that weren't tracked by TARGET_RUNTIME_DLLS (e.g., external
  # CUDA/cuDNN DLLs that aren't IMPORTED targets).
  set_property(
    TEST tests PROPERTY ENVIRONMENT_MODIFICATION
                        PATH=path_list_prepend:$<TARGET_RUNTIME_DLL_DIRS:tests>)
endif()
