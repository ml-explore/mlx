cmake_minimum_required(VERSION 3.25)

if(NOT MLX_VERSION)
  file(STRINGS "mlx/version.h" _mlx_h_version REGEX "^#define MLX_VERSION_.*$")
  string(REGEX MATCH "#define MLX_VERSION_MAJOR ([0-9]+)" _ "${_mlx_h_version}")
  set(_major ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define MLX_VERSION_MINOR ([0-9]+)" _ "${_mlx_h_version}")
  set(_minor ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define MLX_VERSION_PATCH ([0-9]+)" _ "${_mlx_h_version}")
  set(_patch ${CMAKE_MATCH_1})
  set(MLX_PROJECT_VERSION "${_major}.${_minor}.${_patch}")
  set(MLX_VERSION ${MLX_PROJECT_VERSION})
else()
  string(REGEX REPLACE "^([0-9]+\.[0-9]+\.[0-9]+).*" "\\1" MLX_PROJECT_VERSION
                       ${MLX_VERSION})
endif()

project(
  mlx
  LANGUAGES C CXX
  VERSION ${MLX_PROJECT_VERSION})

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_MESSAGE NEVER)

# ----------------------------- Configuration -----------------------------
option(MLX_BUILD_TESTS "Build tests for mlx" ON)
option(MLX_BUILD_EXAMPLES "Build examples for mlx" ON)
option(MLX_BUILD_BENCHMARKS "Build benchmarks for mlx" OFF)
option(MLX_BUILD_PYTHON_BINDINGS "Build python bindings for mlx" OFF)
option(MLX_BUILD_METAL "Build metal backend" ON)
option(MLX_BUILD_CPU "Build cpu backend" ON)
option(MLX_BUILD_CUDA "Build cuda backend" OFF)
option(MLX_METAL_DEBUG "Enhance metal debug workflow" OFF)
option(MLX_ENABLE_X64_MAC "Enable building for x64 macOS" OFF)
option(MLX_BUILD_GGUF "Include support for GGUF format" ON)
option(MLX_BUILD_SAFETENSORS "Include support for safetensors format" ON)
option(MLX_BUILD_BLAS_FROM_SOURCE "Build OpenBLAS from source code" OFF)
option(MLX_METAL_JIT "Use JIT compilation for Metal kernels" OFF)
option(MLX_USE_CCACHE "Use CCache for compilation cache when available" ON)
option(BUILD_SHARED_LIBS "Build mlx as a shared library" OFF)
option(USE_SYSTEM_FMT "Use system's provided fmt library" OFF)

# --------------------- Processor tests -------------------------
message(
  STATUS
    "Building MLX for ${CMAKE_SYSTEM_PROCESSOR} processor on ${CMAKE_SYSTEM_NAME}"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    if(NOT MLX_ENABLE_X64_MAC)
      message(
        FATAL_ERROR
          "Building for x86_64 on macOS is not supported."
          " If you are on an Apple silicon system, check the build"
          " documentation for possible fixes: "
          "https://ml-explore.github.io/mlx/build/html/install.html#build-from-source"
      )
    else()
      set(MLX_BUILD_METAL OFF)
      message(WARNING "Building for x86_64 arch is not officially supported.")
    endif()
  endif()
else()
  set(MLX_BUILD_METAL OFF)
endif()

if(MLX_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

# ----------------------------- Lib -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

add_library(mlx)

if(MLX_BUILD_CUDA)
  enable_language(CUDA)
endif()

if(MLX_BUILD_METAL)
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  find_library(QUARTZ_LIB QuartzCore)
  if(METAL_LIB)
    message(STATUS "Metal found ${METAL_LIB}")
  else()
    message(
      FATAL_ERROR
        "Metal not found. Set MLX_BUILD_METAL=OFF to build without GPU")
  endif()

  if(MLX_METAL_DEBUG)
    add_compile_definitions(MLX_METAL_DEBUG)
  endif()

  # Throw an error if xcrun not found
  execute_process(
    COMMAND zsh "-c" "/usr/bin/xcrun -sdk macosx --show-sdk-version"
    OUTPUT_VARIABLE MACOS_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

  if(${MACOS_SDK_VERSION} LESS 14.0)
    message(
      FATAL_ERROR
        "MLX requires macOS SDK >= 14.0 to be built with MLX_BUILD_METAL=ON")
  endif()
  message(STATUS "Building with macOS SDK version ${MACOS_SDK_VERSION}")

  set(METAL_CPP_URL
      https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip)

  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(XCRUN_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  execute_process(
    COMMAND
      zsh "-c"
      "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1 | tr -d '\n'"
    OUTPUT_VARIABLE MLX_METAL_VERSION COMMAND_ERROR_IS_FATAL ANY)
  FetchContent_Declare(metal_cpp URL ${METAL_CPP_URL})

  FetchContent_MakeAvailable(metal_cpp)
  target_include_directories(
    mlx PUBLIC $<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>
               $<INSTALL_INTERFACE:include/metal_cpp>)
  target_link_libraries(mlx PUBLIC ${METAL_LIB} ${FOUNDATION_LIB} ${QUARTZ_LIB})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # With newer clang/gcc versions following libs are implicitly linked, but when
  # building on old distributions they need to be explicitly listed.
  target_link_libraries(mlx PRIVATE dl pthread)
endif()

if(WIN32)
  if(MSVC)
    # GGUF does not build with MSVC.
    set(MLX_BUILD_GGUF OFF)
    # There is no prebuilt OpenBLAS distribution for MSVC.
    set(MLX_BUILD_BLAS_FROM_SOURCE ON)
  endif()
  # Windows implementation of dlfcn.h APIs.
  FetchContent_Declare(
    dlfcn-win32
    GIT_REPOSITORY https://github.com/dlfcn-win32/dlfcn-win32.git
    GIT_TAG v1.4.1
    EXCLUDE_FROM_ALL)
  block()
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(dlfcn-win32)
  endblock()
  target_include_directories(mlx PRIVATE "${dlfcn-win32_SOURCE_DIR}/src")
  target_link_libraries(mlx PRIVATE dl)
endif()

if(MLX_BUILD_CPU)
  find_library(ACCELERATE_LIBRARY Accelerate)
  if(ACCELERATE_LIBRARY)
    message(STATUS "Accelerate found ${ACCELERATE_LIBRARY}")
    set(MLX_BUILD_ACCELERATE ON)
  else()
    message(STATUS "Accelerate or arm neon not found, using default backend.")
    set(MLX_BUILD_ACCELERATE OFF)
  endif()

  if(MLX_BUILD_ACCELERATE)
    target_link_libraries(mlx PUBLIC ${ACCELERATE_LIBRARY})
    add_compile_definitions(MLX_USE_ACCELERATE)
    add_compile_definitions(ACCELERATE_NEW_LAPACK)
  elseif(MLX_BUILD_BLAS_FROM_SOURCE)
    # Download and build OpenBLAS from source code.
    FetchContent_Declare(
      openblas
      GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
      GIT_TAG v0.3.28
      EXCLUDE_FROM_ALL)
    set(BUILD_STATIC_LIBS ON) # link statically
    set(NOFORTRAN ON) # msvc has no fortran compiler
    FetchContent_MakeAvailable(openblas)
    target_link_libraries(mlx PRIVATE openblas)
    target_include_directories(
      mlx PRIVATE "${openblas_SOURCE_DIR}/lapack-netlib/LAPACKE/include"
                  "${CMAKE_BINARY_DIR}/generated" "${CMAKE_BINARY_DIR}")
  else()
    if(${CMAKE_HOST_APPLE})
      # The blas shipped in macOS SDK is not supported, search homebrew for
      # openblas instead.
      set(BLA_VENDOR OpenBLAS)
      set(LAPACK_ROOT
          "${LAPACK_ROOT};$ENV{LAPACK_ROOT};/usr/local/opt/openblas")
    endif()
    # Search and link with lapack.
    find_package(LAPACK REQUIRED)
    if(NOT LAPACK_FOUND)
      message(FATAL_ERROR "Must have LAPACK installed")
    endif()
    find_path(LAPACK_INCLUDE_DIRS lapacke.h /usr/include /usr/local/include
              /usr/local/opt/openblas/include)
    message(STATUS "Lapack lib " ${LAPACK_LIBRARIES})
    message(STATUS "Lapack include " ${LAPACK_INCLUDE_DIRS})
    target_include_directories(mlx PRIVATE ${LAPACK_INCLUDE_DIRS})
    target_link_libraries(mlx PRIVATE ${LAPACK_LIBRARIES})
    # List blas after lapack otherwise we may accidentally incldue an old
    # version of lapack.h from the include dirs of blas.
    find_package(BLAS REQUIRED)
    if(NOT BLAS_FOUND)
      message(FATAL_ERROR "Must have BLAS installed")
    endif()
    # TODO find a cleaner way to do this
    find_path(BLAS_INCLUDE_DIRS cblas.h /usr/include /usr/local/include
              $ENV{BLAS_HOME}/include)
    message(STATUS "Blas lib " ${BLAS_LIBRARIES})
    message(STATUS "Blas include " ${BLAS_INCLUDE_DIRS})
    target_include_directories(mlx PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(mlx PRIVATE ${BLAS_LIBRARIES})
  endif()
else()
  set(MLX_BUILD_ACCELERATE OFF)
endif()

message(STATUS "Downloading json")
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_include_directories(
  mlx PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/mlx)

target_include_directories(
  mlx PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
             $<INSTALL_INTERFACE:include>)

# Do not add mlx_EXPORTS define for shared library.
set_target_properties(mlx PROPERTIES DEFINE_SYMBOL "")

if(USE_SYSTEM_FMT)
  find_package(fmt REQUIRED)
else()
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(fmt)
endif()
target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

if(MLX_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
  find_package(
    Python 3.8
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/python/src)
endif()

if(MLX_BUILD_TESTS)
  include(CTest)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif()

if(MLX_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples/cpp)
endif()

if(MLX_BUILD_BENCHMARKS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/benchmarks/cpp)
endif()

# ----------------------------- Installation -----------------------------
include(GNUInstallDirs)

# Install library
install(
  TARGETS mlx
  EXPORT MLXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mlx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "backend/metal/kernels.h" EXCLUDE)

# Install metal dependencies
if(MLX_BUILD_METAL)

  # Install metal cpp
  install(
    DIRECTORY ${metal_cpp_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metal_cpp
    COMPONENT metal_cpp_source)

endif()

# Install cmake config
set(MLX_CMAKE_BUILD_CONFIG ${CMAKE_BINARY_DIR}/MLXConfig.cmake)
set(MLX_CMAKE_BUILD_VERSION_CONFIG ${CMAKE_BINARY_DIR}/MLXConfigVersion.cmake)
set(MLX_CMAKE_INSTALL_MODULE_DIR share/cmake/MLX)

install(
  EXPORT MLXTargets
  FILE MLXTargets.cmake
  DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${MLX_CMAKE_BUILD_VERSION_CONFIG}
  COMPATIBILITY SameMajorVersion
  VERSION ${MLX_VERSION})

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mlx.pc.in ${MLX_CMAKE_BUILD_CONFIG}
  INSTALL_DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
            MLX_CMAKE_INSTALL_MODULE_DIR)

install(FILES ${MLX_CMAKE_BUILD_CONFIG} ${MLX_CMAKE_BUILD_VERSION_CONFIG}
        DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR})

install(DIRECTORY ${CMAKE_MODULE_PATH}/
        DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR})
