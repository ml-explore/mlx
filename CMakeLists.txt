cmake_minimum_required(VERSION 3.24)

project(mlx LANGUAGES CXX)

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
option(BUILD_SHARED_LIBS "Build mlx as a shared library" OFF)

if(NOT MLX_VERSION)
  set(MLX_VERSION 0.0.6)
endif()

# --------------------- Processor tests -------------------------

message(STATUS "Building MLX for ${CMAKE_HOST_SYSTEM_PROCESSOR} processor on ${CMAKE_SYSTEM_NAME}")

set(MLX_BUILD_ARM OFF)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

  if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    message(WARNING 
      "Building for x86_64 on macOS is not supported." 
      " If you are on an Apple silicon system, "
      " make sure you are building for arm64.")
  elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
    set(MLX_BUILD_ARM ON)
  endif()

else()
  message(WARNING "MLX is prioritised for Apple silicon systems using macOS.")
endif()

# ----------------------------- Lib -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

add_library(mlx)

if (MLX_BUILD_METAL)
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  find_library(QUARTZ_LIB QuartzCore)
endif()

if (MLX_BUILD_METAL AND NOT METAL_LIB)
  message(STATUS "Metal not found. Unable to build GPU")
  set(MLX_BUILD_METAL OFF)
elseif (MLX_BUILD_METAL)
  message(STATUS "Building METAL sources")
  add_compile_definitions(_METAL_)

  # Throw an error if xcrun not found
  execute_process(COMMAND zsh "-c" "/usr/bin/xcrun -sdk macosx --show-sdk-version"
                  OUTPUT_VARIABLE MACOS_VERSION
                  COMMAND_ERROR_IS_FATAL ANY)

  message(STATUS "Building with SDK for macOS version ${MACOS_VERSION}")
  
  if (${MACOS_VERSION} GREATER_EQUAL 14.2)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip)
  elseif (${MACOS_VERSION} GREATER_EQUAL 14.0)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14_iOS17-beta.zip)
  elseif (${MACOS_VERSION} GREATER_EQUAL 13.3)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13.3_iOS16.4.zip)
  else()
    message(FATAL_ERROR "MLX requires macOS >= 13.4 to be built with MLX_BUILD_METAL=ON" )
  endif()

  FetchContent_Declare(
    metal_cpp
    URL ${METAL_CPP_URL}
  )

  FetchContent_MakeAvailable(metal_cpp)
  target_include_directories(
    mlx PUBLIC
    $<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/metal_cpp>
  )
  target_link_libraries(
    mlx
    ${METAL_LIB}
    ${FOUNDATION_LIB}
    ${QUARTZ_LIB})
endif()

find_library(ACCELERATE_LIBRARY Accelerate)
if (MLX_BUILD_ARM AND ACCELERATE_LIBRARY)
  message(STATUS "Accelerate found ${ACCELERATE_LIBRARY}")
  set(MLX_BUILD_ACCELERATE ON)
  target_link_libraries(mlx ${ACCELERATE_LIBRARY})
  add_compile_definitions(ACCELERATE_NEW_LAPACK)
else()
  message(STATUS "Accelerate or arm neon not found, using default backend.")
  set(MLX_BUILD_ACCELERATE OFF)
  #set(BLA_VENDOR Generic)
  find_package(BLAS REQUIRED)
  if (NOT BLAS_FOUND)
    message(FATAL_ERROR "Must have BLAS installed")
  endif()
  # TODO find a cleaner way to do this
  find_path(BLAS_INCLUDE_DIRS cblas.h
    /usr/include
    /usr/local/include
    $ENV{BLAS_HOME}/include)
  message(STATUS ${BLAS_LIBRARIES})
  message(STATUS ${BLAS_INCLUDE_DIRS})
  target_include_directories(mlx PRIVATE ${BLAS_INCLUDE_DIRS})
  target_link_libraries(mlx ${BLAS_LIBRARIES})
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/mlx)

target_include_directories(
  mlx 
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>
)

if (MLX_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
  find_package(Python COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/python/src)
endif()

if (MLX_BUILD_TESTS)
  include(CTest)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif()

if (MLX_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples/cpp)
endif()

if (MLX_BUILD_BENCHMARKS)
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
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mlx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)

# Install metal dependencies
if (MLX_BUILD_METAL)

  # Install metal cpp
  install(
      DIRECTORY ${metal_cpp_SOURCE_DIR}/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metal_cpp
      COMPONENT metal_cpp_source
  )

endif()

# Install cmake config
set(MLX_CMAKE_BUILD_CONFIG ${CMAKE_BINARY_DIR}/MLXConfig.cmake)
set(MLX_CMAKE_BUILD_VERSION_CONFIG ${CMAKE_BINARY_DIR}/MLXConfigVersion.cmake)
set(MLX_CMAKE_INSTALL_MODULE_DIR share/cmake/MLX)

install(
  EXPORT MLXTargets
  FILE MLXTargets.cmake
  DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${MLX_CMAKE_BUILD_VERSION_CONFIG}
  COMPATIBILITY SameMajorVersion
  VERSION ${MLX_VERSION}
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mlx.pc.in
  ${MLX_CMAKE_BUILD_CONFIG}
  INSTALL_DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR MLX_CMAKE_INSTALL_MODULE_DIR
)

install(
  FILES ${MLX_CMAKE_BUILD_CONFIG} ${MLX_CMAKE_BUILD_VERSION_CONFIG}
  DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR}
)

install(
  DIRECTORY ${CMAKE_MODULE_PATH}/
  DESTINATION ${MLX_CMAKE_INSTALL_MODULE_DIR}
)
