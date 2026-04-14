cmake_minimum_required(VERSION 3.24)
project(jaccl LANGUAGES CXX)

# Default to Release when built standalone. Without this CMake uses an empty
# build type (-O0) which severely impacts the reduction / memcpy hot paths.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# nlohmann/json for device file parsing
message(STATUS "Downloading json for JACCL")
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# Check platform and SDK version requirements
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(STATUS "JACCL requires macOS (Darwin). Skipping JACCL build.")
  return()
endif()

# Try to determine MACOS_SDK_VERSION if not set
if(NOT DEFINED MACOS_SDK_VERSION)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-version
    OUTPUT_VARIABLE MACOS_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(MACOS_SDK_VERSION)
    message(STATUS "Detected macOS SDK version: ${MACOS_SDK_VERSION}")
  endif()
endif()

if(DEFINED MACOS_SDK_VERSION AND MACOS_SDK_VERSION VERSION_LESS "26.2")
  message(STATUS "JACCL requires macOS SDK >= 26.2. Skipping JACCL build.")
  return()
endif()

add_library(
  jaccl
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/tcp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/rdma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/ring.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/jaccl.cpp)

target_include_directories(
  jaccl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
               $<INSTALL_INTERFACE:include>)

target_include_directories(
  jaccl PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)

target_compile_features(jaccl PUBLIC cxx_std_20)

# Install targets
install(
  TARGETS jaccl
  EXPORT jacclTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/
  DESTINATION include/jaccl
  FILES_MATCHING
  PATTERN "*.h")

install(
  EXPORT jacclTargets
  FILE jacclTargets.cmake
  NAMESPACE jaccl::
  DESTINATION lib/cmake/jaccl)
