# FindNCCL.cmake This module finds the NVIDIA NCCL library and its include
# directories.

set(NCCL_ROOT_DIR
    $ENV{NCCL_ROOT_DIR}
    CACHE PATH "Folder contains NVIDIA NCCL")

find_path(
  NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR} ${NCCL_ROOT_DIR} ${NCCL_ROOT_DIR}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include)

if($ENV{USE_STATIC_NCCL})
  message(
    STATUS "USE_STATIC_NCCL detected. Linking against static NCCL library")
  set(NCCL_LIBNAME "libnccl_static.a")
else()
  set(NCCL_LIBNAME "nccl")
endif()

find_library(
  NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR}
        ${NCCL_ROOT_DIR}
        ${NCCL_ROOT_DIR}/lib
        ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
        ${NCCL_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS
                                  NCCL_LIBRARIES)

if(NCCL_FOUND)
  set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message(
    STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
  file(
    STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
    REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$"
    LIMIT_COUNT 1)
  if(NCCL_MAJOR_VERSION_DEFINED)
    string(REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                         NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
    message(STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}")
  endif()
  message(
    STATUS
      "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()
