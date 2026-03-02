# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified from
# https://github.com/NVIDIA/cudnn-frontend/blob/main/cmake/cuDNN.cmake

# Return the last file matching the pattern.
function(find_file_glob VAR PATTERN)
  file(GLOB _RESULT "${PATTERN}")
  if(_RESULT)
    list(LENGTH ${_RESULT} _RESULT_LENGTH)
    if(_RESULT_LENGTH GREATER 0)
      list(GET ${_RESULT} -1 _RESULT)
    endif()
    set(${VAR}
        "${_RESULT}"
        PARENT_SCOPE)
  endif()
endfunction()

# Find the dir including the "cudnn.h" file.
find_path(
  CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDNN_INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES include OPTIONAL)

# Glob searching "cudnn.h" for Windows.
if(WIN32 AND NOT CUDNN_INCLUDE_DIR)
  find_file_glob(
    CUDNN_H_PATH
    "C:/Program Files/NVIDIA/CUDNN/*/include/${CUDAToolkit_VERSION_MAJOR}.*/cudnn.h"
  )
  if(CUDNN_H_PATH)
    get_filename_component(CUDNN_INCLUDE_DIR "${CUDNN_H_PATH}" DIRECTORY)
  endif()
endif()

if(NOT CUDNN_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "Unable to find cudnn.h, please make sure cuDNN is installed and pass CUDNN_INCLUDE_PATH to cmake."
  )
endif()

# Get cudnn version.
file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" cudnn_version_header)
string(REGEX MATCH "#define CUDNN_MAJOR [1-9]+" macrodef
             "${cudnn_version_header}")
string(REGEX MATCH "[1-9]+" CUDNN_MAJOR_VERSION "${macrodef}")

# Function for searching library files.
function(find_cudnn_library NAME)
  if(NOT "${ARGV1}" STREQUAL "OPTIONAL")
    set(_CUDNN_REQUIRED TRUE)
  else()
    set(_CUDNN_REQUIRED FALSE)
  endif()

  find_library(
    ${NAME}_LIBRARY
    NAMES ${NAME} "lib${NAME}.so.${CUDNN_MAJOR_VERSION}" NAMES_PER_DIR
    HINTS ${CUDNN_LIBRARY_PATH} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib64 lib/x64 lib OPTIONAL)

  if(WIN32 AND NOT ${NAME}_LIBRARY)
    find_file_glob(
      ${NAME}_LIBRARY
      "C:/Program Files/NVIDIA/CUDNN/*/lib/${CUDAToolkit_VERSION_MAJOR}.*/x64/${NAME}.lib"
    )
  endif()

  if(NOT ${NAME}_LIBRARY AND ${_CUDNN_REQUIRED})
    message(
      FATAL_ERROR
        "Unable to find ${NAME}, please make sure cuDNN is installed and pass CUDNN_LIBRARY_PATH to cmake."
    )
  endif()

  if(${NAME}_LIBRARY)
    add_library(CUDNN::${NAME} UNKNOWN IMPORTED)
    set_target_properties(
      CUDNN::${NAME}
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
                 IMPORTED_LOCATION ${${NAME}_LIBRARY})
    set(${NAME}_LIBRARY
        "${${NAME}_LIBRARY}"
        PARENT_SCOPE)
  else()
    message(STATUS "${NAME} not found.")
  endif()
endfunction()

# Search for the main cudnn library.
find_cudnn_library(cudnn)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN REQUIRED_VARS CUDNN_INCLUDE_DIR
                                                      cudnn_LIBRARY)

if(CUDNN_INCLUDE_DIR AND cudnn_LIBRARY)
  set(CUDNN_FOUND
      ON
      CACHE INTERNAL "cuDNN Library Found")
else()
  set(CUDNN_FOUND
      OFF
      CACHE INTERNAL "cuDNN Library Not Found")
endif()

# Find out all the DLL files for Windows.
if(WIN32 AND cudnn_LIBRARY)
  get_filename_component(CUDNN_BIN_DIR "${cudnn_LIBRARY}" DIRECTORY)
  string(REPLACE "/lib/" "/bin/" CUDNN_BIN_DIR "${CUDNN_BIN_DIR}")
  file(
    GLOB CUDNN_DLL_NAMES
    RELATIVE "${CUDNN_BIN_DIR}"
    "${CUDNN_BIN_DIR}/*.dll")
endif()

# Create an interface library that users can link with.
add_library(CUDNN::cudnn_all INTERFACE IMPORTED)
target_link_libraries(CUDNN::cudnn_all INTERFACE CUDNN::cudnn)
target_include_directories(
  CUDNN::cudnn_all INTERFACE $<INSTALL_INTERFACE:include>
                             $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>)

# Add other components of cudnn.
if(CUDNN_MAJOR_VERSION EQUAL 8)
  find_cudnn_library(cudnn_adv_infer)
  find_cudnn_library(cudnn_adv_train)
  find_cudnn_library(cudnn_cnn_infer)
  find_cudnn_library(cudnn_cnn_train)
  find_cudnn_library(cudnn_ops_infer)
  find_cudnn_library(cudnn_ops_train)

  target_link_libraries(
    CUDNN::cudnn_all
    INTERFACE CUDNN::cudnn_adv_train CUDNN::cudnn_ops_train
              CUDNN::cudnn_cnn_train CUDNN::cudnn_adv_infer
              CUDNN::cudnn_cnn_infer CUDNN::cudnn_ops_infer)

elseif(CUDNN_MAJOR_VERSION EQUAL 9)
  find_cudnn_library(cudnn_graph)
  find_cudnn_library(cudnn_engines_runtime_compiled)
  find_cudnn_library(cudnn_ops OPTIONAL)
  find_cudnn_library(cudnn_cnn OPTIONAL)
  find_cudnn_library(cudnn_adv OPTIONAL)
  find_cudnn_library(cudnn_engines_precompiled OPTIONAL)
  find_cudnn_library(cudnn_heuristic OPTIONAL)

  target_link_libraries(
    CUDNN::cudnn_all
    INTERFACE CUDNN::cudnn_graph
              CUDNN::cudnn_engines_runtime_compiled
              CUDNN::cudnn_ops
              CUDNN::cudnn_cnn
              CUDNN::cudnn_adv
              CUDNN::cudnn_engines_precompiled
              CUDNN::cudnn_heuristic)
endif()
