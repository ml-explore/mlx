# Based on: https://github.com/sivachandran/cmake-bin2h
#
# Copyright 2020 Sivachandran Paramasivam
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(CMakeParseArguments)

# Function to wrap a given string into multiple lines at the given column
# position.
#
# Parameters:
#
# * VARIABLE - The name of the CMake variable holding the string.
# * AT_COLUMN - The column position at which string will be wrapped.
function(WRAP_STRING)
  set(oneValueArgs VARIABLE AT_COLUMN)
  cmake_parse_arguments(WRAP_STRING "${options}" "${oneValueArgs}" "" ${ARGN})

  string(LENGTH ${${WRAP_STRING_VARIABLE}} stringLength)
  math(EXPR offset "0")

  while(stringLength GREATER 0)
    if(stringLength GREATER ${WRAP_STRING_AT_COLUMN})
      math(EXPR length "${WRAP_STRING_AT_COLUMN}")
    else()
      math(EXPR length "${stringLength}")
    endif()

    string(SUBSTRING ${${WRAP_STRING_VARIABLE}} ${offset} ${length} line)
    set(lines "${lines}\n ${line}")

    math(EXPR stringLength "${stringLength} - ${length}")
    math(EXPR offset "${offset} + ${length}")
  endwhile()

  set(${WRAP_STRING_VARIABLE}
      "${lines}"
      PARENT_SCOPE)
endfunction()

# Function to embed contents of a file as byte array in C/C++ header file(.h).
# The header file will contain a byte array and integer variable holding the
# size of the array.
#
# Parameters:
#
# * SOURCE_FILES - The paths of source files whose contents will be embedded in
#   the header file.
# * VARIABLE_NAME - The name of the variable for the byte array. The string
#   "_SIZE" will be append to this name and will be used a variable name for
#   size variable.
# * HEADER_FILE - The path of header file.
# * APPEND - If specified appends to the header file instead of overwriting it
# * HEADER_NAMESPACE - The namespace, where the array should be located in.
# * NULL_TERMINATE - If specified a null byte(zero) will be append to the byte
#   array.
#
# Usage:
#
# bin2h(SOURCE_FILE "Logo.png" HEADER_FILE "Logo.h" VARIABLE_NAME "LOGO_PNG")
function(BIN2H)
  set(options APPEND NULL_TERMINATE)
  set(oneValueArgs VARIABLE_NAME HEADER_FILE HEADER_NAMESPACE)
  set(multiValueArgs SOURCE_FILES)
  cmake_parse_arguments(BIN2H "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(arrayDefinition "")
  foreach(SOURCE_FILE IN LISTS BIN2H_SOURCE_FILES)
    # get filename without extension
    get_filename_component(FILE_NAME_WE ${SOURCE_FILE} NAME_WE)
    # convert the filename to a valid C identifier
    string(MAKE_C_IDENTIFIER "${FILE_NAME_WE}" VALID_FILE_NAME)

    # reads source file contents as hex string
    file(READ ${SOURCE_FILE} hexString HEX)

    # append null
    if(BIN2H_NULL_TERMINATE)
      string(APPEND hexString "00")
    endif()

    # wraps the hex string into multiple lines
    wrap_string(VARIABLE hexString AT_COLUMN 24)

    # strip the © in source code
    string(REGEX REPLACE "c2a9" "2020" arrayValues ${hexString})

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" " 0x\\1," arrayValues
                         ${arrayValues})

    # make a full variable name for the array
    set(FULL_VARIABLE_NAME "${BIN2H_VARIABLE_NAME}_${VALID_FILE_NAME}")

    # declares byte array and the length variables
    string(APPEND arrayDefinition
           "constexpr char ${FULL_VARIABLE_NAME}[] = {${arrayValues}\n};\n\n")
  endforeach()

  # add namespace wrapper if defined
  if(DEFINED BIN2H_HEADER_NAMESPACE)
    set(namespaceStart "namespace ${BIN2H_HEADER_NAMESPACE} {")
    set(namespaceEnd "} // namespace ${BIN2H_HEADER_NAMESPACE}")
    set(declarations "${namespaceStart}\n\n${arrayDefinition}${namespaceEnd}\n")
  endif()

  set(arrayIncludes "#pragma once")
  string(PREPEND declarations "${arrayIncludes}\n\n")

  if(BIN2H_APPEND)
    file(APPEND ${BIN2H_HEADER_FILE} "${declarations}")
  else()
    file(WRITE ${BIN2H_HEADER_FILE} "${declarations}")
  endif()
endfunction()

# ----------------------------- CLI args -----------------------------

string(REPLACE ":" ";" MLX_JIT_SOURCES_LIST ${MLX_JIT_SOURCES})
foreach(source ${MLX_JIT_SOURCES_LIST})
  list(APPEND MLX_JIT_SOURCES_ABS "${MLX_SOURCE_ROOT}/${source}")
endforeach()

bin2h(
  SOURCE_FILES
  ${MLX_JIT_SOURCES_ABS}
  NULL_TERMINATE
  VARIABLE_NAME
  "jit_source"
  HEADER_NAMESPACE
  "mlx::core"
  HEADER_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/gen/cuda_jit_sources.h")
