include(CMakeParseArguments)

function(_mlx_metal_supports_logging OUTPUT_VARIABLE)
  execute_process(
    COMMAND xcrun -sdk macosx metal --help
    RESULT_VARIABLE _MLX_METAL_HELP_RESULT
    OUTPUT_VARIABLE _MLX_METAL_HELP_OUTPUT
    ERROR_VARIABLE _MLX_METAL_HELP_ERROR
    TIMEOUT 10)
  set(_MLX_METAL_HELP "${_MLX_METAL_HELP_OUTPUT}\n${_MLX_METAL_HELP_ERROR}")
  string(FIND "${_MLX_METAL_HELP}" "-fmetal-enable-logging"
              _MLX_METAL_LOGGING_OPTION_INDEX)
  if(_MLX_METAL_HELP_RESULT STREQUAL "0"
     AND NOT _MLX_METAL_LOGGING_OPTION_INDEX EQUAL -1)
    set(${OUTPUT_VARIABLE}
        TRUE
        PARENT_SCOPE)
  else()
    set(${OUTPUT_VARIABLE}
        FALSE
        PARENT_SCOPE)
  endif()
endfunction()

# clang format off
#
# ##############################################################################
# Build metal library
#
# Adds a custom target ${TARGET} to build ${OUTPUT_DIRECTORY}/{TITLE}.metallib.
# Each source in ${SOURCES} is compiled to an AIR file before the AIR files are
# linked into the metallib, including list ${INCLUDE_DIRS}, depends on list
# ${DEPS}, and passes ${COMPILE_OPTIONS} to the Metal compiler.
#
# Args: TARGET: Custom target to be added for the metal library TITLE: Name of
# the .metallib OUTPUT_DIRECTORY: Where to place ${TITLE}.metallib SOURCES: List
# of source files INCLUDE_DIRS: List of include dirs DEPS: List of dependency
# files (like headers) COMPILE_OPTIONS: Additional Metal compiler options DEBUG:
# Boolean, if true, enables debug compile options for this specific library. If
# not provided, uses global MLX_METAL_DEBUG.
#
# clang format on

macro(mlx_build_metallib)
  # Parse args
  set(oneValueArgs TARGET TITLE OUTPUT_DIRECTORY DEBUG)
  set(multiValueArgs SOURCES INCLUDE_DIRS DEPS COMPILE_OPTIONS)
  cmake_parse_arguments(MTLLIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set output
  set(MTLLIB_BUILD_TARGET "${MTLLIB_OUTPUT_DIRECTORY}/${MTLLIB_TITLE}.metallib")

  # Collect compile options
  set(_MTLLIB_COMPILE_OPTIONS
      -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions
      -Wmetal-addr-spaces)
  if(MLX_METAL_DEBUG
     OR MTLLIB_DEBUG
     OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_MTLLIB_COMPILE_OPTIONS ${_MTLLIB_COMPILE_OPTIONS} -gline-tables-only
                                -frecord-sources)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    _mlx_metal_supports_logging(_MTLLIB_SUPPORTS_LOGGING)
    if(_MTLLIB_SUPPORTS_LOGGING)
      list(APPEND _MTLLIB_COMPILE_OPTIONS -fmetal-enable-logging)
    endif()
  endif()

  set(_MTLLIB_LINK_OPTIONS)
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(_MTLLIB_DEPLOYMENT_OPTION
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    list(APPEND _MTLLIB_COMPILE_OPTIONS ${_MTLLIB_DEPLOYMENT_OPTION})
    list(APPEND _MTLLIB_LINK_OPTIONS ${_MTLLIB_DEPLOYMENT_OPTION})
  endif()

  # Compile each Metal source separately for incremental builds
  set(_MTLLIB_AIR_TARGETS)
  set(_MTLLIB_SOURCE_INDEX 0)
  foreach(_MTLLIB_SOURCE IN LISTS MTLLIB_SOURCES)
    get_filename_component(_MTLLIB_SOURCE_STEM "${_MTLLIB_SOURCE}" NAME_WE)
    set(_MTLLIB_AIR_NAME
        "${MTLLIB_TARGET}_${_MTLLIB_SOURCE_INDEX}_${_MTLLIB_SOURCE_STEM}.air")
    set(_MTLLIB_AIR_TARGET "${CMAKE_CURRENT_BINARY_DIR}/${_MTLLIB_AIR_NAME}")
    set(_MTLLIB_DEPFILE "${_MTLLIB_AIR_TARGET}.d")
    add_custom_command(
      OUTPUT ${_MTLLIB_AIR_TARGET}
      COMMAND
        xcrun -sdk macosx metal
        "$<LIST:TRANSFORM,${MTLLIB_INCLUDE_DIRS},PREPEND,-I>"
        ${_MTLLIB_COMPILE_OPTIONS} ${MTLLIB_COMPILE_OPTIONS} -MMD -MF
        ${_MTLLIB_DEPFILE} -MT ${_MTLLIB_AIR_TARGET} -c ${_MTLLIB_SOURCE} -o
        ${_MTLLIB_AIR_TARGET}
      DEPENDS ${MTLLIB_DEPS} ${_MTLLIB_SOURCE}
      DEPFILE ${_MTLLIB_DEPFILE}
      COMMAND_EXPAND_LISTS
      COMMENT "Building ${_MTLLIB_AIR_NAME}"
      VERBATIM)
    list(APPEND _MTLLIB_AIR_TARGETS ${_MTLLIB_AIR_TARGET})
    math(EXPR _MTLLIB_SOURCE_INDEX "${_MTLLIB_SOURCE_INDEX} + 1")
  endforeach()

  # Link the AIR files into a metallib
  add_custom_command(
    OUTPUT ${MTLLIB_BUILD_TARGET}
    COMMAND xcrun -sdk macosx metal ${_MTLLIB_LINK_OPTIONS}
            ${_MTLLIB_AIR_TARGETS} -o ${MTLLIB_BUILD_TARGET}
    DEPENDS ${_MTLLIB_AIR_TARGETS}
    COMMENT "Building ${MTLLIB_TITLE}.metallib"
    VERBATIM)

  # Add metallib custom target
  add_custom_target(${MTLLIB_TARGET} DEPENDS ${MTLLIB_BUILD_TARGET})

endmacro(mlx_build_metallib)
