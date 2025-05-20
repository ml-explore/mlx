include(CMakeParseArguments)

# clang format off
#
# ##############################################################################
# Build metal library
#
# Adds a custom target ${TARGET} to build ${OUTPUT_DIRECTORY}/{TITLE}.metallib
# from list ${SOURCES}, including list ${INCLUDE_DIRS}, depends on list ${DEPS}
#
# Args: TARGET: Custom target to be added for the metal library TITLE: Name of
# the .metallib OUTPUT_DIRECTORY: Where to place ${TITLE}.metallib SOURCES: List
# of source files INCLUDE_DIRS: List of include dirs DEPS: List of dependency
# files (like headers) DEBUG: Boolean, if true, enables debug compile options
# for this specific library. If not provided, uses global MLX_METAL_DEBUG.
#
# clang format on

macro(mlx_build_metallib)
  # Parse args
  set(oneValueArgs TARGET TITLE OUTPUT_DIRECTORY DEBUG)
  set(multiValueArgs SOURCES INCLUDE_DIRS DEPS)
  cmake_parse_arguments(MTLLIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set output
  set(MTLLIB_BUILD_TARGET "${MTLLIB_OUTPUT_DIRECTORY}/${MTLLIB_TITLE}.metallib")

  # Collect compile options
  set(MTLLIB_COMPILE_OPTIONS -Wall -Wextra -fno-fast-math -Wno-c++17-extensions)
  if(MLX_METAL_DEBUG OR MTLLIB_DEBUG)
    set(MTLLIB_COMPILE_OPTIONS ${MTLLIB_COMPILE_OPTIONS} -gline-tables-only
                               -frecord-sources)
  endif()

  # Prepare metallib build command
  add_custom_command(
    OUTPUT ${MTLLIB_BUILD_TARGET}
    COMMAND
      xcrun -sdk macosx metal
      "$<LIST:TRANSFORM,${MTLLIB_INCLUDE_DIRS},PREPEND,-I>"
      ${MTLLIB_COMPILE_OPTIONS} ${MTLLIB_SOURCES} -o ${MTLLIB_BUILD_TARGET}
    DEPENDS ${MTLLIB_DEPS} ${MTLLIB_SOURCES}
    COMMAND_EXPAND_LISTS
    COMMENT "Building ${MTLLIB_TITLE}.metallib"
    VERBATIM)

  # Add metallib custom target
  add_custom_target(${MTLLIB_TARGET} DEPENDS ${MTLLIB_BUILD_TARGET})

endmacro(mlx_build_metallib)
