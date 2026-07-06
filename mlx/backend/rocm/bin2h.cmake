# Copyright © 2025 Apple Inc.

# Script to embed kernel source files as header for JIT compilation

set(MLX_OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen/rocm_jit_sources.h")
set(MLX_KERNEL_HEADER
    "#pragma once\n\n#include <unordered_map>\n#include <string>\n\nnamespace mlx::core::rocm {\n\n"
)
set(MLX_KERNEL_FOOTER "\n} // namespace mlx::core::rocm\n")

# Create output directory
get_filename_component(MLX_OUTPUT_DIR ${MLX_OUTPUT_FILE} DIRECTORY)
file(MAKE_DIRECTORY ${MLX_OUTPUT_DIR})

# Write header
file(WRITE ${MLX_OUTPUT_FILE} ${MLX_KERNEL_HEADER})

# Process JIT sources
string(REPLACE ":" ";" MLX_JIT_SOURCES_LIST ${MLX_JIT_SOURCES})

set(MLX_SOURCE_MAP
    "const std::unordered_map<std::string, std::string> kernel_sources = {\n")

foreach(source IN LISTS MLX_JIT_SOURCES_LIST)
  set(source_file "${MLX_SOURCE_ROOT}/${source}")
  if(EXISTS ${source_file})
    # Read source file
    file(READ ${source_file} source_content)

    # Escape content for C++ string literal
    string(REPLACE "\\" "\\\\" source_content "${source_content}")
    string(REPLACE "\"" "\\\"" source_content "${source_content}")
    string(REPLACE "\n" "\\n\"\n\"" source_content "${source_content}")

    # Add to map
    set(MLX_SOURCE_MAP
        "${MLX_SOURCE_MAP}  {\"${source}\", \"${source_content}\"},\n")
  endif()
endforeach()

set(MLX_SOURCE_MAP "${MLX_SOURCE_MAP}};\n")

# Write source map
file(APPEND ${MLX_OUTPUT_FILE} ${MLX_SOURCE_MAP})

# Write footer
file(APPEND ${MLX_OUTPUT_FILE} ${MLX_KERNEL_FOOTER})
