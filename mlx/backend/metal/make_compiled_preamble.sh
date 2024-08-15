#!/bin/bash
#
# This script generates a C++ function that provides the Metal unary and binary
# ops at runtime for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.

OUTPUT_DIR=$1
CC=$2
SRC_DIR=$3
SRC_FILE=$4
CFLAGS=$5
SRC_NAME=$(basename -- "${SRC_FILE}")
INPUT_FILE=${SRC_DIR}/mlx/backend/metal/kernels/${SRC_FILE}.h
OUTPUT_FILE=${OUTPUT_DIR}/${SRC_NAME}.cpp

mkdir -p "$OUTPUT_DIR"

CONTENT=$($CC -I "$SRC_DIR" -DMLX_METAL_JIT -E -P "$INPUT_FILE" $CFLAGS 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
$CONTENT
)preamble";
}

} // namespace mlx::core::metal
EOF
