#!/bin/bash
#
# This script generates a C++ function that provides the Metal unary and binary
# ops at runtime for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.


OUTPUT_FILE=$1
CC=$2
SRCDIR=$3

CONTENT=$($CC -I $SRCDIR -E $SRCDIR/mlx/backend/metal/kernels/compiled_preamble.h 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
// Copyright © 2023-24 Apple Inc.

namespace mlx::core::metal {

const char* get_kernel_preamble() {
  return R"preamble(
$CONTENT
)preamble";

}

} // namespace mlx::core::metal
EOF
