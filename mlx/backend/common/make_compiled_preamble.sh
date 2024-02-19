#!/bin/bash
#
# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.


OUTPUT_FILE=$1
CC=$2
SRCDIR=$3

CONTENT=$($CC -I $SRCDIR -E $SRCDIR/mlx/backend/common/compiled_preamble.h 2>/dev/null)

echo $OUTPUT_FILE
cat << EOF > "$OUTPUT_FILE"
const char* get_kernel_preamble() {
return R"(
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
$CONTENT
using namespace mlx::core::detail;
)";
}
EOF
