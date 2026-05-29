#!/bin/bash
#
# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2023-2026 Apple Inc.


OUTPUT_FILE=$1
GCC=$2
SRCDIR=$3
CLANG=$4
ARCH=$5
SIMD_FLAGS=$6  # Optional, e.g. "-mavx2 -mbmi2 -mfma -mf16c"
EXTRA_INCLUDE=$7  # Optional, e.g. Highway headers for JIT SIMD preambles.
FUNCTION_NAME=${8:-get_prebuilt_preamble}

if [ "$CLANG" = "TRUE" ]; then
  read -r -d '' INCLUDES <<- EOM
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#include <arm_fp16.h>
#endif
EOM
CC_FLAGS="-arch ${ARCH} -nobuiltininc -nostdinc"
else
CC_FLAGS="-std=c++17"
fi

EXTRA_INCLUDE_FLAGS=()
if [ -n "$EXTRA_INCLUDE" ]; then
  EXTRA_INCLUDE_FLAGS=(-I "$EXTRA_INCLUDE")
fi

CONTENT=$(
  "$GCC" $CC_FLAGS $SIMD_FLAGS -I "$SRCDIR" "${EXTRA_INCLUDE_FLAGS[@]}" \
    -E -P "$SRCDIR/mlx/backend/cpu/compiled_preamble.h" 2>/dev/null
)

cat << EOF > "$OUTPUT_FILE"
const char* $FUNCTION_NAME() {
return R"preamble(
$INCLUDES
$CONTENT
)preamble";
}
EOF
