#!/bin/bash
#
# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2023-2026 Apple Inc.


# Optional arguments must come last: CMake writes empty arguments into the
# generated build files unquoted, and the shell then drops them, silently
# shifting everything after into the wrong variable. Keeping the optional ones
# trailing means a dropped empty just leaves them unset.
OUTPUT_FILE=$1
GCC=$2
SRCDIR=$3
PREAMBLE_MODE=$4
ARCH=$5
FUNCTION_NAME=${6:-get_prebuilt_preamble}
SIMD_FLAGS=$7  # Optional, e.g. "-mavx2 -mbmi2 -mfma -mf16c"
EXTRA_INCLUDE=$8  # Optional, e.g. Highway headers for JIT SIMD preambles.

case "$FUNCTION_NAME" in
  [A-Za-z_]*) ;;
  *)
    echo "Bad preamble function name '$FUNCTION_NAME' -- arguments shifted" >&2
    exit 1
    ;;
esac

if [ "$PREAMBLE_MODE" = "DARWIN" ]; then
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
elif [ "$PREAMBLE_MODE" = "KEEP_SYSTEM_INCLUDES" ]; then
  # Reparse system headers at JIT time; expanded libstdc++ is not valid Clang input.
  CC_FLAGS="-std=c++17 -fkeep-system-includes"
else
CC_FLAGS="-std=c++17"
fi

EXTRA_INCLUDE_FLAGS=()
if [ -n "$EXTRA_INCLUDE" ]; then
  EXTRA_INCLUDE_FLAGS=(-I "$EXTRA_INCLUDE")
fi

CONTENT=$(
  "$GCC" $CC_FLAGS $SIMD_FLAGS -I "$SRCDIR" "${EXTRA_INCLUDE_FLAGS[@]}" \
    -E -P "$SRCDIR/mlx/backend/cpu/compiled_preamble.h"
) || {
  echo "Failed to preprocess JIT preamble (flags: $SIMD_FLAGS)" >&2
  exit 1
}
if [ -z "$CONTENT" ]; then
  echo "Preprocessed JIT preamble is empty (flags: $SIMD_FLAGS)" >&2
  exit 1
fi

cat << EOF > "$OUTPUT_FILE"
const char* $FUNCTION_NAME() {
return R"preamble(
$INCLUDES
$CONTENT
)preamble";
}
EOF
