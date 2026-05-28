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
HIGHWAY_INCLUDE_DIR=$7  # Optional, used when MLX_ENABLE_AVX2 uses Highway

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

HIGHWAY_INCLUDE_FLAG=()
if [ -n "$HIGHWAY_INCLUDE_DIR" ]; then
  HIGHWAY_INCLUDE_FLAG=(
    -I "$HIGHWAY_INCLUDE_DIR"
    -DMLX_USE_HIGHWAY
    -DHWY_DISABLED_TARGETS=HWY_AVX2-1
    -DHWY_DISABLE_PCLMUL_AES
    -DHWY_COMPILE_ONLY_STATIC
  )
fi

CONTENT=$(
  "$GCC" $CC_FLAGS $SIMD_FLAGS -I "$SRCDIR" "${HIGHWAY_INCLUDE_FLAG[@]}" \
    -E -P "$SRCDIR/mlx/backend/cpu/compiled_preamble.h" 2>/dev/null
)

cat << EOF > "$OUTPUT_FILE"
const char* get_prebuilt_preamble() {
return R"preamble(
$INCLUDES
$CONTENT
)preamble";
}
EOF
