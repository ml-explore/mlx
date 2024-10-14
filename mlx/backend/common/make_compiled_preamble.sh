#!/bin/bash
#
# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2023-24 Apple Inc.


OUTPUT_FILE=$1
GCC=$2
SRCDIR=$3
CLANG=$4

if [ "$CLANG" = "TRUE" ]; then
  read -r -d '' INCLUDES <<- EOM
  #include <cmath>
  #include <complex>
  #include <cstdint>
  #include <vector>
EOM
CC_FLAGS=""
else
CC_FLAGS="-std=c++17"
fi

CONTENT=$($GCC $CC_FLAGS -I "$SRCDIR" -E "$SRCDIR/mlx/backend/common/compiled_preamble.h" 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
const char* get_kernel_preamble() {
return R"preamble(
$INCLUDES
$CONTENT
using namespace mlx::core;
using namespace mlx::core::detail;
)preamble";
}
EOF
