# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2024 Apple Inc.

$OUTPUT_FILE = $args[0]
$CL = $args[1]
$SRCDIR = $args[2]

# Get command result as array.
$CONTENT = & $CL /std:c++17 /EP "/I$SRCDIR" /Tp "$SRCDIR/mlx/backend/common/compiled_preamble.h"
# Remove empty lines.
# Otherwise there will be too much empty lines making the result unreadable.
$CONTENT = $CONTENT | Where-Object { $_.Trim() -ne '' }
# Concatenate to string.
$CONTENT = $CONTENT -join '`n'

# Append extra content.
$CONTENT = @"
$($CONTENT)
using namespace mlx::core;
using namespace mlx::core::detail;
"@

# Convert each char to ASCII code.
# Unlike the unix script that outputs string literal directly, the output from
# MSVC is way too large to be embedded as string and compilation will fail, so
# we store it as static array instead.
$CHARCODES = ([System.Text.Encoding]::ASCII.GetBytes($CONTENT) -join ', ') + ', 0'

$OUTPUT = @"
const char* get_kernel_preamble() {
  static char preamble[] = { $CHARCODES };
  return preamble;
}
"@

Set-Content -Path $OUTPUT_FILE -Value $OUTPUT
