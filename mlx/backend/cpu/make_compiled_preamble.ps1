# This script generates a C++ function that provides the CPU
# code for use with kernel generation.
#
# Copyright © 2024-2026 Apple Inc.

$OUTPUT_FILE = $args[0]
$CL = $args[1]
$SRCDIR = $args[2]
# args[3] = preamble mode (unused on Windows; MSVC and clang-cl use the same flags)
# args[4] = CMAKE_SYSTEM_PROCESSOR (unused on Windows)
# Optional arguments come last: CMake writes empty arguments into the generated
# build files unquoted and the shell drops them, so anything after an empty
# argument lands in the wrong slot. Trailing optionals just go missing instead.
$FUNCTION_NAME = if ($args.Count -gt 5 -and $args[5]) { $args[5] } else { "get_prebuilt_preamble" }
$SIMD_FLAGS = if ($args.Count -gt 6) { $args[6] } else { "" }  # e.g. "/arch:AVX2"
$EXTRA_INCLUDE = if ($args.Count -gt 7) { $args[7] } else { "" }  # e.g. Highway headers

if ($FUNCTION_NAME -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
  throw "Bad preamble function name '$FUNCTION_NAME' -- arguments shifted"
}

# Detect compiler type: MSVC/clang-cl use /EP, GCC/clang++ use -E -P
$CL_NAME = [System.IO.Path]::GetFileNameWithoutExtension($CL)
$IS_MSVC_LIKE = ($CL_NAME -eq 'cl') -or ($CL_NAME -eq 'clang-cl')

# Build the full argument list to avoid PowerShell splatting issues
# (splatting a single-element array can degrade to char-by-char expansion).
$CL_ARGS = [System.Collections.ArrayList]::new()
if ($IS_MSVC_LIKE) {
  [void]$CL_ARGS.Add('/std:c++17')
  [void]$CL_ARGS.Add('/EP')
} else {
  [void]$CL_ARGS.Add('-std=c++17')
  [void]$CL_ARGS.Add('-E')
  [void]$CL_ARGS.Add('-P')
}
if ($SIMD_FLAGS) {
  foreach ($f in ($SIMD_FLAGS -split ' ')) { [void]$CL_ARGS.Add($f) }
}
if ($IS_MSVC_LIKE) {
  [void]$CL_ARGS.Add("/I$SRCDIR")
  if ($EXTRA_INCLUDE) {
    [void]$CL_ARGS.Add("/I$EXTRA_INCLUDE")
  }
  [void]$CL_ARGS.Add('/Tp')
} else {
  [void]$CL_ARGS.Add('-I')
  [void]$CL_ARGS.Add("$SRCDIR")
  if ($EXTRA_INCLUDE) {
    [void]$CL_ARGS.Add('-I')
    [void]$CL_ARGS.Add("$EXTRA_INCLUDE")
  }
}
[void]$CL_ARGS.Add("$SRCDIR/mlx/backend/cpu/compiled_preamble.h")

# Get command result as array. Redirect stderr to null to suppress warnings.
$CONTENT = & $CL @CL_ARGS 2>$null
if ($LASTEXITCODE -ne 0) {
  throw "Failed to preprocess JIT preamble with $CL (exit code $LASTEXITCODE)"
}
# Remove empty lines.
# Otherwise there will be too much empty lines making the result unreadable.
$CONTENT = $CONTENT | Where-Object { $_.Trim() -ne '' }
if (-not $CONTENT) {
  throw "Failed to preprocess JIT preamble with ${CL}: no output"
}
# Concatenate to string.
$CONTENT = $CONTENT -join "`n"

# Convert each char to ASCII code.
# Unlike the unix script that outputs string literal directly, the output from
# MSVC is way too large to be embedded as string and compilation will fail, so
# we store it as static array instead.
$CHARCODES = ([System.Text.Encoding]::ASCII.GetBytes($CONTENT) -join ', ') + ', 0'

$OUTPUT = @"
const char* $FUNCTION_NAME() {
  static char preamble[] = { $CHARCODES };
  return preamble;
}
"@

Set-Content -Path $OUTPUT_FILE -Value $OUTPUT
