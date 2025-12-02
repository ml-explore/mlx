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
JIT_INCLUDES=${SRC_DIR}/mlx/backend/metal/kernels/jit
INPUT_FILE=${SRC_DIR}/mlx/backend/metal/kernels/${SRC_FILE}.h
OUTPUT_FILE=${OUTPUT_DIR}/${SRC_NAME}.cpp

mkdir -p "$OUTPUT_DIR"
# CONTENT=$($CC -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT -E -P "$INPUT_FILE" $CFLAGS 2>/dev/null)

CCC="xcrun -sdk macosx metal -x metal"

HDRS=$( $CCC -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT -E -P -CC -C -H "$INPUT_FILE" $CFLAGS -w 2>&1 1>/dev/null )

declare -a HDRS_LIST=($HDRS)
declare -a HDRS_STACK=()
declare -a HDRS_SORTED=()

length=${#HDRS_LIST[@]}

HDRS_LIST+=(".")

for ((i=0; i<${length}; i+=2));
do 

  header="${HDRS_LIST[$i+1]#$SRC_DIR/}"

  str_this="${HDRS_LIST[$i]}"
  str_next="${HDRS_LIST[$i + 2]}"

  depth_this=${#str_this}
  depth_next=${#str_next}

  # If we have a dependency then we stack it
  if [ $depth_next -gt $depth_this ]; then 
    HDRS_STACK=($header ${HDRS_STACK[@]})

  # If we are done with this level 
  else 
    # We add the header to out list
    HDRS_SORTED+=($header) 

    # Pop the stacked up dependencies
    pop_len=$((depth_this - depth_next))
    for popped_header in "${HDRS_STACK[@]:0:$pop_len}"
    do 
      HDRS_SORTED+=($popped_header)
    done 

    HDRS_STACK=(${HDRS_STACK[@]:$pop_len})
  fi  

done

HDRS_SORTED+=("${INPUT_FILE#$SRC_DIR/}")

CONTENT=$(
echo "// Copyright © 2025 Apple Inc."
echo "" 
echo "// Auto generated source for $INPUT_FILE" 
echo "" 

for header in "${HDRS_SORTED[@]}"
do 
  echo "///////////////////////////////////////////////////////////////////////////////"
  echo "// Contents from \"${header}\""
  echo "///////////////////////////////////////////////////////////////////////////////"
  echo ""

  echo "#line 1 \"${header}\""

  grep -h -v -G -e "#include \".*.h\"" -e "#pragma once" "${SRC_DIR}/${header}" 
  
  echo ""
  
done

echo "///////////////////////////////////////////////////////////////////////////////"
)

cat << EOF > "$OUTPUT_FILE"
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
$CONTENT
)preamble";
}

} // namespace mlx::core::metal
EOF
