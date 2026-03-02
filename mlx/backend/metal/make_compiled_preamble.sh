#!/bin/bash
#
# This script generates a C++ function that provides the Metal source code
# at runtime for use with kernel generation.
#
# The steps executed are as follows 
# - Take as input a metal header file in the mlx metal backend 
# - Use the metal compiler to expand the dependency headers 
# - Sort the headers in order of inclusion 
# - Expand the headers in order of inclusion 
# - Export the generated source code content as a C++ function
#
# Doing the expansion this way allows us to retain macros, comments, and 
# formatting in the expanded source. This adds user readibility, and also 
# enables use of the metal macros in the source code which can then be 
# handled by the metal runtime compiler
#
# Copyright © 2023-25 Apple Inc.

OUTPUT_DIR=$1
CC=$2
SRC_DIR=$3
SRC_FILE=$4
CFLAGS=$5
SRC_NAME=$(basename -- "${SRC_FILE}")
JIT_INCLUDES=${SRC_DIR}/mlx/backend/metal/kernels/jit
INPUT_FILE=${SRC_DIR}/mlx/backend/metal/kernels/${SRC_FILE}.h
OUTPUT_FILE=${OUTPUT_DIR}/${SRC_NAME}.cpp

# Prepare output
mkdir -p "$OUTPUT_DIR"

# Use the metal compiler to get a list of headers (with depth)
CCC="xcrun -sdk macosx metal -x metal"
HDRS=$( $CCC -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT -E -P -CC -C -H "$INPUT_FILE" $CFLAGS -w 2>&1 1>/dev/null )

# Remove any included system frameworks (for MetalPerformancePrimitive headers)
HDRS=$(echo "$HDRS" | grep -v "Xcode")

# Use the header depth to sort the files in order of inclusion
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

# Make sure the given metal header is also expanded in the source content
HDRS_SORTED+=("${INPUT_FILE#$SRC_DIR/}")

# Expand the headers in order of inclusion 
CONTENT=$(
echo "// Copyright © 2025 Apple Inc."
echo "" 
echo "// Auto generated source for ${INPUT_FILE#$SRC_DIR/}"
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

# Export the generated source code content as a C++ function
cat << EOF > "$OUTPUT_FILE"
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
$CONTENT
)preamble";
}

} // namespace mlx::core::metal
EOF
