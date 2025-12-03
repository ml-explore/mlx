#!/bin/bash
set -ex

export CMAKE_C_COMPILER=/usr/bin/clang
export CMAKE_CXX_COMPILER=/usr/bin/clang++
BASE_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
if [[ "$(uname -s)" != "Darwin" ]]; then
  BASE_CMAKE_ARGS+=" -DMLX_BUILD_METAL=OFF"
fi

run_test() {
  local sanitizer_name=$1
  local cmake_sanitizer_flag="-DUSE_${sanitizer_name}=ON"
  echo "  Running tests with: ${sanitizer_name}"

  case "$sanitizer_name" in
    ASAN)
      export ASAN_OPTIONS="detect_leaks=0"
      ;;
    UBSAN)
      export UBSAN_OPTIONS="halt_on_error=0:print_stacktrace=1"
      ;;
    TSAN)
      export TSAN_OPTIONS=""
      ;;
  esac

  rm -rf build
  mkdir -p build
  pushd build > /dev/null

  cmake .. ${BASE_CMAKE_ARGS} ${cmake_sanitizer_flag}
  make -j $(nproc)
  ./tests/tests

  popd > /dev/null
  unset ${sanitizer_name}_OPTIONS
}

sanitizer_arg=$(echo "$1" | tr '[:lower:]' '[:upper:]')

if [[ "$sanitizer_arg" == "ASAN" || "$sanitizer_arg" == "UBSAN" || "$sanitizer_arg" == "TSAN" ]]; then
  run_test "$sanitizer_arg"
  echo "  ${sanitizer_arg} test run completed successfully."
else
  echo "Error: Invalid sanitizer '$1'. Please use one of: ASAN, UBSAN, TSAN."
  exit 1
fi
