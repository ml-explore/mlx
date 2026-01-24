#!/bin/bash
set -ex

# [Setup] Install dependencies inside the container.
dnf update -y
dnf install -y \
  blas-devel \
  lapack-devel \
  openblas-devel \
  make \
  cmake \
  clang \
  git
dnf clean all

# [C++] CI Build Sanity Check: Verifies code compilation, not for release.
export CMAKE_ARGS="-DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
export DEBUG=1
export CMAKE_C_COMPILER=/usr/bin/clang
export CMAKE_CXX_COMPILER=/usr/bin/clang++

mkdir -p build
pushd build
cmake .. -DMLX_BUILD_METAL=OFF -DCMAKE_BUILD_TYPE=DEBUG
make -j $(nproc)
./tests/tests
popd
