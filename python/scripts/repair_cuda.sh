#!/bin/bash

auditwheel repair dist/* \
  --plat manylinux_2_35_x86_64 \
  --exclude libcublas* \
  --exclude libnvrtc* \
  -w wheel_tmp


mkdir wheelhouse
cd wheel_tmp
repaired_wheel=$(find . -name "*.whl" -print -quit)
unzip -q "${repaired_wheel}"
rm "${repaired_wheel}"
core_so=$(find mlx -name "core*.so" -print -quit)
rpath=$(patchelf --print-rpath "${core_so}")
rpath=$rpath:\$ORIGIN/../nvidia/cublas/lib:\$ORIGIN/../nvidia/cuda_nvrtc/lib
patchelf --force-rpath --set-rpath "$rpath" "$core_so"

# Re-zip the repaired wheel
zip -r -q "../wheelhouse/${repaired_wheel}" .
