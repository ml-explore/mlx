#!/bin/bash

auditwheel repair dist/* \
  --plat manylinux_2_35_${1} \
  --exclude libcublas* \
  --exclude libnvrtc* \
  --exclude libcuda* \
  --exclude libcudnn* \
  --exclude libnccl* \
  -w wheel_tmp


mkdir wheelhouse
cd wheel_tmp
repaired_wheel=$(find . -name "*.whl" -print -quit)
unzip -q "${repaired_wheel}"
rm "${repaired_wheel}"
mlx_so="mlx/lib/libmlx.so"
rpath=$(patchelf --print-rpath "${mlx_so}")
base="\$ORIGIN/../../nvidia"
rpath=$rpath:${base}/cublas/lib:${base}/cuda_nvrtc/lib:${base}/cudnn/lib:${base}/nccl/lib
patchelf --force-rpath --set-rpath "$rpath" "$mlx_so"
python ../python/scripts/repair_record.py ${mlx_so}

# Re-zip the repaired wheel
zip -r -q "../wheelhouse/${repaired_wheel}" .
