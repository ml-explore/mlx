#!/bin/bash

# If nvidia-smi fails (no NVIDIA GPU or driver), default to 90a.
if ! nvidia-smi >/dev/null 2>&1; then
    echo "90a"
    exit 0
fi

# Otherwise, query the native architecture.
arch=`__nvcc_device_query`
case "$arch" in
    "90")
        echo "90a" ;;
    "100")
        echo "100a" ;;
    "121")
        echo "121a" ;;
    *)
        echo "native" ;;
esac
