#!/bin/bash

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
