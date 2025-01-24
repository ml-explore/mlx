#!/bin/bash

tmpfile=$(mktemp)
cat <<HOSTFILE >$tmpfile
[
    ["127.0.0.1:5000"],
    ["127.0.0.1:5001"],
    ["127.0.0.1:5002"],
    ["127.0.0.1:5003"],
    ["127.0.0.1:5004"],
    ["127.0.0.1:5005"],
    ["127.0.0.1:5006"],
    ["127.0.0.1:5007"]
]
HOSTFILE

ring_test="$(dirname ${BASH_SOURCE[0]})/ring_test_distributed.py"

for i in {0..7}; do
    if (($i == 7)); then
        sleep 1
    fi
    DEVICE=cpu MLX_RING_VERBOSE=1 MLX_HOSTFILE=$tmpfile MLX_RANK=$i python $ring_test &
done
wait
