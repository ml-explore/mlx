import argparse

import matplotlib
import mlx.core as mx
import numpy as np
from time_utils import measure_runtime

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def had(x):
    y = mx.hadamard_transform(x)
    mx.eval(y)


def copy(x):
    y = x + 1.0
    mx.eval(y)


def run(dtype):
    system_size = 2**26
    outputs = {}
    for test_fn in (had, copy):
        for m in [1, 12, 20, 28]:
            if test_fn == copy:
                key = "copy"
            elif m == 1:
                key = "had_2^k"
            else:
                key = "had_m*2^k"
            outputs.setdefault(key, {})
            for k in range(7, 14):
                n = m * 2**k
                if n > 2**15:
                    continue
                x_np = np.random.normal(size=(system_size // n, n)).astype(dtype)
                x = mx.array(x_np)
                runtime_ms = measure_runtime(test_fn, x=x)
                bytes_per_gb = 1e9
                ms_per_s = 1e3
                bytes_per_had = np.dtype(x_np.dtype).itemsize * 2
                bandwidth_gb = (
                    system_size * bytes_per_had / runtime_ms * ms_per_s / bytes_per_gb
                )
                print(n, bandwidth_gb)
                outputs[key][n] = bandwidth_gb

    colors = {
        "copy": "black",
        "had_2^k": "steelblue",
        "had_m*2^k": "skyblue",
    }
    for key, output in outputs.items():
        plt.scatter(output.keys(), output.values(), color=colors[key], label=key)
    plt.title(f"MLX Hadamard Benchmark -- {dtype.__name__}")
    plt.xlabel("N")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.savefig(f"bench_{dtype.__name__}.png")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    dtype = np.float16 if args.fp16 else np.float32
    run(dtype)
