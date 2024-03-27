# Copyright © -2024 Apple Inc.

import matplotlib
import mlx.core as mx
import numpy as np
from time_utils import measure_runtime

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bandwidth_gb(runtime_ms, system_size):
    bytes_per_fft = np.dtype(np.complex64).itemsize * 2
    bytes_per_gb = 1e9
    ms_per_s = 1e3
    return system_size * bytes_per_fft / runtime_ms * ms_per_s / bytes_per_gb


def run_bench(system_size):
    def fft(x):
        out = mx.fft.fft(x)
        mx.eval(out)
        return out

    bandwidths = []
    for k in range(4, 12):
        n = 2**k
        x = mx.random.uniform(shape=(system_size // n, n)).astype(mx.float32)
        x = x.astype(mx.complex64)
        mx.eval(x)
        runtime_ms = measure_runtime(fft, x=x)
        bandwidths.append(bandwidth_gb(runtime_ms, system_size))

    return bandwidths


def time_fft():

    with mx.stream(mx.cpu):
        cpu_bandwidths = run_bench(system_size=int(2**22))

    with mx.stream(mx.gpu):
        gpu_bandwidths = run_bench(system_size=int(2**29))

    # plot bandwidths
    x = [2**k for k in range(4, 12)]
    plt.scatter(x, gpu_bandwidths, color="green", label="GPU")
    plt.scatter(x, cpu_bandwidths, color="red", label="CPU")
    plt.title("MLX FFT Benchmark")
    plt.xlabel("N")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.savefig("fft_plot.png")


if __name__ == "__main__":
    time_fft()
