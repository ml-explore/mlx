# Copyright © 2024 Apple Inc.

import matplotlib
import mlx.core as mx
import numpy as np
import torch
from time_utils import measure_runtime
from tqdm import tqdm

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


def run_bench_mps(system_size):
    def fft(x):
        out = torch.fft.fft(x)
        torch.mps.synchronize()
        return out

    bandwidths = []
    for n in tqdm(range(1, 512)):
        # for k in range(4, 12):
        # n = 2**k
        x_np = np.random.uniform(size=(system_size // n, n)).astype(np.complex64)
        x = torch.tensor(x_np, device="mps")

        runtime_ms = measure_runtime(fft, x=x)
        bandwidths.append(bandwidth_gb(runtime_ms, system_size // n * n))

    return bandwidths


def time_fft():
    mps_bandwidths = run_bench_mps(system_size=int(2**24))
    # print('mps_bandwidths', mps_bandwidths)

    # with mx.stream(mx.cpu):
    #     cpu_bandwidths = run_bench(system_size=int(2**22))

    # with mx.stream(mx.gpu):
    #     gpu_bandwidths = run_bench(system_size=int(2**24))
    #     print('gpu_bandwidths', gpu_bandwidths)

    # plot bandwidths
    # x = [2**k for k in range(4, 12)]
    x = list(range(1, 512))
    # plt.scatter(x, gpu_bandwidths, color="green", label="GPU")
    np.save("mps_bandwidths", mps_bandwidths)
    plt.scatter(x, mps_bandwidths, color="blue", label="MPS")
    # plt.scatter(x, cpu_bandwidths, color="red", label="CPU")
    plt.title("MLX FFT Benchmark")
    plt.xlabel("N")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.savefig("fft_plot.png")


if __name__ == "__main__":
    time_fft()
