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


def run_bench(system_size, fft_sizes):
    def fft(x):
        out = mx.fft.fft(x)
        mx.eval(out)
        return out

    bandwidths = []
    for n in fft_sizes:
        x = mx.random.uniform(shape=(system_size // n, n)).astype(mx.float32)
        x = x.astype(mx.complex64)
        mx.eval(x)
        runtime_ms = measure_runtime(fft, x=x)
        bandwidth = bandwidth_gb(runtime_ms, system_size // n * n)
        print("bandwidth", n, bandwidth)
        bandwidths.append(bandwidth)

    return bandwidths


def run_bench_mps(system_size, fft_sizes):
    def fft(x):
        out = torch.fft.fft(x)
        torch.mps.synchronize()
        return out

    bandwidths = []
    for n in fft_sizes:
        x_np = np.random.uniform(size=(system_size // n, n)).astype(np.complex64)
        x = torch.tensor(x_np, device="mps")
        torch.mps.synchronize()

        runtime_ms = measure_runtime(fft, x=x)
        bandwidth = bandwidth_gb(runtime_ms, system_size // n * n)
        print("bandwidth", n, bandwidth)
        bandwidths.append(bandwidth)

    return bandwidths


def time_fft():
    x = range(4, 1024)
    system_size = int(2**26)

    # with mx.stream(mx.gpu):
    #     gpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    # np.save("gpu_bandwidths", gpu_bandwidths)

    # mps_bandwidths = run_bench_mps(system_size=system_size, fft_sizes=x)

    # np.save("mps_bandwidths", mps_bandwidths)

    # system_size = int(2**21)
    # with mx.stream(mx.cpu):
    #     cpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    # np.save("cpu_bandwidths", cpu_bandwidths)

    gpu_bandwidths = np.load("gpu_bandwidths.npy")
    cpu_bandwidths = np.load("cpu_bandwidths.npy")

    # with mx.stream(mx.cpu):
    #     cpu_bandwidths = run_bench(system_size=int(2**22))

    # x = list(range(4, 1025))

    x = np.array(x)

    # multiples = (
    #     np.array([i for i in x if any(i % p == 0 for p in (2, 3, 5, 7, 11, 13))]) - 4
    # )
    # non_multiples = (
    #     np.array([i for i in x if all(i % p != 0 for p in (2, 3, 5, 7, 11, 13))]) - 4
    # )

    # plot bandwidths
    plt.scatter(x, gpu_bandwidths, color="green", label="GPU")
    # plt.scatter(x, mps_bandwidths, color="blue", label="MPS")
    plt.scatter(x, cpu_bandwidths, color="red", label="CPU")
    plt.title("MLX FFT Benchmark")
    plt.xlabel("N")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.savefig("fftm_plot.png")
    plt.clf()

    # diffs = gpu_bandwidths - mps_bandwidths

    # pos = np.where(diffs > 0)
    # neg = np.where(diffs < 0)
    # plt.scatter(x[pos], diffs[pos], color="green")
    # plt.scatter(x[neg], diffs[neg], color="red")
    # # plt.scatter(x, cpu_bandwidths, color="red", label="CPU")
    # plt.title("MLX FFT Benchmark")
    # plt.xlabel("N")
    # plt.ylabel("Bandwidth diff")
    # plt.legend()
    # plt.savefig("diff_plot.png")


if __name__ == "__main__":
    time_fft()
