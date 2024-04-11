# Copyright © 2024 Apple Inc.

import matplotlib
import mlx.core as mx
import numpy as np
import sympy
import torch
from time_utils import measure_runtime

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
    system_size = int(2**24)

    with mx.stream(mx.gpu):
        gpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    np.save("gpu_bandwidths", gpu_bandwidths)

    mps_bandwidths = run_bench_mps(system_size=system_size, fft_sizes=x)

    np.save("mps_bandwidths", mps_bandwidths)

    system_size = int(2**21)
    with mx.stream(mx.cpu):
        cpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    np.save("cpu_bandwidths", cpu_bandwidths)

    # cpu_bandwidths = np.load("cpu_bandwidths.npy")
    # gpu_bandwidths = np.load("gpu_bandwidths.npy")
    # mps_bandwidths = np.load("mps_bandwidths.npy")

    x = np.array(x)

    all_indices = x - x[0]
    radix_2to13 = (
        np.array([i for i in x if all(p <= 13 for p in sympy.primefactors(i))]) - x[0]
    )
    bluesteins = (
        np.array([i for i in x if any(p > 13 for p in sympy.primefactors(i))]) - x[0]
    )

    for indices, name in [
        (all_indices, "All"),
        (radix_2to13, "Radix 2-13"),
        (bluesteins, "Bluestein's"),
    ]:
        # plot bandwidths
        plt.scatter(x[indices], gpu_bandwidths[indices], color="green", label="GPU")
        plt.scatter(x[indices], mps_bandwidths[indices], color="blue", label="MPS")
        plt.scatter(x[indices], cpu_bandwidths[indices], color="red", label="CPU")
        plt.title(f"MLX FFT Benchmark -- {name}")
        plt.xlabel("N")
        plt.ylabel("Bandwidth (GB/s)")
        plt.legend()
        plt.savefig(f"{name}.png")
        plt.clf()

    av_gpu_bandwidth = np.mean(gpu_bandwidths)
    av_mps_bandwidth = np.mean(mps_bandwidths)
    av_cpu_bandwidth = np.mean(cpu_bandwidths)
    print("Average bandwidths:")
    print("GPU:", av_gpu_bandwidth)
    print("MPS:", av_mps_bandwidth)
    print("CPU:", av_cpu_bandwidth)

    portion_faster = len(np.where(gpu_bandwidths > mps_bandwidths)[0]) / len(x)
    print("Percent MLX faster than MPS: ", portion_faster * 100)


if __name__ == "__main__":
    time_fft()
