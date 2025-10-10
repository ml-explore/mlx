import math
import os
import subprocess
import time
from functools import partial

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

RESULTS_DIR = "./results"


if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

DEVICE_NAME = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
DEVICE_NAME = DEVICE_NAME.decode("utf-8").strip("\n")

TORCH_DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


N_WARMUP = 5
N_ITER_BENCH = 50
N_ITER_FUNC = 20

VECTOR_LENGTHS = [4096 * (2**i) for i in range(5)]
MASK_DENSITIES = [0.1, 0.25, 0.5, 0.75]
D_TYPES = ("float32", "float16")


def _power_of_two_formatter(value, _position):
    if value <= 0:
        return ""
    exponent = int(round(math.log2(value)))
    if abs(value - (1 << exponent)) / value > 1e-6:
        return f"{value:g}"
    return f"$2^{{{exponent}}}$"


def torch_sync():
    if TORCH_DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif TORCH_DEVICE.type == "mps":
        torch.mps.synchronize()


def masked_scatter_mlx(self_arr, mask_arr, src_arr):
    outs = [mx.masked_scatter(self_arr, mask_arr, src_arr) for _ in range(N_ITER_FUNC)]
    mx.eval(outs)
    return outs


@torch.no_grad()
def masked_scatter_torch(self_tensor, mask_tensor, src_tensor):
    outs = []
    for _ in range(N_ITER_FUNC):
        out = self_tensor.clone()
        out.masked_scatter_(mask_tensor, src_tensor)
        outs.append(out)
    torch_sync()
    return outs


def measure(fn):
    for _ in range(N_WARMUP):
        fn()
    start = time.perf_counter_ns()
    for _ in range(N_ITER_BENCH):
        fn()
    end = time.perf_counter_ns()
    return (end - start) * 1e-9


def bytes_touched(length, true_count, item_size):
    mask_bytes = length
    self_bytes = length * item_size * 2  # read + write
    src_bytes = true_count * item_size
    return (mask_bytes + self_bytes + src_bytes) * N_ITER_FUNC * N_ITER_BENCH


def build_case(length, density, np_dtype, torch_dtype):
    true_count = max(1, int(round(length * density)))

    rng = np.random.default_rng()
    self_np = rng.normal(0.0, 1.0, length).astype(np_dtype)
    mask_np = np.zeros(length, dtype=bool)
    mask_np[:true_count] = True
    rng.shuffle(mask_np)
    src_np = rng.normal(0.0, 1.0, true_count).astype(np_dtype)

    self_mlx = mx.array(self_np)
    mask_mlx = mx.array(mask_np)
    src_mlx = mx.array(src_np)

    self_torch = torch.from_numpy(self_np).to(device=TORCH_DEVICE, dtype=torch_dtype)
    mask_torch = torch.from_numpy(mask_np).to(device=TORCH_DEVICE)
    src_torch = torch.from_numpy(src_np).to(device=TORCH_DEVICE, dtype=torch_dtype)

    # Correctness check once per configuration
    mx_out = mx.masked_scatter(self_mlx, mask_mlx, src_mlx)
    mx.eval(mx_out)
    torch_out = self_torch.clone()
    torch_out.masked_scatter_(mask_torch, src_torch)

    atol = 5e-3 if np_dtype == np.float16 else 1e-5
    if not np.allclose(np.array(mx_out), torch_out.cpu().numpy(), atol=atol):
        raise AssertionError("masked_scatter results diverged between MLX and Torch")

    return (self_mlx, mask_mlx, src_mlx, self_torch, mask_torch, src_torch, true_count)


def bench_case(length, density, dtype):
    np_dtype = getattr(np, dtype)
    torch_dtype = getattr(torch, dtype)
    (
        self_mlx,
        mask_mlx,
        src_mlx,
        self_torch,
        mask_torch,
        src_torch,
        true_count,
    ) = build_case(length, density, np_dtype, torch_dtype)

    time_mlx = measure(partial(masked_scatter_mlx, self_mlx, mask_mlx, src_mlx))
    time_torch = measure(
        partial(masked_scatter_torch, self_torch, mask_torch, src_torch)
    )

    total_bytes = bytes_touched(length, true_count, np_dtype().itemsize)
    bytes_per_gb = float(1024**3)
    mlx_gbps = (total_bytes / bytes_per_gb) / time_mlx
    torch_gbps = (total_bytes / bytes_per_gb) / time_torch

    return time_mlx, time_torch, mlx_gbps, torch_gbps


def plot_density(ax_perf, ax_speedup, density, dtype):
    mlx_gbps = []
    torch_gbps = []
    mlx_times = []
    torch_times = []

    for length in VECTOR_LENGTHS:
        t_mlx, t_torch, gbps_mlx, gbps_torch = bench_case(length, density, dtype)
        mlx_gbps.append(gbps_mlx)
        torch_gbps.append(gbps_torch)
        mlx_times.append(t_mlx)
        torch_times.append(t_torch)

    ax_perf.plot(VECTOR_LENGTHS, mlx_gbps, "tab:blue", label="MLX")
    ax_perf.plot(VECTOR_LENGTHS, torch_gbps, "tab:red", label="Torch")
    ax_perf.set_xscale("log", base=2)
    ax_perf.set_xticks(VECTOR_LENGTHS)
    formatter = FuncFormatter(_power_of_two_formatter)
    ax_perf.xaxis.set_major_formatter(formatter)
    ax_perf.set_title(f"density={density:.2f}")
    ax_perf.set_ylabel("GB/s")
    ax_perf.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_perf.legend()

    speedup = np.array(torch_times) / np.array(mlx_times)
    ax_speedup.plot(VECTOR_LENGTHS, speedup, "tab:green")
    ax_speedup.axhline(1.0, color="tab:gray", linestyle="--")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xticks(VECTOR_LENGTHS)
    ax_speedup.xaxis.set_major_formatter(formatter)
    ax_speedup.set_ylabel("Speedup (Torch_t / MLX_t)")
    ax_speedup.grid(True, which="both", linestyle=":", alpha=0.4)


def main():
    for dtype in D_TYPES:
        fig, axs = plt.subplots(
            len(MASK_DENSITIES),
            2,
            figsize=(10, 12),
            layout="constrained",
            sharex=True,
        )

        for i, density in enumerate(MASK_DENSITIES):
            plot_density(axs[i][0], axs[i][1], density, dtype)
            axs[i][0].set_xlabel("vector length")
            axs[i][1].set_xlabel("vector length")

        fig.suptitle(
            f"{DEVICE_NAME.replace('Apple ', '')} ({TORCH_DEVICE.type}) | dtype={dtype}"
        )
        output_path = os.path.join(
            RESULTS_DIR,
            f"{DEVICE_NAME.replace(' ', '_')}_masked_scatter_{dtype}.pdf",
        )
        fig.savefig(output_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
