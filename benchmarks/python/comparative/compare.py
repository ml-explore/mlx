# Copyright © 2023 Apple Inc.

#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from subprocess import run

BENCH_MLX = Path(__file__).parent / "bench_mlx.py"
BENCH_TORCH = Path(__file__).parent / "bench_torch.py"


def run_or_raise(*args, **kwargs):
    try:
        result = run(*args, capture_output=True, **kwargs)
        return float(result.stdout)
    except ValueError:
        raise ValueError(
            f"stdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
        )


def compare(args):
    t_mlx = run_or_raise(["python", BENCH_MLX] + args)
    t_torch = run_or_raise(["python", BENCH_TORCH] + args)

    print((t_torch - t_mlx) / t_torch, " ".join(args), sep="\t")


def compare_mlx_quant(args_base, bits_list):
    """Compare quantized matmul across bit widths (MLX only, no PyTorch)."""
    results = {}
    for bits in bits_list:
        bench_args = args_base.replace("{bits}", str(bits)).split()
        results[bits] = run_or_raise(["python", BENCH_MLX] + bench_args)
    baseline = max(results.values())
    for bits in bits_list:
        speedup = (baseline - results[bits]) / baseline if baseline > 0 else 0
        print(f"{speedup:.4f}\t{args_base.replace('{bits}', str(bits))}")


def compare_mlx_dtypes(args, dt1, dt2):
    t_mlx_dt1 = run_or_raise(["python", BENCH_MLX] + args + ["--dtype", dt1])
    t_mlx_dt2 = run_or_raise(["python", BENCH_MLX] + args + ["--dtype", dt2])

    print((t_mlx_dt2 - t_mlx_dt1) / t_mlx_dt2, " ".join(args), sep="\t")


def make_regex_search(regexes):
    compiled_regexes = list(map(re.compile, regexes))

    def search(x):
        return (c.search(x) is not None for c in compiled_regexes)

    return search


def make_predicate(positive_filter, negative_filter):
    if positive_filter is not None:
        positive_filter_search = make_regex_search(positive_filter)
        positive_filter = lambda x: all(positive_filter_search(x))
    else:
        positive_filter = lambda x: True

    if negative_filter is not None:
        negative_filter_search = make_regex_search(negative_filter)
        negative_filter = lambda x: not any(negative_filter_search(x))
    else:
        negative_filter = lambda x: True

    def predicate(x):
        return positive_filter(x) and negative_filter(x)

    return predicate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparisons against PyTorch")
    parser.add_argument(
        "--filter", "-f", help="Regex filter to select benchmarks", nargs="+"
    )
    parser.add_argument(
        "--negative_filter", "-n", help="Regex filter to remove benchmarks", nargs="+"
    )
    parser.add_argument(
        "--mlx_dtypes",
        "-d",
        help="Compare mlx benchmarks between the 2 provided data types",
        nargs=2,
    )
    args, rest = parser.parse_known_args()

    _filter = make_predicate(args.filter, args.negative_filter)

    if args.mlx_dtypes:
        compare_filtered = lambda x: (
            compare_mlx_dtypes(x.split() + rest, args.mlx_dtypes[0], args.mlx_dtypes[1])
            if _filter(x)
            else None
        )
    else:
        compare_filtered = lambda x: compare(x.split() + rest) if _filter(x) else None

    # Binary ops
    compare_filtered("add --size 10x1024x128 --size 1x1024x128 --cpu")
    compare_filtered("add --size 10x1024x128 --size 1x1024x128")
    compare_filtered("add --size 1024x128 --size 1x128 --cpu")
    compare_filtered("add --size 1024x128 --size 1x128")
    compare_filtered("add --size 1024x4096 --size 1x4096 --cpu")
    compare_filtered("add --size 1024x4096 --size 1x4096")
    compare_filtered("add --size 1024x4096 --size 1x1024 --transpose 1,0 --cpu")
    compare_filtered("add --size 1024x4096 --size 1x1024 --transpose 1,0")
    compare_filtered("add --size 1024x1024 --size 1024x1024 --cpu")
    compare_filtered("add --size 1024x1024 --size 1024x1024")
    compare_filtered("add --size 1024x1024 --size 1024x1024 --transpose 1,0 --cpu")
    compare_filtered("add --size 1024x1024 --size 1024x1024 --transpose 1,0")
    compare_filtered(
        "add --size 1024x1024 --size 1024x1024 --transpose 1,0 --transpose 1,0 --cpu"
    )
    compare_filtered(
        "add --size 1024x1024 --size 1024x1024 --transpose 1,0 --transpose 1,0"
    )

    # Reduction ops
    compare_filtered("sum_all --size 10x1024x128 --cpu")
    compare_filtered("sum_all --size 10x1024x128")
    compare_filtered("sum_axis --size 16x1024x128 --axis 2 --cpu")
    compare_filtered("sum_axis --size 16x1024x128 --axis 2")
    compare_filtered("sum_axis --size 16x128x1024 --axis 2 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 2")
    compare_filtered("sum_axis --size 1024x1024 --axis 1 --cpu")
    compare_filtered("sum_axis --size 1024x1024 --axis 1")
    compare_filtered("sum_axis --size 1024x1024 --axis 0 --cpu")
    compare_filtered("sum_axis --size 1024x1024 --axis 0")
    compare_filtered("sum_axis --size 16x128x1024 --axis 1 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 1")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,1 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,1")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,2 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,2")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,1 --transpose 0,2,1 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,1 --transpose 0,2,1")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,2 --transpose 0,2,1 --cpu")
    compare_filtered("sum_axis --size 16x128x1024 --axis 0,2 --transpose 0,2,1")
    compare_filtered("argmax --size 10x1024x128 --axis 1 --cpu")
    compare_filtered("argmax --size 10x1024x128 --axis 1")
    compare_filtered("argmax --size 10x1024x128 --axis 2 --cpu")
    compare_filtered("argmax --size 10x1024x128 --axis 2")
    compare_filtered("argmax --size 1024x1024 --axis 1 --cpu")
    compare_filtered("argmax --size 1024x1024 --axis 1")

    # Matmul ops
    compare_filtered("matmul_square --size 1024x1024")
    compare_filtered("matmul_square --size 1024x1024 --cpu")
    compare_filtered("matmul_square --size 16x1024x1024")
    compare_filtered("matmul_square --size 16x1024x1024 --cpu")
    compare_filtered(
        "matmul --size 16x768x768 --size 16x768x768 --transpose= --transpose 0,2,1"
    )
    compare_filtered(
        "matmul --size 16x768x768 --size 16x768x768 --transpose= --transpose 0,2,1 --cpu"
    )
    compare_filtered(
        "matmul --size 16x768x128 --size 16x768x128 --transpose= --transpose 0,2,1"
    )
    compare_filtered(
        "matmul --size 16x768x128 --size 16x768x128 --transpose= --transpose 0,2,1 --cpu"
    )
    compare_filtered("matmul --size 512x8192 --size 8192x512")
    compare_filtered("matmul --size 512x8192 --size 8192x512 --cpu")
    # compare_filtered("matmul --size 512x131072 --size 131072x512")
    # compare_filtered("matmul --size 512x131072 --size 131072x512 --cpu")
    compare_filtered("matmul --size 8192x512 --size 512x8192")
    compare_filtered("matmul --size 8192x512 --size 512x8192 --cpu")
    # compare_filtered("matmul --size 131072x512 --size 512x512")
    # compare_filtered("matmul --size 131072x512 --size 512x512 --cpu")
    compare_filtered("linear --size 1024x1024 --size 1024 --size 128x1024")
    compare_filtered("linear --size 1024x1024 --size 1024 --size 128x1024 --cpu")
    compare_filtered("linear --size 1024x1024 --size 1024 --size 128x1024 --fused")
    compare_filtered(
        "linear --size 1024x1024 --size 1024 --size 128x1024 --fused --cpu"
    )

    # Matvec ops
    compare_filtered("matmul --size 1x1x4096 --size 4096x4096 --cpu")
    compare_filtered("matmul --size 1x1x4096 --size 4096x4096")
    compare_filtered(
        "matmul --size 1x1x4096 --size 4096x4096 --transpose= --transpose 1,0 --cpu"
    )
    compare_filtered(
        "matmul --size 1x1x4096 --size 4096x4096 --transpose= --transpose 1,0"
    )
    compare_filtered("matmul --size 32x1x1000 --size 32x1000x128 --cpu")
    compare_filtered("matmul --size 32x1x1000 --size 32x1000x128")
    compare_filtered(
        "matmul --size 32x1x1000 --size 32x128x1000 --transpose= --transpose 0,2,1 --cpu"
    )
    compare_filtered(
        "matmul --size 32x1x1000 --size 32x128x1000 --transpose= --transpose 0,2,1"
    )

    # Various ops
    compare_filtered("softmax --size 32x16x1024 --axis 2")
    compare_filtered("softmax --size 32x16x1024 --axis 2 --cpu")
    compare_filtered("softmax --size 32x16x1024 --axis 2 --fused")
    compare_filtered("softmax --size 32x16x1024 --axis 2 --fused --cpu")
    compare_filtered("softmax --size 2x1024x1024 --axis 1")
    compare_filtered("softmax --size 2x1024x1024 --axis 1 --cpu")
    compare_filtered("softmax --size 2x1024x1024 --axis 1 --fused")
    compare_filtered("softmax --size 2x1024x1024 --axis 1 --fused --cpu")
    compare_filtered("relu --size 32x16x1024")
    compare_filtered("relu --size 32x16x1024 --cpu")
    compare_filtered("leaky_relu --size 32x16x1024")
    compare_filtered("leaky_relu --size 32x16x1024 --cpu")
    compare_filtered("elu --size 32x16x1024")
    compare_filtered("elu --size 32x16x1024 --cpu")
    compare_filtered("relu6 --size 32x16x1024")
    compare_filtered("relu6 --size 32x16x1024 --cpu")
    compare_filtered("softplus --size 32x16x1024")
    compare_filtered("softplus --size 32x16x1024 --cpu")
    compare_filtered("celu --size 32x16x1024")
    compare_filtered("celu --size 32x16x1024 --cpu")
    compare_filtered("log_sigmoid --size 32x16x1024")
    compare_filtered("log_sigmoid --size 32x16x1024 --cpu")
    compare_filtered("step --size 32x16x1024")
    compare_filtered("step --size 32x16x1024 --cpu")
    compare_filtered("selu --size 32x16x1024")
    compare_filtered("selu --size 32x16x1024 --cpu")
    # compare_filtered("mish --size 32x16x1024") NOTE: Torch does not implement Mish in MPS atm
    compare_filtered("mish --size 32x16x1024 --cpu")
    compare_filtered("prelu --size 32x16x1024")
    compare_filtered("prelu --size 32x16x1024 --cpu")

    compare_filtered("scalar_mul --size 32x16x1024")
    compare_filtered("scalar_mul --size 32x16x1024 --cpu")
    compare_filtered("cross_entropy --size 256x1024")
    compare_filtered("cross_entropy --size 256x1024 --cpu")
    compare_filtered("logsumexp --size 1024x1024 --axis 1")
    compare_filtered("logsumexp --size 1024x1024 --axis 1 --cpu")
    compare_filtered("logsumexp --size 1024x1024 --axis 0")
    compare_filtered("logsumexp --size 1024x1024 --axis 0 --cpu")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 2")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 2 --cpu")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 1")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 1 --cpu")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 0")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1024x128 --axis 0 --cpu")
    compare_filtered("concatenate --size 32x1024x128 --size 32x16x128 --axis 1")
    compare_filtered("concatenate --size 32x1024x128 --size 32x16x128 --axis 1 --cpu")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1x128 --axis 1")
    compare_filtered("concatenate --size 32x1024x128 --size 32x1x128 --axis 1 --cpu")
    compare_filtered("concatenate --size 1x32x1024x128 --size 1x32x1x128 --axis 2")
    compare_filtered(
        "concatenate --size 1x32x1024x128 --size 1x32x1x128 --axis 2 --cpu"
    )
    compare_filtered("conv1d --size 1x1000x80 --size 128x11x80")
    compare_filtered("conv1d --size 1x1000x80 --size 128x11x80 --cpu")
    compare_filtered("conv1d --size 16x1000x80 --size 128x11x80")
    compare_filtered("conv1d --size 4x1000x80 --size 128x11x80 --cpu")
    compare_filtered("conv2d --size 1x256x256x3 --size 8x3x3x3")
    compare_filtered("conv2d --size 1x256x256x3 --size 8x3x3x3 --cpu")
    compare_filtered("conv2d --size 16x256x256x3 --size 8x3x3x3")
    compare_filtered("conv2d --size 4x256x256x3 --size 8x3x3x3 --cpu")
    compare_filtered("cumsum --size 1024x1024 --axis 1 --cpu")
    compare_filtered("cumsum --size 1024x1024 --axis 0 --cpu")
    compare_filtered("cumsum --size 1024x1024 --axis 1")
    compare_filtered("cumsum --size 1024x1024 --axis 0")
    compare_filtered("cumsum --size 128x1024 --axis 1")
    compare_filtered("cumsum --size 128x1024 --axis 0")
    compare_filtered("cumsum --size 1024x4096 --axis 1")
    compare_filtered("cumsum --size 1024x4096 --axis 0")
    compare_filtered("cumsum --size 128x4096 --axis 1")
    compare_filtered("cumsum --size 128x4096 --axis 0")
    compare_filtered("cumsum --size 1024x7777 --axis 1")
    compare_filtered("cumsum --size 1024x7777 --axis 0")
    compare_filtered("cumsum --size 128x7777 --axis 1")
    compare_filtered("cumsum --size 128x7777 --axis 0")
    compare_filtered("cumsum --size 32768x128 --axis 1")
    compare_filtered("cumsum --size 32768x128 --axis 0")

    compare_filtered("sort --size 1024x1024 --axis 0")
    compare_filtered("sort --size 1024x1024 --axis 1")
    compare_filtered("sort --size 32768x128 --axis 0")
    compare_filtered("sort --size 32768x128 --axis 1")
    compare_filtered("sort --size 128x128 --axis 0 --cpu")
    compare_filtered("sort --size 128x128 --axis 1 --cpu")

    compare_filtered("topk --size 1024x1024 --axis 0")
    compare_filtered("topk --size 1024x1024 --axis 1")
    compare_filtered("topk --size 32768x128 --axis 0")
    compare_filtered("topk --size 32768x128 --axis 1")
    compare_filtered("topk --size 128x128 --axis 0 --cpu")
    compare_filtered("topk --size 128x128 --axis 1 --cpu")

    # Quantized matmul ops (MLX only — compare across bit widths)
    # qmv path (M=1, token generation, memory-bandwidth bound)
    for gs in [64, 128]:
        compare_mlx_quant(
            f"quant_matmul_t_{gs}_{{bits}} --size 1x4096 --size 4096x4096",
            [1, 2, 4, 8],
        )
        compare_mlx_quant(
            f"quant_matmul_t_{gs}_{{bits}} --size 1x4096 --size 11008x4096",
            [1, 2, 4, 8],
        )
    # qmm path (prompt processing, more compute bound)
    for gs in [64, 128]:
        for M in [32, 512]:
            compare_mlx_quant(
                f"quant_matmul_t_{gs}_{{bits}} --size {M}x4096 --size 4096x4096",
                [1, 2, 4, 8],
            )
            compare_mlx_quant(
                f"quant_matmul_t_{gs}_{{bits}} --size {M}x4096 --size 11008x4096",
                [1, 2, 4, 8],
            )
