"""Benchmark single-row affine gather-QMV on Metal.

Run this script after building the parent and PR revisions:

    python benchmarks/python/gather_qmv_bench.py

Each shape is ``BITS,K,N``. The defaults cover the existing fast/scalar
alignment classes and medium-alignment candidates added by this PR.
"""

import argparse
import math
import statistics
import time

import mlx.core as mx


DEFAULT_SHAPES = (
    # bits, K, N
    (4, 256, 1024),  # medium
    (4, 512, 1024),  # fast control
    (4, 640, 1024),  # scalar control
    (4, 768, 2048),  # medium, Qwen3 MoE down projection
    (4, 1280, 2048),  # medium
    (5, 768, 1024),  # medium
    (6, 384, 1024),  # medium
    (8, 384, 1024),  # medium
)


def parse_shape(value):
    try:
        shape = tuple(int(x) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be BITS,K,N") from error
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be BITS,K,N")
    bits, in_features, out_features = shape
    if bits not in (2, 3, 4, 5, 6, 8):
        raise argparse.ArgumentTypeError("BITS must be one of 2,3,4,5,6,8")
    if in_features <= 0 or out_features <= 0:
        raise argparse.ArgumentTypeError("K and N must be positive")
    return shape


def pack_factor(bits):
    if bits in (3, 5):
        return 8
    if bits == 6:
        return 4
    return 32 // bits


def alignment_class(bits, in_features, out_features):
    factor = pack_factor(bits)
    fast_alignment = factor * (1 if bits == 2 else 2) * 32
    medium_alignment = factor * 32
    if out_features % 8 == 0 and in_features % fast_alignment == 0:
        return "fast"
    if (
        bits >= 4
        and out_features % 8 == 0
        and in_features % medium_alignment == 0
    ):
        return "medium-candidate"
    return "scalar"


def make_inputs(
    bits, in_features, out_features, num_experts, top_k, group_size, seed
):
    mx.random.seed(seed)
    scale = 1.0 / math.sqrt(in_features)
    weight = (
        mx.random.normal((num_experts, out_features, in_features)) * scale
    ).astype(mx.float16)
    quantized = mx.quantize(weight, group_size=group_size, bits=bits)
    del weight

    x = mx.random.normal((1, top_k, 1, in_features)).astype(mx.float16)
    indices = mx.arange(top_k, dtype=mx.uint32)[None]
    mx.eval(x, indices, quantized)
    return x, indices, quantized


def gather_qmv(x, indices, quantized, group_size, bits):
    return mx.gather_qmm(
        x,
        *quantized,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


def check_correctness(x, indices, quantized, group_size, bits):
    actual = gather_qmv(x, indices, quantized, group_size, bits)
    selected = tuple(array[indices] for array in quantized)
    weight = mx.dequantize(
        *selected, group_size=group_size, bits=bits
    )
    expected = x @ mx.swapaxes(weight, -1, -2)
    mx.eval(actual, expected)

    error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    max_abs_error = error.max().item()
    relative_linf_error = max_abs_error / max(
        mx.abs(expected).max().item(), 1e-9
    )
    return max_abs_error, relative_linf_error


def run_chain(x, indices, quantized, group_size, bits, chain):
    outputs = [
        gather_qmv(x, indices, quantized, group_size, bits)
        for _ in range(chain)
    ]
    mx.eval(outputs)


def measure(
    x,
    indices,
    quantized,
    group_size,
    bits,
    chain,
    warmup,
    iterations,
    rounds,
):
    for _ in range(warmup):
        run_chain(x, indices, quantized, group_size, bits, chain)

    best = math.inf
    round_medians = []
    for _ in range(rounds):
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            run_chain(x, indices, quantized, group_size, bits, chain)
            elapsed = (time.perf_counter_ns() - start) * 1e-9 / chain
            best = min(best, elapsed)
            samples.append(elapsed)
        round_medians.append(statistics.median(samples))

    return best, statistics.median(round_medians), min(round_medians), max(
        round_medians
    )


def logical_bytes(
    bits, in_features, out_features, top_k, group_size
):
    weight = top_k * out_features * in_features * bits // 8
    scales_and_biases = (
        2 * top_k * out_features * (in_features // group_size) * 2
    )
    activations = top_k * (in_features + out_features) * 2
    return weight + scales_and_biases + activations


def benchmark_shape(shape, args):
    bits, in_features, out_features = shape
    if in_features % args.group_size != 0:
        raise ValueError(
            f"K={in_features} must be divisible by group size "
            f"{args.group_size}"
        )
    if args.top_k > args.num_experts:
        raise ValueError("top-k must not exceed the number of experts")

    x, indices, quantized = make_inputs(
        bits,
        in_features,
        out_features,
        args.num_experts,
        args.top_k,
        args.group_size,
        args.seed,
    )
    max_abs_error, relative_linf_error = check_correctness(
        x, indices, quantized, args.group_size, bits
    )
    best, median, median_low, median_high = measure(
        x,
        indices,
        quantized,
        args.group_size,
        bits,
        args.chain,
        args.warmup,
        args.iterations,
        args.rounds,
    )
    traffic = logical_bytes(
        bits,
        in_features,
        out_features,
        args.top_k,
        args.group_size,
    )
    return {
        "alignment": alignment_class(bits, in_features, out_features),
        "best_us": best * 1e6,
        "median_us": median * 1e6,
        "median_low_us": median_low * 1e6,
        "median_high_us": median_high * 1e6,
        "logical_gbps": traffic / median / 1e9,
        "max_abs_error": max_abs_error,
        "relative_linf_error": relative_linf_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark affine gather-QMV with M=1 on Metal."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        metavar="BITS,K,N",
        help="shape to benchmark; may be repeated",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--chain", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_info = mx.device_info(mx.gpu)
    print(
        f"device={device_info.get('device_name', 'unknown')} "
        f"architecture={device_info.get('architecture', 'unknown')}"
    )
    print(
        f"experts={args.num_experts} top_k={args.top_k} "
        f"group_size={args.group_size} chain={args.chain} "
        f"warmup={args.warmup} iterations={args.iterations} "
        f"rounds={args.rounds}"
    )
    print(
        f"{'bits':>4} {'K':>5} {'N':>5} {'alignment':>16} "
        f"{'best us':>9} {'median us [range]':>27} "
        f"{'logical GB/s':>12} {'rel L-inf':>9}"
    )
    for shape in args.shape or DEFAULT_SHAPES:
        result = benchmark_shape(shape, args)
        bits, in_features, out_features = shape
        print(
            f"{bits:>4} {in_features:>5} {out_features:>5} "
            f"{result['alignment']:>16} {result['best_us']:>9.1f} "
            f"{result['median_us']:>9.1f} "
            f"[{result['median_low_us']:>6.1f},"
            f"{result['median_high_us']:>6.1f}] "
            f"{result['logical_gbps']:>12.1f} "
            f"{result['relative_linf_error']:>9.2e}"
        )


if __name__ == "__main__":
    main()
