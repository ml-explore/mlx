"""Benchmark single-token decode through the Metal SDPA vector kernels.

Run the same command after building the PR parent and head revisions:

    python benchmarks/python/sdpa_vector_decode_bench.py

The default shapes select the generic ``sdpa_vector_2pass_1`` kernel changed
by this PR. Use ``--shape QH,KVH,CTX,D`` to benchmark additional shapes.
"""

import argparse
import math
import statistics
import time

import mlx.core as mx


DEFAULT_SHAPES = (
    # q_heads, kv_heads, context, head_dim
    (32, 32, 4096, 128),
    (32, 32, 32768, 128),
    (4, 4, 65536, 128),
    (8, 4, 4096, 128),
    (8, 4, 32768, 128),
    (16, 4, 4096, 128),
    (16, 4, 32768, 128),
    # GQA=8 selects the generic 2-pass kernel below context 8192.
    (32, 4, 4096, 128),
)


def parse_shape(value):
    try:
        shape = tuple(int(x) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be q_heads,kv_heads,context,head_dim"
        ) from error
    if len(shape) != 4:
        raise argparse.ArgumentTypeError(
            "shape must be q_heads,kv_heads,context,head_dim"
        )
    q_heads, kv_heads, context, head_dim = shape
    if min(shape) <= 0 or q_heads % kv_heads != 0:
        raise argparse.ArgumentTypeError(
            "shape values must be positive and q_heads divisible by kv_heads"
        )
    return shape


def kernel_path(q_heads, kv_heads, context, head_dim, architecture):
    gqa = q_heads // kv_heads
    device_class = architecture[-1] if architecture else ""
    uses_two_pass = (
        device_class in ("d", "s") and context >= 1024
    ) or (kv_heads < q_heads and context >= 4096)
    if not uses_two_pass:
        return "vector-1pass"

    specialized_gqa = (gqa == 8 and head_dim in (64, 128)) or (
        gqa in (12, 16) and head_dim == 128
    )
    if specialized_gqa and context >= 8192:
        return "gqa-specialized"
    return "generic-2pass"


def reference_sdpa(q, k, v, scale):
    n_repeats = q.shape[1] // k.shape[1]
    if n_repeats > 1:
        k = mx.repeat(k, n_repeats, axis=1)
        v = mx.repeat(v, n_repeats, axis=1)
    scores = (q @ mx.swapaxes(k, -1, -2)) * scale
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return probabilities @ v


def run_chain(q, k, v, scale, chain):
    out = q
    for _ in range(chain):
        out = mx.fast.scaled_dot_product_attention(
            out, k, v, scale=scale
        )
    mx.eval(out)


def measure(q, k, v, scale, chain, warmup, iterations, rounds):
    for _ in range(warmup):
        run_chain(q, k, v, scale, chain)

    round_medians = []
    best = math.inf
    for _ in range(rounds):
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            run_chain(q, k, v, scale, chain)
            elapsed = (time.perf_counter_ns() - start) * 1e-9 / chain
            samples.append(elapsed)
        best = min(best, min(samples))
        round_medians.append(statistics.median(samples))

    return best, statistics.median(round_medians), min(round_medians), max(
        round_medians
    )


def benchmark_shape(shape, args, architecture):
    q_heads, kv_heads, context, head_dim = shape
    scale = 1.0 / math.sqrt(head_dim)

    mx.random.seed(args.seed)
    q = mx.random.normal((1, q_heads, 1, head_dim)).astype(mx.float16)
    k = mx.random.normal((1, kv_heads, context, head_dim)).astype(mx.float16)
    v = mx.random.normal((1, kv_heads, context, head_dim)).astype(mx.float16)
    mx.eval(q, k, v)

    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    expected = reference_sdpa(q, k, v, scale)
    mx.eval(actual, expected)
    error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    max_abs_error = error.max().item()
    relative_linf_error = max_abs_error / max(
        mx.abs(expected).max().item(), 1e-9
    )

    best, median, median_low, median_high = measure(
        q,
        k,
        v,
        scale,
        args.chain,
        args.warmup,
        args.iterations,
        args.rounds,
    )

    # Logical traffic of the generic kernel: every query head streams one K
    # and one V head. This is intentionally called logical bandwidth because
    # shared KV heads can be served from cache on GQA shapes.
    itemsize = 2
    logical_bytes = (
        q_heads * context * head_dim * itemsize * 2
    )
    logical_gbps = logical_bytes / median / 1e9

    return {
        "gqa": q_heads // kv_heads,
        "path": kernel_path(*shape, architecture),
        "best_us": best * 1e6,
        "median_us": median * 1e6,
        "median_low_us": median_low * 1e6,
        "median_high_us": median_high * 1e6,
        "logical_gbps": logical_gbps,
        "max_abs_error": max_abs_error,
        "relative_linf_error": relative_linf_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-token decode through the Metal "
            "sdpa_vector_2pass kernels."
        )
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        metavar="QH,KVH,CTX,D",
        help="shape to benchmark; may be repeated",
    )
    parser.add_argument("--chain", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shapes = args.shape or DEFAULT_SHAPES
    device_info = mx.device_info(mx.gpu)
    architecture = device_info.get("architecture", "")
    print(
        f"device={device_info.get('device_name', 'unknown')} "
        f"architecture={architecture}"
    )
    print(
        f"chain={args.chain} warmup={args.warmup} "
        f"iterations={args.iterations} rounds={args.rounds}"
    )
    print(
        f"{'QH':>3} {'KVH':>3} {'GQA':>3} {'CTX':>6} {'D':>3} "
        f"{'path':>15} {'best us':>9} {'median us [range]':>27} "
        f"{'logical GB/s':>12} {'rel L-inf':>9}"
    )
    for shape in shapes:
        result = benchmark_shape(shape, args, architecture)
        q_heads, kv_heads, context, head_dim = shape
        print(
            f"{q_heads:>3} {kv_heads:>3} {result['gqa']:>3} "
            f"{context:>6} {head_dim:>3} {result['path']:>15} "
            f"{result['best_us']:>9.1f} "
            f"{result['median_us']:>9.1f} "
            f"[{result['median_low_us']:>6.1f},"
            f"{result['median_high_us']:>6.1f}] "
            f"{result['logical_gbps']:>12.1f} "
            f"{result['relative_linf_error']:>9.2e}"
        )


if __name__ == "__main__":
    main()
