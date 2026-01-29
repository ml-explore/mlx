import argparse
import time

import mlx.core as mx


def time_fn(fn, *args, warmup=5, iters=100, **kwargs):
    """Time a function, return milliseconds per call."""
    for _ in range(warmup):
        mx.eval(fn(*args, **kwargs))

    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    return 1e3 * (toc - tic) / iters


def quant_sdpa(q, k, v, bits, mode, loops=1):
    for _ in range(loops):
        q = mx.fast.quantized_scaled_dot_product_attention(
            q, *k, *v, scale=1.0, mask=None, bits=bits, mode=mode
        )
    return q


def sdpa(q, k, v, loops=1):
    for _ in range(loops):
        q = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)
    return q


def run_benchmark(
    seq_lengths,
    modes,
    H=32,
    H_k=8,
    D=128,
    dtype=mx.float16,
    loops=20,
    warmup=5,
    iters=100,
):
    """Run benchmarks across sequence lengths and modes."""
    results = {}

    print(f"\n{'=' * 70}")
    print(f"Quant SDPA Benchmark: H={H}, H_k={H_k}, D={D}, GQA={H // H_k}x")
    print(f"{'=' * 70}")

    # Header
    header = f"{'SeqLen':>8}"
    for mode, bits in modes:
        header += f" | {mode}({bits}b):ms"
    header += " | fp16:ms"
    print(header)
    print("-" * len(header))

    for L in seq_lengths:
        mx.random.seed(42)
        q = mx.random.uniform(shape=(1, H, 1, D), dtype=dtype)
        k = mx.random.uniform(shape=(1, H_k, L, D), dtype=dtype)
        v = mx.random.uniform(shape=(1, H_k, L, D), dtype=dtype)
        mx.eval(q, k, v)

        row = f"{L:>8}"
        results[L] = {}

        # Benchmark each quant mode
        for mode, bits in modes:
            k_quant = mx.quantize(k, bits=bits, mode=mode)
            v_quant = mx.quantize(v, bits=bits, mode=mode)
            mx.eval(k_quant, v_quant)

            ms = time_fn(
                quant_sdpa,
                q,
                k_quant,
                v_quant,
                bits,
                mode,
                loops=loops,
                warmup=warmup,
                iters=iters,
            )
            ms_per_call = ms / loops
            results[L][(mode, bits)] = ms_per_call
            row += f" |    {ms_per_call:8.4f}"

        # Benchmark fp16 baseline
        ms = time_fn(sdpa, q, k, v, loops=loops, warmup=warmup, iters=iters)
        ms_per_call = ms / loops
        results[L]["fp16"] = ms_per_call
        row += f" |  {ms_per_call:8.4f}"

        print(row)

    return results


def print_speedup_table(results, modes):
    """Print speedup vs fp16 baseline."""
    print(f"\n{'=' * 60}")
    print("Speedup vs fp16")
    print(f"{'=' * 60}")

    header = f"{'SeqLen':>8}"
    for mode, bits in modes:
        header += f" | {mode}({bits}b)"
    print(header)
    print("-" * len(header))

    for L, data in results.items():
        fp16_ms = data["fp16"]
        row = f"{L:>8}"
        for mode, bits in modes:
            quant_ms = data[(mode, bits)]
            speedup = fp16_ms / quant_ms
            row += f" |   {speedup:5.2f}x"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Quant SDPA")
    parser.add_argument("--heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--loops", type=int, default=20, help="Loops per timing call")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timing iterations")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["mxfp4", "mxfp8"],
        help="Quantization modes to test",
    )
    args = parser.parse_args()

    # Map mode names to (mode, bits)
    mode_map = {
        "mxfp4": ("mxfp4", 4),
        "mxfp8": ("mxfp8", 8),
        "affine4": ("affine", 4),
        "affine8": ("affine", 8),
        "nvfp4": ("nvfp4", 4),
    }

    modes = [mode_map[m] for m in args.modes if m in mode_map]

    results = run_benchmark(
        seq_lengths=args.seq_lengths,
        modes=modes,
        H=args.heads,
        H_k=args.kv_heads,
        D=args.dim,
        loops=args.loops,
        warmup=args.warmup,
        iters=args.iters,
    )

    print_speedup_table(results, modes)


if __name__ == "__main__":
    main()
