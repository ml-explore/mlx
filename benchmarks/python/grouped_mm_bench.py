# Copyright © 2026 Apple Inc.

import argparse
import statistics
import time

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m", type=int, default=32768, help="Input rows before routing"
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--skewed", action="store_true")
    parser.add_argument("--transpose-b", action="store_true")
    parser.add_argument("--compare-gather", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.groups, args.top_k, args.iterations) <= 0:
        parser.error("Dimensions, group count, and iterations must be positive")
    if args.top_k > args.groups:
        parser.error("top-k must not exceed the number of groups")
    if args.top_k > 1 and args.skewed:
        parser.error("top-k routing uses random scores; omit --skewed")

    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    dtype = getattr(mx, args.dtype)
    sizes = [args.m // args.groups] * args.groups
    sizes[-1] += args.m % args.groups
    if args.skewed and args.groups > 1:
        sizes = [args.m // (8 * args.groups)] * args.groups
        sizes[0] += args.m - sum(sizes)
    offsets = mx.array([sum(sizes[:g]) for g in range(args.groups)], mx.int32)
    a = mx.random.normal((args.m, args.k), dtype=dtype)
    if args.top_k > 1:
        scores = mx.random.uniform(shape=(args.m, args.groups))
        routes = mx.argpartition(scores, args.groups - args.top_k, axis=-1)[
            :, -args.top_k :
        ].flatten()
        order = mx.argsort(routes)
        indices = routes[order].astype(mx.uint32)
        a = a[order // args.top_k]
        offsets = mx.searchsorted(
            indices, mx.arange(args.groups, dtype=mx.uint32)
        ).astype(mx.int32)
    b_shape = (
        (args.groups, args.n, args.k)
        if args.transpose_b
        else (args.groups, args.k, args.n)
    )
    b = mx.random.normal(b_shape, dtype=dtype)
    if args.transpose_b:
        b = b.swapaxes(-1, -2)
    mx.eval(a, b, offsets)

    functions = {"grouped": lambda: mx.grouped_mm(a, b, token_offsets=offsets)}
    if args.compare_gather:
        if args.top_k == 1:
            indices = mx.array(
                [g for g, size in enumerate(sizes) for _ in range(size)], mx.uint32
            )
        a_gather = a[:, None, :]
        mx.eval(a_gather, indices)
        functions["gather"] = lambda: mx.gather_mm(
            a_gather, b, rhs_indices=indices, sorted_indices=True
        )
        grouped = functions["grouped"]().astype(mx.float32)
        gathered = functions["gather"]().squeeze(-2).astype(mx.float32)
        if not mx.allclose(grouped, gathered, rtol=1e-2, atol=1e-2).item():
            raise AssertionError("grouped_mm and gather_mm outputs differ")
        max_abs_diff = mx.max(mx.abs(grouped - gathered)).item()

    for _ in range(5):
        for fn in functions.values():
            mx.eval(fn())
    times = {name: [] for name in functions}
    for iteration in range(args.iterations):
        order = list(functions)
        if iteration % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            mx.eval(functions[name]())
            times[name].append(time.perf_counter() - start)
    elapsed = statistics.median(times["grouped"])
    routed_m = args.m * args.top_k
    print(
        f"tokens={args.m} top_k={args.top_k} M={routed_m} "
        f"N={args.n} K={args.k} groups={args.groups} "
        f"dtype={args.dtype} skewed={args.skewed} transpose_b={args.transpose_b} "
        f"median_ms={elapsed * 1e3:.4f} "
        f"tflops={2 * routed_m * args.n * args.k / elapsed / 1e12:.3f}"
    )
    if args.compare_gather:
        gather_elapsed = statistics.median(times["gather"])
        print(
            f"gather_ms={gather_elapsed * 1e3:.4f} "
            f"grouped_ms={elapsed * 1e3:.4f} "
            f"speedup={gather_elapsed / elapsed:.3f} "
            f"max_abs_diff={max_abs_diff:.6g}"
        )


if __name__ == "__main__":
    main()
