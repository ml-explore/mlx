# Copyright © 2026 Apple Inc.

import argparse
import platform
import time

import mlx.core as mx


def measure(fn, repeats):
    for _ in range(5):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description="All-element float32 minmax benchmark")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4096, 1 << 20, 1 << 24]
    )
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0 or any(size <= 0 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    if not mx.metal.is_available():
        parser.error("the fused benchmark requires Metal")
    mx.set_default_device(mx.gpu)
    print(f"Hardware: {mx.device_info()['device_name']} ({platform.machine()})")
    print(f"dtype=float32, contiguous, repeats={args.repeats}")
    print("Synchronized end-to-end timings include Python/dispatch overhead.")
    print("Large inputs use a fused first pass and separate final reductions.")
    print("size       fused ms   separate ms   speedup")
    for size in args.sizes:
        x = mx.random.uniform(shape=(size,), dtype=mx.float32)
        mx.eval(x)

        def fused():
            return mx.minmax(x)

        def separate():
            return mx.min(x), mx.max(x)

        actual, expected = fused(), separate()
        mx.eval(actual, expected)
        assert all(a.item() == b.item() for a, b in zip(actual, expected))
        fused_ms = measure(fused, args.repeats)
        separate_ms = measure(separate, args.repeats)
        print(
            f"{size:<10} {fused_ms:9.5f} {separate_ms:13.5f} "
            f"{separate_ms / fused_ms:9.2f}x"
        )


if __name__ == "__main__":
    main()
