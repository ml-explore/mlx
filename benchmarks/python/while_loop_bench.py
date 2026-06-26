# Copyright © 2026 Apple Inc.
"""Benchmark: chunked mx.while_loop vs naive per-iter-sync Python loop.

Reproduces the per-iteration GPU-sync pathology of
https://github.com/tillahoffmann/jax-mps/issues/83 and shows the chunked
amortization. Prints results; does NOT assert a speedup.
"""

import math
import time

import mlx.core as mx


def naive_while(cond, body, init, max_iterations):
    """Python while loop that syncs every iteration (the pathology)."""
    carry = init
    n = 0
    while bool(mx.all(cond(carry))):
        carry = body(carry)
        mx.eval(carry)  # per-iteration host sync
        n += 1
        if n >= max_iterations:
            break
    return carry


def time_it(fn, *args, warmup=2, iters=5, **kwargs):
    """Return median ms per call (syncs included)."""
    for _ in range(warmup):
        mx.eval(fn(*args, **kwargs))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn(*args, **kwargs))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    print(f"MLX version: {mx.__version__}  Metal: {mx.metal.is_available()}")

    def make_cond(n):
        # Capture n by value (default arg) to avoid late-binding surprises if
        # n is reassigned later in this function.
        def cond(c, _n=n):
            return c < _n

        return cond

    def body(c):
        return c + 1

    # Warm up compile caches.
    mx.while_loop(
        make_cond(1000), body, mx.array(0), max_iterations=2000, chunk_size=64
    )
    naive_while(make_cond(100), body, mx.array(0), 100)

    # Naive is O(N) syncs; only measure at N=1000 (the pathology reproducer).
    print("\n=== Naive (per-iter sync) vs chunked (chunk_size=64), N=1000 ===")
    N = 1000
    cond = make_cond(N)
    init = mx.array(0)
    t_naive = time_it(naive_while, cond, body, init, N)
    t_chunk = time_it(
        lambda init, cap, cs: mx.while_loop(
            cond, body, init, max_iterations=cap, chunk_size=cs
        ),
        init,
        N * 2,
        64,
    )
    print(
        f"  N={N}: naive={t_naive:8.1f}ms  chunked={t_chunk:7.1f}ms  "
        f"speedup={t_naive / t_chunk:5.1f}x  syncs_chunked~={math.ceil(N / 64)}"
    )

    print("\n=== chunked only at N=10000 (naive too slow to time) ===")
    N = 10000
    cond = make_cond(N)
    init = mx.array(0)
    t_chunk10 = time_it(
        lambda init, cap, cs: mx.while_loop(
            cond, body, init, max_iterations=cap, chunk_size=cs
        ),
        init,
        N * 2,
        64,
    )
    print(f"  N={N} chunked(cs=64): {t_chunk10:7.1f}ms  syncs~={math.ceil(N / 64)}")

    print("\n=== chunk_size sweep at N=10000 (skipping cs<4, too slow) ===")
    for cs in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
        t = time_it(
            lambda init, cap, cs: mx.while_loop(
                cond, body, init, max_iterations=cap, chunk_size=cs
            ),
            init,
            N * 2,
            cs,
        )
        print(f"  chunk_size={cs:4d}: {t:7.1f}ms  syncs~={math.ceil(N / cs)}")


if __name__ == "__main__":
    main()
