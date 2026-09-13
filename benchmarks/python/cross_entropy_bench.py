"""Fused Metal cross entropy vs the decomposed fallback."""

import time

import mlx.core as mx


def fallback(logits, targets):
    score = mx.take_along_axis(logits, mx.expand_dims(targets, -1), -1).squeeze(-1)
    return mx.logsumexp(logits.astype(mx.float32), axis=-1) - score.astype(mx.float32)


def timeit(fn, *args, iters=50, warmup=10):
    # Multi-GiB logits leave the buffer cache under pressure, which otherwise
    # bleeds between measurements and makes later shapes look far slower.
    mx.clear_cache()
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


shapes = [
    (8192, 4096),
    (4096, 32000),
    (2048, 128256),
    (1024, 151936),
]

print(
    f"{'rows x vocab':>20} {'dtype':>10} {'fallback':>10} {'fused':>10} {'speedup':>9}"
)
print("-" * 64)

for dtype in [mx.float32, mx.bfloat16]:
    for rows, V in shapes:
        logits = mx.random.normal(shape=(rows, V), scale=2.0).astype(dtype)
        targets = mx.random.randint(0, V, shape=(rows,))
        mx.eval(logits, targets)

        t_ref = timeit(lambda: fallback(logits, targets))
        t_fused = timeit(lambda: mx.fast.cross_entropy(logits, targets))
        print(
            f"{rows:>7} x {V:<10} {str(dtype).split('.')[-1]:>10} "
            f"{t_ref:>9.3f}ms {t_fused:>9.3f}ms {t_ref / t_fused:>8.2f}x"
        )

print()
print("--- forward + backward ---")
print(
    f"{'rows x vocab':>20} {'dtype':>10} {'fallback':>10} {'fused':>10} {'speedup':>9}"
)
print("-" * 64)

for dtype in [mx.float32, mx.bfloat16]:
    for rows, V in shapes:
        logits = mx.random.normal(shape=(rows, V), scale=2.0).astype(dtype)
        targets = mx.random.randint(0, V, shape=(rows,))
        mx.eval(logits, targets)

        g_ref = mx.grad(lambda x, y: fallback(x, y).sum(), argnums=0)
        g_fused = mx.grad(lambda x, y: mx.fast.cross_entropy(x, y).sum(), argnums=0)
        t_ref = timeit(g_ref, logits, targets, iters=20)
        t_fused = timeit(g_fused, logits, targets, iters=20)
        print(
            f"{rows:>7} x {V:<10} {str(dtype).split('.')[-1]:>10} "
            f"{t_ref:>9.3f}ms {t_fused:>9.3f}ms {t_ref / t_fused:>8.2f}x"
        )

print()
print("--- peak memory, forward + backward, 4096 x 128256 bf16 ---")
for label, fn in [
    ("fallback", lambda x, y: fallback(x, y).sum()),
    ("fused", lambda x, y: mx.fast.cross_entropy(x, y).sum()),
]:
    logits = mx.random.normal(shape=(4096, 128256), scale=2.0).astype(mx.bfloat16)
    targets = mx.random.randint(0, 128256, shape=(4096,))
    mx.eval(logits, targets)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(mx.grad(fn, argnums=0)(logits, targets))
    print(f"  {label:>10}: {mx.get_peak_memory() / 2**30:.2f} GiB")
