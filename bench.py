import os
import shutil
import time

import mlx.core as mx
import numpy as np

import mlx


def bench(fn, rounds=20, label=""):
    for _ in range(3):
        r = fn()
        mx.eval(r)

    times = []
    for _ in range(rounds):
        mx.eval()
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    worst = times[-1]
    print(f"{label}")
    print(
        f"median={median * 1000:.3f}ms | min={best * 1000:.3f}ms | max={worst * 1000:.3f}ms"
    )
    return r


def capture_trace(path, fn, warmup=5, iters=10):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

    for _ in range(warmup):
        r = fn()
        mx.eval(r)

    # Flush warmup work before starting the capture
    mx.synchronize(mx.gpu)

    mx.metal.start_capture(path)
    print(f"Capturing trace to {path}...")
    for _ in range(iters):
        r = fn()
        mx.eval(r)
        print(mx.array(r))  # Force synchronization to ensure the work is captured

    # Drain work that was enqueued during capture, then stop
    mx.synchronize(mx.gpu)
    mx.metal.stop_capture()


for size in [1, 16, 256, 256, 4096, 16384, 131072, 1000000, 50000000]:
    with mx.stream(mx.gpu):
        a = mx.random.normal(shape=(size,), dtype=mx.float32, stream=mx.gpu)
        b = mx.random.normal(shape=(size,), dtype=mx.float32, stream=mx.gpu)
    a_np = np.array(a, copy=False)
    b_np = np.array(b, copy=False)

    print(f"Size: {size}")
    ccc = bench(lambda: mx.inner(a, b), label="MLX native")
    cc = bench(lambda: np.dot(a_np, b_np), label="NumPy")
    # aligned = bench(lambda: mx.inner(a[:-1], b[:-1]), label="aligned, same length")
    # unaligned = bench(lambda: mx.inner(a[1:], b[1:]), label="unaligned, same length")

    # ref = np.dot(a_np[1:], b_np[1:])
    # print(f"mx.inner : {float(ccc)}")
    # print(f"numpy : {float(cc)}")
    print(f"rel error : {abs(float(ccc) - float(cc)) / abs(float(cc)) * 100:.6f}%")

# capture_trace("trace.gputrace", lambda: mx.inner(a, b))
