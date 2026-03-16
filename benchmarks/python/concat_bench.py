# Copyright © 2025 Apple Inc.
import mlx.core as mx
from time_utils import measure_runtime


def benchmark_concat(arrays, axis):
    def concat(arrays):
        mx.eval(mx.concatenate(arrays, axis=axis))

    runtime = measure_runtime(concat, arrays=arrays)
    return runtime


if __name__ == "__main__":

    for n_arrays in [32]:
        for shape in [(4096, 4096)]:
            arrays = [mx.random.normal(shape) for _ in range(n_arrays)]
            mx.eval(arrays)
            runtime = benchmark_concat(arrays, axis=0)
            total_elems = n_arrays * shape[0] * shape[1]
            print(
                f"  {n_arrays}x {shape}  axis=0  "
                f"({total_elems / 1e6:.1f}M elems): {runtime:.3f}ms"
            )