# Copyright © 2026 Apple Inc.

import argparse

import mlx.core as mx
from time_utils import measure_runtime


def gumbel_categorical(logits, num_samples):
    """The Gumbel-max trick, which perturbs every category once per sample."""
    g = mx.random.gumbel(shape=(num_samples, logits.size))
    return mx.argmax(g + logits, axis=-1)


def peak_memory(fn, **kwargs):
    mx.reset_peak_memory()
    fn(**kwargs)
    return mx.get_peak_memory()


def bench(n, m, memory_budget):
    logits = mx.zeros((n,))
    mx.eval(logits)

    def inverse_cdf():
        mx.eval(mx.random.categorical(logits, num_samples=m))

    def gumbel():
        mx.eval(gumbel_categorical(logits, m))

    cdf_time = measure_runtime(inverse_cdf)
    cdf_peak = peak_memory(inverse_cdf)

    # The Gumbel path holds an (m, n) perturbation and the same again for the
    # sum, so only run it while that fits
    if 8 * n * m <= memory_budget:
        gumbel_time = measure_runtime(gumbel)
        gumbel_peak = peak_memory(gumbel)
    else:
        gumbel_time = gumbel_peak = float("nan")

    return cdf_time, cdf_peak, gumbel_time, gumbel_peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Categorical sampling benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=4e9,
        help="Skip the Gumbel path once its intermediate exceeds this many bytes.",
    )
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)

    header = f"{'N':>9} {'M':>9} {'cdf':>12} {'gumbel':>12} {'cdf peak':>12} {'gumbel peak':>13}"
    print(header)
    print("-" * len(header))
    for n, m in [
        (128, 8),
        (10**4, 10**4),
        (5 * 10**4, 10**4),
        (10**5, 10**5),
        (10**6, 10**6),
    ]:
        cdf_time, cdf_peak, gumbel_time, gumbel_peak = bench(n, m, args.memory_budget)
        gumbel_cell = (
            "skipped" if gumbel_time != gumbel_time else f"{gumbel_time:10.3f}ms"
        )
        gumbel_peak_cell = (
            "skipped" if gumbel_peak != gumbel_peak else f"{gumbel_peak / 1e6:10.1f}MB"
        )
        print(
            f"{n:>9} {m:>9} {cdf_time:10.3f}ms {gumbel_cell:>12} "
            f"{cdf_peak / 1e6:10.1f}MB {gumbel_peak_cell:>13}"
        )
