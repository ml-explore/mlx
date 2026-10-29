import argparse
import csv
import itertools
import os
import time

import mlx.core as mx

RESET = "\033[0m"

N_warmup = 8
N_iter_bench = 80
N_iter_func = 5


def bench(f, *args):
    for _ in range(N_warmup):
        f(*args)
    mx.synchronize()

    s = time.perf_counter_ns()
    for _ in range(N_iter_bench):
        f(*args)
    mx.synchronize()
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def do_kernel_bench(f, *args):
    ys = []
    for _ in range(N_iter_func):
        out, hf = f(*args)
        ys.append(out)
        ys.append(hf)
    mx.eval(ys)
    return ys


def do_grad_bench(f, *args):
    ys = []
    for _ in range(N_iter_func):
        ys.extend(f(*args))
    mx.eval(ys)
    return ys


def make_grad_fn():
    def f(q, k, v, g, b, h0):
        out, state = mx.fast.gated_delta_update(q, k, v, g, b, h0)
        return out.sum() + state.sum()

    return mx.grad(f, argnums=(0, 1, 2, 3, 4, 5))


def make_inputs(B, T, Hk, Hv, Dk, Dv):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.normal(shape=(B, T, Hv)) * 0.1 - 1.0
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    mx.eval(q, k, v, g, b, h0)
    return q, k, v, g, b, h0


def benchmark_shape(B, T, Hk, Hv, Dk, Dv, chunk_sizes):
    q, k, v, g, b, h0 = make_inputs(B, T, Hk, Hv, Dk, Dv)

    shape_str = f"B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}"
    denom = N_iter_bench * N_iter_func

    os.environ["GATED_DELTA_CHUNK"] = "0"
    mx.eval(*mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0))
    ms_seq = (
        bench(do_kernel_bench, mx.fast.gated_delta_update, q, k, v, g, b, h0)
        / denom
        * 1e3
    )

    speedups = []
    for C in (c for c in chunk_sizes if c != 0):
        try:
            os.environ["GATED_DELTA_CHUNK"] = str(C)
            mx.eval(*mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0))
            ms_c = (
                bench(do_kernel_bench, mx.fast.gated_delta_update, q, k, v, g, b, h0)
                / denom
                * 1e3
            )
            speedups.append(ms_seq / ms_c if ms_c > 0 else float("nan"))
        except Exception as ex:
            print(f"  chunk {C} failed: {ex}")
            speedups.append(float("nan"))

    return shape_str, f"{ms_seq:.3f}", speedups, ms_seq


def run_benchmark(run_full, to_csv=False, csv_path="benchmark_results.csv"):
    if run_full:
        Bs = [1, 4, 8, 16]
        Ts = [8, 64, 256, 512, 1024, 2048]
    else:
        Bs = [1, 8, 16]
        Ts = [8, 512, 1024, 2048]
    Hks, Hvs, Dks, Dvs = [16], [32], [128], [128]

    chunk_sizes = [0, 8, 16]
    non_zero_Cs = [C for C in chunk_sizes if C != 0]

    headers = ["B", "T", "Hk", "Hv", "Dk", "Dv", "time_seq (ms)"] + [
        f"C={C} (speedup)" for C in non_zero_Cs
    ]

    col_widths = [6, 6, 6, 6, 6, 6, 15] + [25] * len(non_zero_Cs)
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)

    rows = []

    print(fmt.format(*headers))
    print("-" * sum(col_widths))

    for B, T, Hk, Hv, Dk, Dv in itertools.product(Bs, Ts, Hks, Hvs, Dks, Dvs):
        shapes_s, base_time_s, speedups, base_time = benchmark_shape(
            B, T, Hk, Hv, Dk, Dv, chunk_sizes
        )
        row = [f"{B}", f"{T}", f"{Hk}", f"{Hv}", f"{Dk}", f"{Dv}", base_time_s]
        for speed in speedups:
            row.append(f"{(base_time / speed):<8.2f} ({speed:<5.2f}x)")

        print(fmt.format(*row), end="")
        print(f"{RESET}")

        rows.append(row)

    if to_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nResults also written to {csv_path}")


def benchmark_variants_shape(B, T, Hk, Hv, Dk, Dv, do_backward, variants):
    q, k, v, g, b, h0 = make_inputs(B, T, Hk, Hv, Dk, Dv)
    denom = N_iter_bench * N_iter_func

    if do_backward:
        fn = make_grad_fn()
        runner = do_grad_bench
    else:
        fn = mx.fast.gated_delta_update
        runner = do_kernel_bench

    def time_one(variant):
        os.environ["GATED_DELTA_VJP_FALLBACK"] = "1" if variant == "fallback" else "0"
        C = "16" if variant == "nax" else "0"
        if do_backward:
            os.environ["GATED_DELTA_CHUNK"] = "16"
            os.environ["GATED_DELTA_CHUNK_VJP"] = C
        else:
            os.environ["GATED_DELTA_CHUNK"] = C
        mx.eval(*fn(q, k, v, g, b, h0))
        return bench(runner, fn, q, k, v, g, b, h0) / denom * 1e3

    times = []
    for variant in variants:
        try:
            times.append(time_one(variant))
        except Exception as ex:
            print(f"  {variant} failed: {ex}")
            times.append(float("nan"))
    mx.clear_cache()
    return times


def run_variants_benchmark(run_full, do_backward=False, do_fallback=False):
    if run_full:
        Bs = [1, 4, 8, 16]
        Ts = [8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        Bs = [1, 8]
        Ts = [8, 512, 1024]
    if do_fallback:
        Ts = [8, 32, 64, 128, 256, 512]
    Hks, Hvs, Dks, Dvs = [16], [32], [128], [128]

    variants = ["seq", "nax"]
    if do_fallback:
        variants = ["fallback"] + variants

    headers = ["B", "T", "Hk", "Hv", "Dk", "Dv"]
    headers += [f"{v} (ms)" for v in variants]
    headers += [f"{v} (speedup)" for v in variants[1:]]

    col_widths = [6, 6, 6, 6, 6, 6] + [16] * len(variants) + [16] * (len(variants) - 1)
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)

    mode = "BACKWARD" if do_backward else "FORWARD"
    print(f"\n=== {mode}: {' vs '.join(variants)} ===")
    print(fmt.format(*headers))
    print("-" * sum(col_widths))

    for B, T, Hk, Hv, Dk, Dv in itertools.product(Bs, Ts, Hks, Hvs, Dks, Dvs):
        try:
            times = benchmark_variants_shape(
                B, T, Hk, Hv, Dk, Dv, do_backward, variants
            )
        except Exception as ex:
            print(f"  B={B} T={T} failed: {ex}")
            mx.clear_cache()
            continue

        base = times[0]
        row = [f"{B}", f"{T}", f"{Hk}", f"{Hv}", f"{Dk}", f"{Dv}"]
        row += [f"{t:.3f}" for t in times]
        row += [f"{base / t:.2f}x" if t > 0 else "nan" for t in times[1:]]
        print(fmt.format(*row))
        print(RESET, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gated delta benchmark")
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument("--csv", "-c", action="store_true")
    parser.add_argument("--csv_out", "-co", default="benchmark_results.csv")
    parser.add_argument("--fallback", "-fb", action="store_true")
    parser.add_argument("--backward", "-bw", action="store_true")
    args = parser.parse_args()

    if args.backward or args.fallback:
        run_variants_benchmark(
            args.full, do_backward=args.backward, do_fallback=args.fallback
        )
    else:
        run_benchmark(args.full, to_csv=args.csv, csv_path=args.csv_out)
