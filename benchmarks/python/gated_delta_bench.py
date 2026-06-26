import argparse
import itertools
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import mlx.core as mx

N_warmup = 5
N_iter_bench = 40
N_iter_func = 8

os.environ["MTL_CAPTURE_ENABLED"] = "1"

RED_BOLD = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


def bench(f, *args):
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def do_kernel_bench(f, *args):
    q_out = args[0]

    for i in range(N_iter_func):
        out, hf = f(*args)

    mx.eval(out, hf)
    return q_out


def profile(f, *args):
    C = args[-1]
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M")
    trace_file = f"traces/mlx_trace_{C}_{timestamp}.gputrace"

    mx.metal.start_capture(trace_file)

    for i in range(N_iter_func):
        f(*args)

    mx.metal.stop_capture()
    print(f"Writing trace: {trace_file}")


def gated_delta_ref(
    q: mx.array,  # [B, T, Hk, Dk]
    k: mx.array,  # [B, T, Hk, Dk]
    v: mx.array,  # [B, T, Hv, Dv]
    g: mx.array,  # [B, T, Hv] or [B, T, Hv, Dk]
    beta: mx.array,  # [B, T, Hv]
    state: Optional[mx.array] = None,  # [B, Hv, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """
    Implements:
        S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T
        o_t = S_t q_t
    """
    B, T, Hk, Dk = q.shape
    Dv = v.shape[-1]
    Hv = v.shape[-2]

    if state is None:
        state = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    outputs = []
    for t in range(T):
        q_t = q[:, t]  # [B, H, Dk]
        k_t = k[:, t]  # [B, H, Dk]
        v_t = v[:, t]  # [B, H, Dv]
        g_t = g[:, t]  # [B, H] or [B, H, Dk]
        beta_t = beta[:, t]  # [B, H]

        # decay
        if g_t.ndim == 2:
            decay = g_t[..., None, None]  # [B, H, 1, 1]
        else:
            decay = g_t[..., None, :]  # [B, H, 1, Dk]

        # S = a S
        state = state * decay

        # kv = S * k_t, [B, H, Dv, Dk] * [B, H, 1, Dk] = [B, H, Dv, Dk] -> reduction on Dk
        kv_mem = (state * k_t[..., None, :]).sum(axis=-1)  # [B, H, Dv]

        # delta = b_t * (v_t - kv)
        delta = (v_t - kv_mem) * beta_t[..., None]  # [B, H, Dv]

        # S = S + delta * k_t^T, [B, H, Dv, 1] * [B, H, 1, Dk] = [B, H, Dv, Dk]
        state = state + delta[..., None] * k_t[..., None, :]  # [B, H, Dv, Dk]

        # o_t = S_t * q_t, [B, H, Dv, Dk] * [B, H, 1, Dk] = [B, H, Dv, Dk] -> reduction on Dk
        o_t = (state * q_t[..., None, :]).sum(axis=-1)  # [B, H, Dv]
        outputs.append(o_t)

    return mx.stack(outputs, axis=1), state  # [B, T, H, Dv], [B, H, Dv, Dk]


def benchmark_shape(B, T, Hk, Hv, Dk, Dv, chunk_sizes):
    mx.random.seed(42)

    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.normal(shape=(B, T, Hv)) * 0.1 - 1.0
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    shape_str = f"B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}"

    out_ref, hf_ref = gated_delta_ref(q, k, v, g, b, state=h0)
    mx.eval(out_ref, hf_ref)

    out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0, C=0)
    mx.eval(out, hf)

    atol = 1e-1
    out_close = mx.all(mx.abs(out - out_ref) < atol).item()
    hf_close = mx.all(mx.abs(hf - hf_ref) < atol).item()

    out_max_diff = mx.abs(out - out_ref).max().item()
    hf_max_diff = mx.abs(hf - hf_ref).max().item()

    # assert out_close, f"output mismatch! max diff: {out_max_diff:.2e}"
    # assert hf_close, f"state mismatch!  max diff: {hf_max_diff:.2e}"

    # for b in range(B):
    #     diff = mx.abs(out[b] - out_ref[b]).max().item()
    #     print(f"batch {b}: {diff:.2e}")
    # exit()
    if not out_close or not hf_close:
        print(f"{RED_BOLD}", end="")

    ms_seq = (
        bench(do_kernel_bench, mx.fast.gated_delta_update, q, k, v, g, b, h0, 0) * 1000
    )

    speedups = []

    non_zero_Cs = [C for C in chunk_sizes if C != 0]
    for C in non_zero_Cs:
        try:
            h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
            out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0, C=C)
            mx.eval(out, hf)

            err_out = mx.abs(out - out_ref).max().item()
            err_hf = mx.abs(hf - hf_ref).max().item()

            ms_c = (
                bench(
                    do_kernel_bench,
                    mx.fast.gated_delta_update,
                    q,
                    k,
                    v,
                    g,
                    b,
                    h0,
                    C,
                )
                * 1000
            )

            speedup = ms_seq / ms_c if ms_c > 0 else float("nan")

            out_close = mx.all(mx.abs(out - out_ref) < atol).item()
            hf_close = mx.all(mx.abs(hf - hf_ref) < atol).item()

            out_max_diff = mx.abs(out - out_ref).max().item()
            hf_max_diff = mx.abs(hf - hf_ref).max().item()

            if not out_close or not hf_close:
                raise Exception

            speedups.append(speedup)

        except Exception as e:
            speedups.append(-1)

    return shape_str, f"{ms_seq:.3f}", speedups


def run_benchmark(run_full):
    if run_full:
        Bs = [1, 2, 4, 8, 16]
        Ts = [2048, 4096, 8192, 16384]
        Hs = [16, 24, 32]
        Dks = [128]
        Dvs = [128]
    else:
        Bs = [1, 8, 16]
        Ts = [10, 512, 1024, 2048]
        # Hs = [16]
        Hks = [16]
        Hvs = [32]
        Dks = [128]
        Dvs = [128]

    rows = []

    chunk_sizes = [0, 8]
    non_zero_Cs = [C for C in chunk_sizes if C != 0]

    headers = ["B", "T", "Hk", "Hv", "Dk", "Dv", "time_seq (ms)"] + [
        f"C={C} speedup" for C in non_zero_Cs
    ]

    col_widths = [6, 6, 6, 6, 6, 6, 15] + [15] * (len(non_zero_Cs))
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths)))

    for B, T, Hk, Hv, Dk, Dv in itertools.product(Bs, Ts, Hks, Hvs, Dks, Dvs):
        shapes_s, base_time, speedups = benchmark_shape(
            B, T, Hk, Hv, Dk, Dv, chunk_sizes
        )
        row = [f"{B}", f"{T}", f"{Hk}", f"{Hv}", f"{Dk}", f"{Dv}", base_time]
        for speed in speedups:
            row.append(f"{speed:.2f}x")

        print(fmt.format(*row), end="")
        print(f"{RESET}")


def run_profile():
    B = 1
    Hk = 16
    Hv = 32
    T = 512
    Dk = 128
    Dv = 128
    CS = [8]  # , 16]#, 32]

    mx.random.seed(42)

    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.normal(shape=(B, T, Hv)) * 0.1 - 1.0
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    mx.eval(q, k, v, g, b, h0)

    out_ref, hf_ref = gated_delta_ref(q, k, v, g, b, state=h0)
    mx.eval(out_ref, hf_ref)

    for C in CS:
        out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0, C=C)
        mx.eval(out, hf)

        assert list(out.shape) == [B, T, Hv, Dv]
        assert list(hf.shape) == [B, Hv, Dv, Dk]
        assert not mx.any(mx.isnan(out)).item(), "NaNs in output!"
        assert not mx.any(mx.isnan(hf)).item(), "NaNs in final state!"

        # correctness check
        atol = 1e-2
        out_close = mx.all(mx.abs(out - out_ref) < atol).item()
        hf_close = mx.all(mx.abs(hf - hf_ref) < atol).item()

        out_max_diff = mx.abs(out - out_ref).max().item()
        hf_max_diff = mx.abs(hf - hf_ref).max().item()

        assert out_close, f"output mismatch! max diff: {out_max_diff:.2e}"
        assert hf_close, f"state mismatch!  max diff: {hf_max_diff:.2e}"

        profile(do_kernel_bench, mx.fast.gated_delta_update, q, k, v, g, b, h0, C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gated delta chunk kernel runner")
    parser.add_argument("--profile", "-p", action="store_true")
    parser.add_argument("--full", "-f", action="store_true")
    args = parser.parse_args()

    if args.profile:
        run_profile()
        exit()

    run_benchmark(args.full)
