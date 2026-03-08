# Copyright © 2026 Apple Inc.

import argparse
import time

import numpy as np

# Keep torch imported before mlx in this benchmark process.
# If mlx imports first, mixed CUDA runtime loading can trigger Torch fp16/bf16
# GEMM failures (CUBLAS_STATUS_INVALID_VALUE) on some setups.
# isort: off
import torch
import mlx.core as mx

# isort: on

MLX_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_cases(cases):
    parsed = []
    for spec in cases.split(","):
        m, n, k, s = [int(x) for x in spec.split("x")]
        parsed.append((m, n, k, s))
    return parsed


def make_segments(k, num_segments, pattern, seed):
    if pattern == "equal":
        cuts = np.linspace(0, k, num_segments + 1, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        cuts = rng.integers(0, k + 1, size=(num_segments - 1,), dtype=np.int64)
        cuts = np.sort(cuts)
        cuts = np.concatenate(([0], cuts, [k]))
    return np.stack([cuts[:-1], cuts[1:]], axis=1).astype(np.uint32)


def mlx_segmented_mm_ref(a, b, segments):
    segments_list = segments.tolist()
    out = []
    for start, end in segments_list:
        out.append(a[:, start:end] @ b[start:end, :])
    return mx.stack(out, axis=0)


@torch.no_grad()
def torch_segmented_mm(a, b, segments):
    num_segments = segments.shape[0]
    m = a.shape[0]
    n = b.shape[1]
    out = torch.zeros((num_segments, m, n), device=a.device, dtype=a.dtype)
    for i in range(num_segments):
        start = int(segments[i, 0].item())
        end = int(segments[i, 1].item())
        if end > start:
            out[i] = a[:, start:end] @ b[start:end, :]
    return out


def sync_torch(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_mlx(a, b, segments, warmup, iters):
    for _ in range(warmup):
        y = mx.segmented_mm(a, b, segments)
        mx.eval(y)
    mx.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = mx.segmented_mm(a, b, segments)
        mx.eval(y)
    mx.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e3 / iters


def bench_torch(a, b, segments, warmup, iters, device):
    for _ in range(warmup):
        _ = torch_segmented_mm(a, b, segments)
    sync_torch(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = torch_segmented_mm(a, b, segments)
    sync_torch(device)
    end = time.perf_counter()
    return (end - start) * 1e3 / iters


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return (
            "| "
            + " | ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row))
            + " |"
        )

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=(
            "128x128x1024x16,"
            "128x128x1024x32,"
            "256x256x2048x16,"
            "512x512x4096x32,"
            "1024x1024x4096x32,"
            "1024x1024x8192x64"
        ),
        help="Comma-separated MxNxKxS list.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--segments",
        choices=["equal", "random"],
        default="random",
        help="Segment generation pattern.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    torch_device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    mlx_dtype = MLX_DTYPES[args.dtype]
    torch_dtype = TORCH_DTYPES[args.dtype]

    print(
        f"device={torch_device.type} dtype={args.dtype} warmup={args.warmup} iters={args.iters} segments={args.segments}"
    )

    headers = [
        "Case",
        "MLX ms",
        "Torch ms",
        "Torch/MLX",
        "MLX ref max abs",
        "MLX ref ok (1e-4)",
    ]
    rows = []

    cases = parse_cases(args.cases)
    for idx, (m, n, k, s) in enumerate(cases):
        rng = np.random.default_rng(args.seed + idx)
        a_np = rng.standard_normal((m, k), dtype=np.float32)
        b_np = rng.standard_normal((k, n), dtype=np.float32)
        seg_np = make_segments(k, s, args.segments, args.seed + idx)

        a_mx = mx.array(a_np, dtype=mlx_dtype)
        b_mx = mx.array(b_np, dtype=mlx_dtype)
        seg_mx = mx.array(seg_np, dtype=mx.uint32)
        mx.eval(a_mx, b_mx, seg_mx)

        a_torch = torch.tensor(a_np, dtype=torch_dtype, device=torch_device)
        b_torch = torch.tensor(b_np, dtype=torch_dtype, device=torch_device)
        seg_torch = torch.tensor(seg_np, dtype=torch.int64, device=torch_device)
        sync_torch(torch_device)

        verify_mlx_ref = True
        mlx_ref_max_abs = float("nan")
        if not args.no_check:
            y_mx = mx.segmented_mm(a_mx, b_mx, seg_mx)
            y_ref = mlx_segmented_mm_ref(a_mx, b_mx, seg_mx)
            mx.eval(y_mx, y_ref)
            verify_mlx_ref = bool(mx.allclose(y_ref, y_mx, atol=1e-4).item())
            mlx_ref_max_abs = float(mx.max(mx.abs(y_ref - y_mx)).item())
            if not verify_mlx_ref:
                raise RuntimeError(
                    f"MLX reference check failed for case {m}x{n}x{k}x{s}: max_abs={mlx_ref_max_abs:.6e}"
                )

        t_mlx = bench_mlx(a_mx, b_mx, seg_mx, args.warmup, args.iters)
        t_torch = bench_torch(
            a_torch, b_torch, seg_torch, args.warmup, args.iters, torch_device
        )
        ratio = t_torch / t_mlx if t_mlx > 0 else float("inf")
        case_name = f"{m}x{n}x{k}x{s}"
        verify_mark = "✓" if verify_mlx_ref else "✗"
        rows.append(
            [
                case_name,
                f"{t_mlx:.6f}",
                f"{t_torch:.6f}",
                f"{ratio:.4f}",
                f"{mlx_ref_max_abs:.6e}",
                verify_mark,
            ]
        )

    print_table(headers, rows)


if __name__ == "__main__":
    main()
