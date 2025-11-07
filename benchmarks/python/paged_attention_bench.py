# ABOUTME: Microbenchmark for mx.fast.paged_attention throughput.
# ABOUTME: Emits CSV metrics (stdout/file) across TG/block/vector sweeps.

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional

TG_ENV = "MLX_PAGED_ATTN_TG_SIZE"
VEC_ENV = "MLX_PAGED_ATTN_VEC_WIDTH"
PROFILE_ENV = "MLX_METAL_PROFILING"

os.environ.setdefault(PROFILE_ENV, "1")

try:
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover - allows parser tests without MLX build
    mx = None

try:
    # Ensure Python fallback registers mx.fast.paged_attention if the native op exists.
    import mlx.nn.paged_kv as _paged_kv  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional import for parser tests
    _paged_kv = None

HEADER = (
    "batch,q_heads,kv_heads,head_dim,block_size,max_blocks,active_blocks,repeats,"
    "dtype,tg_size,vec_width,tokens_per_sec,gpu_ms_last"
)


@dataclass
class SweepConfig:
    block_size: int
    tg_size: Optional[int]
    vec_width: Optional[int]


def build_parser():
    parser = argparse.ArgumentParser(description="Paged attention microbench")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        help="Optional sweep over multiple block sizes.",
    )
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--active-blocks", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument(
        "--tg-size",
        type=int,
        default=None,
        help="Override threads-per-head (e.g., 32/64/128)",
    )
    parser.add_argument(
        "--tg-sizes",
        type=int,
        nargs="+",
        help="Optional sweep across multiple TG overrides.",
    )
    parser.add_argument(
        "--vec-width",
        type=int,
        default=None,
        help="Override vector width (forwarded via MLX_PAGED_ATTN_VEC_WIDTH).",
    )
    parser.add_argument(
        "--vec-widths",
        type=int,
        nargs="+",
        help="Optional sweep across multiple vector widths.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path (stdout always prints).",
    )
    return parser


def expand_sweep_args(args) -> List[SweepConfig]:
    block_sizes = args.block_sizes or [args.block_size]
    tg_sizes = args.tg_sizes or [args.tg_size]
    vec_widths: List[Optional[int]]
    if args.vec_widths:
        vec_widths = list(args.vec_widths)
    elif args.vec_width is not None:
        vec_widths = [args.vec_width]
    else:
        vec_widths = [None]
    return [
        SweepConfig(block_size=b, tg_size=tg, vec_width=vw)
        for b in block_sizes
        for tg in tg_sizes
        for vw in vec_widths
    ]


def build_inputs(args, dtype):
    if mx is None:
        raise RuntimeError(
            "mlx.core is required to build inputs; install MLX or run inside the repo venv."
        )
    batch = args.batch
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    block_size = args.block_size
    max_blocks = args.max_blocks

    q = mx.random.normal((batch, q_heads, 1, head_dim), dtype=dtype)
    k_cache = mx.random.normal(
        (kv_heads, max_blocks, block_size, head_dim), dtype=dtype
    )
    v_cache = mx.random.normal(
        (kv_heads, max_blocks, block_size, head_dim), dtype=dtype
    )
    block_tables = mx.full((batch, max_blocks), -1, dtype=mx.int32)
    context_lens = mx.zeros((batch,), dtype=mx.int32)

    for b in range(batch):
        active_blocks = min(max_blocks, args.active_blocks)
        block_tables[b, :active_blocks] = mx.arange(active_blocks, dtype=mx.int32)
        context_lens[b] = active_blocks * block_size

    return q, k_cache, v_cache, block_tables, context_lens


def _set_env_override(name: str, value: Optional[int]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)


def run_once(base_args, dtype, sweep: SweepConfig):
    if mx is None:
        raise RuntimeError(
            "mlx.core is required to execute the benchmark; install MLX or run inside the repo venv."
        )
    args = argparse.Namespace(**vars(base_args))
    args.block_size = sweep.block_size
    args.tg_size = sweep.tg_size
    args.vec_width = sweep.vec_width

    _set_env_override(TG_ENV, sweep.tg_size)
    _set_env_override(VEC_ENV, sweep.vec_width)

    q, k_cache, v_cache, block_tables, context_lens = build_inputs(args, dtype)
    repeats = args.repeats
    total_tokens = repeats * args.batch

    if hasattr(mx.fast, "_paged_attention_prewarm"):
        prewarm_kwargs = {}
        if sweep.tg_size is not None:
            prewarm_kwargs["threads_per_head"] = sweep.tg_size
        if sweep.vec_width is not None:
            prewarm_kwargs["vec_width"] = sweep.vec_width
        mx.fast._paged_attention_prewarm(args.block_size, dtype, **prewarm_kwargs)

    start = time.time()
    last_gpu_ms = 0.0
    for _ in range(repeats):
        q_iter = mx.stop_gradient(q)
        k_iter = mx.stop_gradient(k_cache)
        v_iter = mx.stop_gradient(v_cache)
        block_tables_iter = mx.stop_gradient(block_tables)
        context_lens_iter = mx.stop_gradient(context_lens)
        out = mx.fast.paged_attention(
            q=q_iter,
            k_cache=k_iter,
            v_cache=v_iter,
            block_tables=block_tables_iter,
            context_lens=context_lens_iter,
            layer_idx=0,
            kv_head_mapping=None,
            rope_freqs=None,
            scale=None,
        )
        mx.eval(out)
        if hasattr(mx.fast, "_paged_attention_last_time_ms"):
            last_gpu_ms = mx.fast._paged_attention_last_time_ms()
    duration = time.time() - start
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0
    return tokens_per_sec, last_gpu_ms


def format_row(args, sweep: SweepConfig, tokens_per_sec, last_gpu_ms) -> str:
    tg_value = "" if sweep.tg_size is None else sweep.tg_size
    vec_value = "" if sweep.vec_width is None else sweep.vec_width
    return (
        f"{args.batch},{args.q_heads},{args.kv_heads},{args.head_dim},"
        f"{sweep.block_size},{args.max_blocks},{args.active_blocks},{args.repeats},"
        f"{args.dtype_name},{tg_value},{vec_value},{tokens_per_sec:.2f},"
        f"{last_gpu_ms:.3f}"
    )


def maybe_write_csv(path: Optional[str], rows: List[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(row + "\n")


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.dtype_name = args.dtype
    dtype = getattr(mx, args.dtype)
    sweeps = expand_sweep_args(args)
    print(HEADER)
    rows: List[str] = []
    for sweep in sweeps:
        tokens_per_sec, last_gpu_ms = run_once(args, dtype, sweep)
        row = format_row(args, sweep, tokens_per_sec, last_gpu_ms)
        print(row)
        rows.append(row)
    maybe_write_csv(args.csv, rows)
