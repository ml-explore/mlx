# Copyright © 2026 Apple Inc.

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

# https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json
# https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json
CONFIGS = {
    "glm-5.3-flash": (4096, 2048, 288, 8),
    "glm-5.3": (6144, 2048, 256, 8),
    "qwen3.5-35b-a3b": (2048, 512, 256, 8),
    "qwen3.5-122b-a10b": (3072, 1024, 256, 8),
    "qwen3.5-397b-a17b": (4096, 1024, 512, 10),
}


def make_weights(experts, out_dims, in_dims):
    parts = []
    for start in range(0, experts, 16):
        part = (
            mx.random.normal(
                (min(16, experts - start), out_dims, in_dims), dtype=mx.bfloat16
            )
            * 0.02
        )
        mx.eval(part)
        parts.append(part)
    weights = mx.concatenate(parts, axis=0).swapaxes(-1, -2)
    mx.eval(weights)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIGS, default="glm-5.3-flash")
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(*args.seq_lens, args.iterations) <= 0:
        parser.error("Sequence lengths and iterations must be positive")

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    mx.random.seed(42)
    dims, hidden, experts, top_k = CONFIGS[args.model]
    print(
        f"model={args.model} D={dims} H={hidden} E={experts} top_k={top_k} "
        f"batch=1 dtype=bfloat16 device={mx.device_info()['device_name']}",
        flush=True,
    )
    wg = make_weights(experts, hidden, dims)
    wu = make_weights(experts, hidden, dims)
    wd = make_weights(experts, dims, hidden)
    print("Weights ready", flush=True)

    for seq_len in args.seq_lens:
        x = mx.random.normal((seq_len, dims), dtype=mx.bfloat16)
        scores = mx.random.uniform(shape=(seq_len, experts))
        routes = mx.argpartition(scores, experts - top_k, axis=-1)[:, -top_k:].flatten()
        order = mx.argsort(routes)
        indices = routes[order].astype(mx.uint32)
        xs = x[order // top_k]
        xs3 = xs[:, None, :]
        expert_ids = mx.arange(experts, dtype=mx.uint32)
        mx.eval(xs, xs3, indices, expert_ids)

        def gather():
            up = mx.gather_mm(xs3, wu, rhs_indices=indices, sorted_indices=True)
            gate = mx.gather_mm(xs3, wg, rhs_indices=indices, sorted_indices=True)
            return mx.gather_mm(
                nn.silu(gate) * up, wd, rhs_indices=indices, sorted_indices=True
            )

        def grouped():
            offsets = mx.searchsorted(indices, expert_ids).astype(mx.int32)
            up = mx.grouped_mm(xs, wu, token_offsets=offsets)
            gate = mx.grouped_mm(xs, wg, token_offsets=offsets)
            return mx.grouped_mm(nn.silu(gate) * up, wd, token_offsets=offsets)

        ref = gather().squeeze(-2).astype(mx.float32)
        actual = grouped().astype(mx.float32)
        max_abs_diff = mx.max(mx.abs(ref - actual)).item()
        if not mx.allclose(ref, actual, rtol=1e-2, atol=1e-2).item():
            raise AssertionError(f"MLP outputs differ for sequence length {seq_len}")
        del ref, actual

        functions = {"gather": gather, "grouped": grouped}
        for _ in range(5):
            for fn in functions.values():
                mx.eval(fn())
        times = {name: [] for name in functions}
        for iteration in range(args.iterations):
            names = list(functions)
            if iteration % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                mx.eval(functions[name]())
                times[name].append((time.perf_counter() - start) * 1000)
        gather_ms = statistics.median(times["gather"])
        grouped_ms = statistics.median(times["grouped"])
        print(
            f"S={seq_len} M={seq_len * top_k} "
            f"gather_ms={gather_ms:.4f} grouped_ms={grouped_ms:.4f} "
            f"speedup={gather_ms / grouped_ms:.3f} "
            f"gather_min_ms={min(times['gather']):.4f} "
            f"grouped_min_ms={min(times['grouped']):.4f} "
            f"max_abs_diff={max_abs_diff:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
