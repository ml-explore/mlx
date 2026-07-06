# Copyright © 2026 Apple Inc.

"""Benchmark Qwen3-0.6B bf16 and quantized generation throughput.

Example:
  python benchmarks/python/qwen3_quantized_generate_bench.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

try:
    from mlx_lm import load as lm_load
    from mlx_lm.generate import stream_generate as lm_stream_generate
except Exception:  # pragma: no cover
    lm_load = None
    lm_stream_generate = None

try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm import stream_generate as vlm_stream_generate
except Exception:  # pragma: no cover
    vlm_load = None
    vlm_stream_generate = None

if lm_load is None and vlm_load is None:  # pragma: no cover
    raise RuntimeError(
        "No generation backend available. Install mlx-lm and/or mlx-vlm."
    )


DEFAULT_MODELS = (
    "mlx-community/Qwen3-0.6B-bf16",
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-8bit",
)

DEFAULT_PROMPT = "Explain matrix multiplication in one short paragraph."


@dataclass
class RunStats:
    wall_s: float
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float


def greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _is_likely_vision_model(model_id: str) -> bool:
    model_id = model_id.lower()
    return any(
        token in model_id
        for token in (
            "qwen3.5",
            "vision",
            "multimodal",
            "llava",
            "internvl",
            "gemma3",
        )
    )


def _looks_like_vision_weight_mismatch(exc: Exception) -> bool:
    message = str(exc).lower()
    return "vision_tower" in message or (
        "parameters not in model" in message and "vision" in message
    )


def load_with_backend(
    model_id: str,
) -> tuple[object, object, Callable[..., object], str]:
    if _is_likely_vision_model(model_id) and vlm_load is not None:
        model, processor = vlm_load(model_id)
        return model, processor, vlm_stream_generate, "mlx_vlm"

    if lm_load is not None:
        try:
            model, tokenizer = lm_load(model_id)
            return model, tokenizer, lm_stream_generate, "mlx_lm"
        except Exception as exc:
            if vlm_load is not None and _looks_like_vision_weight_mismatch(exc):
                model, processor = vlm_load(model_id)
                return model, processor, vlm_stream_generate, "mlx_vlm"
            raise

    if vlm_load is not None:
        model, processor = vlm_load(model_id)
        return model, processor, vlm_stream_generate, "mlx_vlm"

    raise RuntimeError("Unable to load model with mlx-lm or mlx-vlm.")


def run_once(
    model,
    processor,
    stream_fn: Callable[..., object],
    prompt: str,
    max_tokens: int,
) -> RunStats:
    start = time.perf_counter()
    final = None
    for response in stream_fn(
        model,
        processor,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
    ):
        final = response
    wall_s = time.perf_counter() - start

    if final is None:
        raise RuntimeError("Generation produced no output.")

    return RunStats(
        wall_s=wall_s,
        prompt_tokens=final.prompt_tokens,
        prompt_tps=final.prompt_tps,
        generation_tokens=final.generation_tokens,
        generation_tps=final.generation_tps,
    )


def summarize(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model ids to benchmark.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs before timed runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Timed runs per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before each run.",
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="MLX device to run on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(device)

    print(f"device={args.device} max_tokens={args.max_tokens} runs={args.runs}")
    print(f"prompt={args.prompt!r}")
    print()

    for model_id in args.models:
        print(f"=== {model_id} ===")

        load_start = time.perf_counter()
        model, processor, stream_fn, backend = load_with_backend(model_id)
        load_s = time.perf_counter() - load_start
        print(f"load_s={load_s:.3f} backend={backend}")

        for _ in range(args.warmup_runs):
            mx.random.seed(args.seed)
            _ = run_once(model, processor, stream_fn, args.prompt, args.max_tokens)

        runs: list[RunStats] = []
        for run_idx in range(args.runs):
            mx.random.seed(args.seed + run_idx)
            runs.append(
                run_once(model, processor, stream_fn, args.prompt, args.max_tokens)
            )

        wall_mean, wall_std = summarize([r.wall_s for r in runs])
        gen_tps_mean, gen_tps_std = summarize([r.generation_tps for r in runs])
        prompt_tps_mean, prompt_tps_std = summarize([r.prompt_tps for r in runs])
        eff_gen_tps_mean, eff_gen_tps_std = summarize(
            [r.generation_tokens / r.wall_s for r in runs]
        )

        print(
            "prompt_tokens={} generation_tokens={}".format(
                runs[-1].prompt_tokens,
                runs[-1].generation_tokens,
            )
        )
        print(
            "prompt_tps_mean={:.2f} prompt_tps_std={:.2f}".format(
                prompt_tps_mean,
                prompt_tps_std,
            )
        )
        print(
            "generation_tps_mean={:.2f} generation_tps_std={:.2f}".format(
                gen_tps_mean,
                gen_tps_std,
            )
        )
        print(
            "effective_gen_tps_mean={:.2f} effective_gen_tps_std={:.2f}".format(
                eff_gen_tps_mean,
                eff_gen_tps_std,
            )
        )
        print("wall_s_mean={:.3f} wall_s_std={:.3f}".format(wall_mean, wall_std))
        print()

        del model
        del processor
        mx.clear_cache()


if __name__ == "__main__":
    main()
