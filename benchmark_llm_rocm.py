#!/usr/bin/env python3

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "glm_4_7_flash_bf16": {
        "mlx_repo": "mlx-community/GLM-4.7-Flash-bf16",
        "llama_hf": "unsloth/GLM-4.7-Flash-GGUF:BF16",
    },
    "glm_4_7_flash_8bit": {
        "mlx_repo": "mlx-community/GLM-4.7-Flash-8bit",
        "llama_hf": "unsloth/GLM-4.7-Flash-GGUF:Q8_0",
    },
    "qwen3_0_6b_bf16": {
        "mlx_repo": "mlx-community/Qwen3-0.6B-bf16",
        "llama_hf": "unsloth/Qwen3-0.6B-GGUF:BF16",
    },
    "qwen3_0_6b_8bit": {
        "mlx_repo": "mlx-community/Qwen3-0.6B-8bit",
        "llama_hf": "unsloth/Qwen3-0.6B-GGUF:Q8_0",
    },
    "qwen3_coder_next_4bit": {
        "mlx_repo": "mlx-community/Qwen3-Coder-Next-4bit",
        "llama_hf": "unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M",
    },
}

DEFAULT_PROMPT = """
You are a coding assistant with deep expertise in GPU programming, machine learning systems, and performance optimization.

Explain, in plain English, how a GPU inference benchmark should be designed to fairly compare two runtimes (such as MLX vs llama.cpp). Provide a comprehensive analysis covering the following aspects:

1. Prompt Length Considerations:
   - Why varying prompt lengths (short, medium, long) reveal different performance characteristics
   - How prompt length affects memory bandwidth utilization vs compute utilization
   - The relationship between prompt length and KV cache behavior
   - Recommended prompt lengths for realistic benchmarks (128, 512, 1024, 2048 tokens)

2. Decode Length Impact:
   - How generation length affects time-to-first-token vs sustained throughput
   - Why short decodes may not represent real-world usage
   - The effect of decode length on memory allocation patterns
   - Recommendations for decode lengths to test (64, 128, 256, 512 tokens)

3. Sampling Settings:
   - Why temperature, top-k, top-p, and min-p settings affect benchmark consistency
   - The trade-off between deterministic (greedy) and stochastic sampling
   - How to choose sampling settings for fair comparisons
   - The impact of different sampling strategies on kernel utilization

4. Warmup Considerations:
   - Why warmup runs are essential for accurate GPU benchmarks
   - How CUDA/ROCm kernel compilation affects first-run latency
   - Memory allocation warmup vs kernel warmup
   - Recommended warmup strategies (number of runs, timing)

5. Memory Pressure Testing:
   - How to test under realistic memory constraints
   - The effect of batch size on memory utilization
   - KV cache memory scaling with sequence length
   - Out-of-memory behavior and graceful degradation

6. Deterministic Seeds:
   - Why deterministic seeds are critical for reproducibility
   - How random seed affects sampling and therefore timing
   - Recommendations for seed management in benchmarks

7. Additional Considerations:
   - GPU temperature throttling and thermal equilibrium
   - Power management and clock frequency stability
   - Multi-GPU scaling considerations
   - Quantization format comparisons (BF16, FP16, INT8, INT4)

Keep the answer structured with clear sections and bullet points. Provide specific numerical recommendations where applicable.
"""


@dataclass
class RunStats:
    variant: str
    backend: str
    model: str
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    gen_tokens: int | None = None
    gen_tps: float | None = None
    peak_mem_gb: float | None = None
    error: str | None = None


def run_command(cmd: list[str]) -> str:
    # Redact prompt from printed command to reduce clutter
    printed_cmd = []
    skip_next = False
    for arg in cmd:
        if skip_next:
            printed_cmd.append("<prompt>")
            skip_next = False
        else:
            printed_cmd.append(arg)
            if arg == "--prompt":
                skip_next = True
    print(f"\n$ {shlex.join(printed_cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}\n{output}")
    return output


def parse_mlx_stats(output: str, variant: str, model: str) -> RunStats:
    stats = RunStats(variant=variant, backend="mlx", model=model)

    m = re.search(r"Prompt:\s*(\d+)\s*tokens,\s*([0-9.]+)\s*tokens-per-sec", output)
    if m:
        stats.prompt_tokens = int(m.group(1))
        stats.prompt_tps = float(m.group(2))

    m = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([0-9.]+)\s*tokens-per-sec", output)
    if m:
        stats.gen_tokens = int(m.group(1))
        stats.gen_tps = float(m.group(2))

    m = re.search(r"Peak memory:\s*([0-9.]+)\s*GB", output)
    if m:
        stats.peak_mem_gb = float(m.group(1))

    return stats


def maybe_fmt_float(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def maybe_fmt_int(v: int | None) -> str:
    if v is None:
        return "n/a"
    return str(v)


def parse_int_token_count(s: str) -> int:
    return int(s.replace(",", ""))


def parse_tps_value(s: str) -> float | None:
    if s.lower() == "inf":
        return None
    return float(s)


def parse_llama_cli_stats(output: str, variant: str, model: str) -> RunStats:
    stats = RunStats(variant=variant, backend="llama", model=model)

    # Typical llama.cpp timing format examples:
    # common_perf_print: prompt eval time = ... / 60 tokens (..., 332.12 tokens per second)
    # common_perf_print:        eval time = ... /  7 runs   (...,  46.40 tokens per second)
    prompt_re = re.compile(
        r"/\s*([0-9,]+)\s*tokens?\s*\(\s*[0-9.]+\s*ms per token,\s*([0-9.]+|inf)\s*(?:tok/s|tokens per second)",
        flags=re.IGNORECASE,
    )
    eval_re = re.compile(
        r"/\s*([0-9,]+)\s*(?:runs|tokens?)\s*\(\s*[0-9.]+\s*ms per token,\s*([0-9.]+|inf)\s*(?:tok/s|tokens per second)",
        flags=re.IGNORECASE,
    )

    for line in output.splitlines():
        low = line.lower()
        if "prompt eval time" in low:
            m = prompt_re.search(line)
            if m:
                stats.prompt_tokens = parse_int_token_count(m.group(1))
                stats.prompt_tps = parse_tps_value(m.group(2))
        elif "eval time" in low:
            m = eval_re.search(line)
            if m:
                stats.gen_tokens = parse_int_token_count(m.group(1))
                stats.gen_tps = parse_tps_value(m.group(2))

    # Fallback for interactive llama-cli output format:
    # [ Prompt: 84.9 t/s | Generation: 50.3 t/s ]
    if stats.prompt_tps is None or stats.gen_tps is None:
        m = re.search(
            r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s",
            output,
            flags=re.IGNORECASE,
        )
        if m:
            stats.prompt_tps = parse_tps_value(m.group(1))
            stats.gen_tps = parse_tps_value(m.group(2))

    return stats


def run_mlx(cfg: dict[str, str], variant: str, args: argparse.Namespace) -> RunStats:
    mlx_model = cfg["mlx_repo"]

    try:
        import time

        import mlx.core as mx

        try:
            import mlx_lm
            from mlx_lm.generate import stream_generate as lm_stream_generate
        except Exception:
            mlx_lm = None
            lm_stream_generate = None

        try:
            from mlx_vlm import load as vlm_load
            from mlx_vlm import stream_generate as vlm_stream_generate
        except Exception:
            vlm_load = None
            vlm_stream_generate = None

        if mlx_lm is None and vlm_load is None:
            raise RuntimeError(
                "No MLX generation backend available. Install mlx-lm and/or mlx-vlm."
            )

        def likely_vision_model(model_id: str) -> bool:
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

        def looks_like_vision_weight_mismatch(exc: Exception) -> bool:
            message = str(exc).lower()
            return "vision_tower" in message or (
                "parameters not in model" in message and "vision" in message
            )

        backend = "mlx_lm"
        stream_generate_fn = lm_stream_generate

        if likely_vision_model(mlx_model) and vlm_load is not None:
            backend = "mlx_vlm"
            stream_generate_fn = vlm_stream_generate
            print(f"  Loading MLX model ({backend}): {mlx_model}")
            model, processor = vlm_load(mlx_model)
        elif mlx_lm is not None:
            try:
                print(f"  Loading MLX model ({backend}): {mlx_model}")
                model, processor = mlx_lm.load(mlx_model)
            except Exception as exc:
                if vlm_load is None or not looks_like_vision_weight_mismatch(exc):
                    raise
                backend = "mlx_vlm"
                stream_generate_fn = vlm_stream_generate
                print(f"  Falling back to {backend} for: {mlx_model}")
                model, processor = vlm_load(mlx_model)
        else:
            backend = "mlx_vlm"
            stream_generate_fn = vlm_stream_generate
            print(f"  Loading MLX model ({backend}): {mlx_model}")
            model, processor = vlm_load(mlx_model)

        # Load model once
        # Warmup runs (model stays loaded, JIT compiles kernels)
        if args.warmup_runs > 0:
            print(f"  Warming up MLX ({args.warmup_runs} runs)...")
            for i in range(args.warmup_runs):
                _ = next(
                    stream_generate_fn(
                        model,
                        processor,
                        prompt=args.prompt,
                        max_tokens=1,
                        sampler=lambda x: mx.argmax(x, axis=-1),
                    )
                )
                mx.synchronize()

        # Timed run
        print(f"  Running timed generation...")

        # Use stream_generate to get accurate per-token timings in a single pass
        # This avoids running the prompt twice and eliminates tokenization overhead from the timing
        start_time = time.perf_counter()
        final_stats = None
        output_text = ""
        stream_kwargs = {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "sampler": lambda x: mx.argmax(x, axis=-1) if args.temp == 0 else None,
        }
        if backend == "mlx_vlm":
            stream_kwargs.update({"temp": args.temp, "top_p": args.top_p})

        for response in stream_generate_fn(model, processor, **stream_kwargs):
            output_text += response.text
            final_stats = response

        mx.synchronize()
        total_time = time.perf_counter() - start_time

        if final_stats is None:
            raise RuntimeError("Generation produced no output.")

        num_prompt_tokens = final_stats.prompt_tokens
        gen_tokens = final_stats.generation_tokens
        prompt_tps = final_stats.prompt_tps
        gen_tps = final_stats.generation_tps

        # Get peak memory
        peak_mem_gb = None
        try:
            peak_mem_gb = mx.metal.get_peak_memory() / (1024**3)
        except:
            try:
                peak_mem_gb = mx.gpu.get_peak_memory() / (1024**3)
            except:
                try:
                    peak_mem_gb = mx.get_peak_memory() / (1024**3)
                except:
                    pass

        if args.show_raw_output:
            print(f"  Output: {output_text[:200]}...")
            print(f"  Prompt: {num_prompt_tokens} tokens, {prompt_tps:.2f} tok/s")
            print(f"  Generation: {gen_tokens} tokens, {gen_tps:.2f} tok/s")

        return RunStats(
            variant=variant,
            backend="mlx",
            model=mlx_model,
            prompt_tokens=num_prompt_tokens,
            prompt_tps=prompt_tps,
            gen_tokens=gen_tokens,
            gen_tps=gen_tps,
            peak_mem_gb=peak_mem_gb,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return RunStats(
            variant=variant,
            backend="mlx",
            model=mlx_model,
            error=str(e),
        )


def run_llama_cli(
    cfg: dict[str, str], variant: str, args: argparse.Namespace
) -> RunStats:
    model_name = (
        cfg.get("gguf_path")
        or cfg.get("llama_hf")
        or (f"{cfg.get('gguf_repo', 'n/a')}:{cfg.get('gguf_filename', 'n/a')}")
    )

    cmd = [
        args.llama_cli_path,
        "--prompt",
        args.prompt,
        "--n-predict",
        str(args.max_tokens),
        "--temp",
        str(args.temp),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--seed",
        str(args.seed),
        "--ctx-size",
        str(args.llama_n_ctx),
        "--batch-size",
        str(args.llama_n_batch),
        "--gpu-layers",
        str(args.llama_n_gpu_layers),
        "--simple-io",
        "--no-mmap",
        "--no-display-prompt",
        "--no-conversation",
        "--perf",
        "-fa",
        "1",
    ]

    if args.llama_n_threads is not None:
        cmd.extend(["--threads", str(args.llama_n_threads)])

    gguf_path = cfg.get("gguf_path")
    if gguf_path:
        cmd.extend(["--model", gguf_path])
    elif cfg.get("llama_hf"):
        cmd.extend(["-hf", cfg["llama_hf"]])
    else:
        gguf_repo = cfg.get("gguf_repo")
        gguf_filename = cfg.get("gguf_filename")
        if not gguf_repo or not gguf_filename:
            return RunStats(
                variant=variant,
                backend="llama",
                model=model_name,
                error=(
                    "Variant must provide one of: gguf_path, llama_hf, or "
                    "(gguf_repo + gguf_filename) for llama-completion"
                ),
            )
        cmd.extend(["--hf-repo", gguf_repo, "--hf-file", gguf_filename])

    try:
        output = run_command(cmd)
        if args.show_raw_output:
            print(output)
        return parse_llama_cli_stats(output, variant=variant, model=model_name)
    except Exception as e:
        return RunStats(
            variant=variant,
            backend="llama",
            model=model_name,
            error=str(e),
        )


def format_row(cols: list[str], widths: list[int]) -> str:
    return " | ".join(col.ljust(width) for col, width in zip(cols, widths))


def print_results_table(results: list[RunStats]) -> None:
    headers = [
        "variant",
        "backend",
        "prompt_tok/s",
        "decode_tok/s",
        "prompt_tok",
        "gen_tok",
        "peak_gb",
        "status",
    ]

    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r.variant,
                r.backend,
                maybe_fmt_float(r.prompt_tps, 3),
                maybe_fmt_float(r.gen_tps, 3),
                maybe_fmt_int(r.prompt_tokens),
                maybe_fmt_int(r.gen_tokens),
                maybe_fmt_float(r.peak_mem_gb, 3),
                "ok" if r.error is None else "error",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    print("\n=== Benchmark results ===")
    print(format_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row, widths))


def print_results_table_compact(results: list[RunStats], variants: list[str]) -> None:
    backend_names = {"llama": "llama", "mlx": "mlx"}

    headers = [
        "variant",
        "backend",
        "prompt_tps",
        "decode_tps",
        "p_tok",
        "g_tok",
        "mem_gb",
        "status",
    ]
    rows: list[list[str]] = []

    for r in results:
        rows.append(
            [
                r.variant,
                backend_names.get(r.backend, r.backend),
                maybe_fmt_float(r.prompt_tps, 2),
                maybe_fmt_float(r.gen_tps, 2),
                maybe_fmt_int(r.prompt_tokens),
                maybe_fmt_int(r.gen_tokens),
                maybe_fmt_float(r.peak_mem_gb, 1),
                "ok" if r.error is None else "er",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    print("\n=== Results (compact) ===")
    print(format_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row, widths))


def print_comparison(
    results: list[RunStats], variants: list[str], compact: bool = False
) -> None:
    by_variant: dict[str, dict[str, RunStats]] = {}
    for r in results:
        by_variant.setdefault(r.variant, {})[r.backend] = r

    print("\n=== Decode ratio (MLX / llama-completion) ===")
    for variant in variants:
        mlx = by_variant.get(variant, {}).get("mlx")
        llama = by_variant.get(variant, {}).get("llama")
        label = variant
        if not mlx or not llama:
            print(f"- {label}: n/a")
            continue
        if mlx.error or llama.error:
            print(f"- {label}: n/a (one or both runs failed)")
            continue
        if not mlx.gen_tps or not llama.gen_tps:
            print(f"- {label}: n/a (missing decode stats)")
            continue
        ratio = mlx.gen_tps / llama.gen_tps
        if compact:
            print(
                f"- {label}: {ratio:.3f}x ({mlx.gen_tps:.2f}/{llama.gen_tps:.2f} tok/s)"
            )
        else:
            print(
                f"- {label}: {ratio:.3f}x "
                f"(mlx {mlx.gen_tps:.3f} tok/s vs llama {llama.gen_tps:.3f} tok/s)"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MLX generate CLI vs llama-completion across model variants."
        )
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1000)

    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs for MLX (default: 2). Use 0 to disable.",
    )

    parser.add_argument(
        "--variants",
        nargs="*",
        default=["all"],
        help="Variant keys from MODEL_VARIANTS. Use 'all' for every variant.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List variants and exit.",
    )

    parser.add_argument("--llama-n-ctx", type=int, default=8192)
    parser.add_argument("--llama-n-batch", type=int, default=2048)
    parser.add_argument("--llama-n-gpu-layers", type=int, default=-1)
    parser.add_argument("--llama-n-threads", type=int, default=None)
    parser.add_argument(
        "--llama-cli-path",
        default="llama-completion",
        help="Path to the llama-completion executable.",
    )

    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="Print raw MLX CLI output for each run.",
    )
    parser.add_argument(
        "--table-mode",
        choices=["compact", "full"],
        default="full",
        help="Table format: full (default) or compact.",
    )
    return parser.parse_args()


def resolve_variants(arg_variants: list[str]) -> list[str]:
    if len(arg_variants) == 1 and arg_variants[0] == "all":
        return list(MODEL_VARIANTS.keys())

    unknown = [v for v in arg_variants if v not in MODEL_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Known: {', '.join(MODEL_VARIANTS.keys())}"
        )
    return arg_variants


def list_variants() -> None:
    print("Available variants:")
    for key, cfg in MODEL_VARIANTS.items():
        mlx_repo = cfg.get("mlx_repo", "n/a")
        gguf = (
            cfg.get("gguf_path")
            or cfg.get("llama_hf")
            or (f"{cfg.get('gguf_repo', 'n/a')}:{cfg.get('gguf_filename', 'n/a')}")
        )
        print(f"- {key}")
        print(f"    mlx:  {mlx_repo}")
        print(f"    llama: {gguf}")


def main() -> int:
    args = parse_args()

    if args.list_variants:
        list_variants()
        return 0

    try:
        variants = resolve_variants(args.variants)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("Running benchmark with shared decode settings:")
    prompt_summary = args.prompt[:50] + "..." if len(args.prompt) > 50 else args.prompt
    print(f"- prompt: {prompt_summary!r} (total {len(args.prompt)} chars)")
    print(f"- max_tokens: {args.max_tokens}")
    print(
        f"- sampling: temp={args.temp}, top_k={args.top_k}, "
        f"top_p={args.top_p}, min_p={args.min_p}, seed={args.seed}"
    )
    print("- execution: strictly serial (no concurrent model loads)")
    print(f"- variants: {', '.join(variants)}")

    results: list[RunStats] = []
    for variant in variants:
        cfg = MODEL_VARIANTS[variant]
        print(f"\n--- Variant: {variant} ---")
        results.append(run_llama_cli(cfg, variant, args))
        results.append(run_mlx(cfg, variant, args))

    if args.table_mode == "compact":
        print_results_table_compact(results, variants)
    else:
        print_results_table(results)
    print_comparison(results, variants, compact=(args.table_mode == "compact"))

    errors = [r for r in results if r.error]
    if errors:
        print("\n=== Errors ===")
        for r in errors:
            print(f"- {r.variant} [{r.backend}]: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
