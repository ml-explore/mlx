"""Pytest-based generation checks for Qwen3, LFM2.5, and Qwen3-Coder-Next variants.

Run with:
  source venv/bin/activate
  pytest -s test_qwen3_generation.py

Environment overrides:
  MLX_TEST_PROMPT="Your deterministic prompt"
  MLX_TEST_SEED=42
  MLX_TEST_MAX_TOKENS=64
  MLX_TEST_DEVICE=gpu|cpu
  MLX_TEST_OUTPUT_DIR=/path/to/save/outputs
  MLX_TEST_REPEATABILITY=1   # rerun each model twice and compare text
"""

from __future__ import annotations

import itertools
import os
import re
import warnings
from pathlib import Path
from typing import Any, cast

# Suppress known third-party SWIG deprecation noise seen during model/tokenizer imports.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)

import mlx.core as mx
import pytest

try:
    from mlx_lm import load
    from mlx_lm.generate import generate
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"mlx_lm is required for this test file: {exc}", allow_module_level=True
    )


MODEL_FAMILIES = [
    "mlx-community/Qwen3-0.6B",
    "mlx-community/LFM2.5-1.2B-Instruct",
    "mlx-community/LFM2.5-1.2B-Thinking",
]
MODEL_VARIANTS = ["bf16", "3bit", "4bit", "6bit", "8bit"]
EXPLICIT_MODELS = [
    "mlx-community/Qwen3-Coder-Next-4bit",
]

# Fixed model list used as pytest cases.
MODELS = [
    f"{model_family}-{variant}"
    for model_family in MODEL_FAMILIES
    for variant in MODEL_VARIANTS
] + EXPLICIT_MODELS

DEFAULT_PROMPT = "Write exactly one short friendly greeting."
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 64
PROMPT = os.getenv("MLX_TEST_PROMPT", DEFAULT_PROMPT)
SEED = int(os.getenv("MLX_TEST_SEED", str(DEFAULT_SEED)))
MAX_TOKENS = int(os.getenv("MLX_TEST_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
DEVICE_NAME = os.getenv("MLX_TEST_DEVICE", "gpu").strip().lower()
OUTPUT_DIR_OVERRIDE = os.getenv("MLX_TEST_OUTPUT_DIR", "").strip()
REPEATABILITY_CHECK = os.getenv("MLX_TEST_REPEATABILITY", "0").strip() == "1"


if DEVICE_NAME not in {"gpu", "cpu"}:
    raise ValueError("MLX_TEST_DEVICE must be one of: gpu, cpu")
if not MODELS:
    raise ValueError("No models configured. Update the MODELS list.")


DEVICE = mx.gpu if DEVICE_NAME == "gpu" else mx.cpu


def _greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _case_id(model_id: str) -> str:
    return model_id.split("/")[-1]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def _text_stats(text: str) -> dict[str, float | int]:
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    word_count = len(words)
    unique_words = len(set(words))
    unique_word_ratio = unique_words / word_count if word_count else 0.0
    longest_char_run = max(
        (sum(1 for _ in group) for _, group in itertools.groupby(text)), default=0
    )
    return {
        "chars": len(text),
        "words": word_count,
        "unique_words": unique_words,
        "unique_word_ratio": unique_word_ratio,
        "longest_char_run": longest_char_run,
    }


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    stack = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        chain.append(current)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return tuple(chain)


def _is_404_error(exc: Exception) -> bool:
    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 404:
            return True
        if getattr(current, "status_code", None) == 404:
            return True
        message = str(current).lower()
        if "404" in message and any(
            token in message
            for token in (
                "not found",
                "does not exist",
                "could not find",
                "couldn't find",
            )
        ):
            return True
    return False


def _generate(model_id: str) -> str:
    mx.set_default_device(cast(Any, DEVICE))
    mx.random.seed(SEED)

    try:
        model, tokenizer, *_ = load(model_id)
    except Exception as exc:
        if _is_404_error(exc):
            pytest.skip(f"{model_id} is unavailable on the hub (404): {exc}")
        raise

    text = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        sampler=_greedy_sampler,
        verbose=False,
    )

    del model
    del tokenizer
    mx.clear_cache()
    return text


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if OUTPUT_DIR_OVERRIDE:
        path = Path(OUTPUT_DIR_OVERRIDE)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("generation_outputs")


@pytest.mark.parametrize("model_id", MODELS, ids=_case_id)
def test_generate_and_show_output(model_id: str, output_dir: Path) -> None:
    text = _generate(model_id)
    stats = _text_stats(text)

    output_path = output_dir / f"{_slug(model_id)}.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"\n=== MODEL: {model_id} ===")
    print(f"device={DEVICE_NAME} seed={SEED} max_tokens={MAX_TOKENS} prompt={PROMPT!r}")
    print(
        "stats: "
        f"chars={stats['chars']} "
        f"words={stats['words']} "
        f"unique_words={stats['unique_words']} "
        f"unique_word_ratio={stats['unique_word_ratio']:.3f} "
        f"longest_char_run={stats['longest_char_run']}"
    )
    print("--- output start ---")
    print(text)
    print("--- output end ---")
    print(f"saved: {output_path}")

    assert text.strip(), f"{model_id} generated empty output"


@pytest.mark.skipif(
    not REPEATABILITY_CHECK,
    reason="Set MLX_TEST_REPEATABILITY=1 to enforce exact repeatability.",
)
@pytest.mark.parametrize("model_id", MODELS, ids=_case_id)
def test_repeatability(model_id: str) -> None:
    first = _generate(model_id)
    second = _generate(model_id)
    assert first == second, (
        f"{model_id} is not repeatable with fixed seed={SEED}, prompt={PROMPT!r}, "
        f"device={DEVICE_NAME}."
    )
