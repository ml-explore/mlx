# Copyright © 2024-2025 Apple Inc.

"""Layer-streaming utilities for memory-constrained inference.

Enables running models that exceed available physical memory by
evaluating each transformer layer individually, allowing the OS
to page out idle layer weights between forward passes. Weights
remain mmap-backed from safetensors — only the active layer's
pages need to be resident.

For a 27B 4-bit model (15GB) on a 16GB Mac:
  Standard: 15GB active -> system thrashes or crashes
  Streaming: ~1GB active at a time -> runs slowly but works
"""

from __future__ import annotations

import gc
from typing import Any, Optional

import mlx.core as mx


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
):
    """Generate tokens with per-layer evaluation for low memory usage.

    Monkey-patches the model's inner transformer to insert mx.eval()
    after each layer, then restores the original. This is model-agnostic
    — it works with any mlx_lm model architecture.

    Args:
        model: An mlx_lm model.
        tokenizer: The model's tokenizer.
        prompt: Input text.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Yields:
        Generated token strings.
    """
    # Find the inner transformer that has .layers
    inner = _find_inner_model(model)
    if inner is None:
        raise AttributeError("Cannot find transformer layers in model")

    # Collect stop token ids
    hf_tok = getattr(tokenizer, '_tokenizer', tokenizer)
    stop_ids = set()
    eos = getattr(hf_tok, 'eos_token_id', None)
    if eos is not None:
        stop_ids.add(eos)
    # Add <end_of_turn> for Gemma models
    if hasattr(hf_tok, 'added_tokens_encoder'):
        for name, tid in hf_tok.added_tokens_encoder.items():
            if name in ('<end_of_turn>', '<|im_end|>', '<|eot_id|>'):
                stop_ids.add(tid)
    if not stop_ids:
        stop_ids.add(2)

    # Patch each layer to eval eagerly after its forward pass.
    # We wrap each layer's __call__ rather than replacing the layers list,
    # because MLX Module.layers is a read-only property.
    original_calls = {}
    for i, layer in enumerate(inner.layers):
        layer_type = type(layer)
        if layer_type not in original_calls:
            original_calls[layer_type] = layer_type.__call__

            def make_streaming_call(orig):
                def streaming_call(self, *args, **kwargs):
                    out = orig(self, *args, **kwargs)
                    mx.eval(out)
                    return out
                return streaming_call

            layer_type.__call__ = make_streaming_call(original_calls[layer_type])

    try:
        tokens = mx.array([tokenizer.encode(prompt)])
        cache = model.make_cache()

        for _ in range(max_tokens):
            logits = model(tokens, cache=cache)
            mx.eval(logits)

            if temperature <= 0:
                token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                scaled = logits[:, -1, :] / temperature
                token = mx.random.categorical(scaled)

            mx.eval(token)
            token_id = token.item()

            if token_id in stop_ids:
                break

            tokens = token.reshape(1, 1)
            yield hf_tok.decode([token_id])
    finally:
        # Restore original layer __call__ methods
        for layer_type, orig in original_calls.items():
            layer_type.__call__ = orig


def _find_inner_model(model):
    """Walk the model tree to find the inner transformer with .layers."""
    candidates = [model]
    for attr in ('language_model', 'model', 'transformer', 'gpt_neox'):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
            # One more level deep
            for attr2 in ('model', 'transformer'):
                obj2 = getattr(obj, attr2, None)
                if obj2 is not None:
                    candidates.append(obj2)

    for c in candidates:
        if hasattr(c, 'layers') and hasattr(c.layers, '__len__'):
            return c
    return None




def estimate_streaming_memory(
    model_size_gb: float,
    num_layers: int,
    kv_cache_per_layer_mb: float = 0,
    num_kv_layers: int = 0,
) -> dict:
    """Estimate memory for streaming vs standard inference.

    Args:
        model_size_gb: Total model size in GB.
        num_layers: Number of transformer layers.
        kv_cache_per_layer_mb: KV cache per layer in MB.
        num_kv_layers: Layers with KV cache (0 = all).

    Returns:
        Dict with standard_gb, streaming_gb, savings_pct.
    """
    if num_kv_layers == 0:
        num_kv_layers = num_layers

    layer_size_gb = model_size_gb / num_layers
    kv_total_gb = (kv_cache_per_layer_mb * num_kv_layers) / 1024
    embedding_overhead = model_size_gb * 0.10

    standard = model_size_gb + kv_total_gb
    streaming = layer_size_gb + embedding_overhead + kv_total_gb

    return {
        "standard_gb": round(standard, 1),
        "streaming_gb": round(streaming, 1),
        "savings_pct": round((1 - streaming / max(standard, 0.1)) * 100),
    }
