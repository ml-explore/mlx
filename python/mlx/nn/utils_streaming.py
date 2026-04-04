# Copyright © 2024-2025 Apple Inc.

"""Layer-streaming utilities for memory-constrained inference.

Enables running models that exceed available physical memory by
evaluating each transformer layer individually, allowing the OS
to page out idle layer weights between forward passes. Weights
remain mmap-backed in safetensors — only the active layer's
pages need to be resident in physical RAM.

For a 70B 4-bit model (35GB) on a 16GB Mac:
  Standard load: all 35GB paged in → system thrashes and freezes
  Streaming:     ~1-2GB active at a time → runs slowly but doesn't crash
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
    """Generate tokens with layer-streaming to minimize memory.

    Each forward pass evaluates one layer at a time, allowing the
    OS to reclaim idle layer weights between evaluations. This is
    slower than standard generation but uses far less physical
    memory.

    Args:
        model: An mlx_lm model.
        tokenizer: The model's tokenizer.
        prompt: Input text.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Yields:
        Generated token strings, one at a time.
    """
    tokens = mx.array([tokenizer.encode(prompt)])
    cache = model.make_cache()

    for _ in range(max_tokens):
        logits = _stream_forward(model, tokens, cache)
        mx.eval(logits)

        if temperature <= 0:
            token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            token = mx.random.categorical(logits)

        mx.eval(token)
        token_id = token.item()

        if token_id == tokenizer.eos_token_id:
            break

        tokens = token.reshape(1, 1)
        yield tokenizer.decode([token_id])


def _stream_forward(model, inputs, cache=None):
    """Forward pass with per-layer evaluation for memory efficiency.

    Instead of building the full computation graph across all layers
    (which forces all layer weights into memory when evaluated), this
    evaluates after each layer, allowing the OS memory manager to
    page out weights that are no longer needed.
    """
    # Navigate to the inner transformer model
    # mlx_lm models: model -> model.language_model -> language_model.model
    lm = getattr(model, 'language_model', model)
    inner = getattr(lm, 'model', lm)

    # Embedding
    h = inner.embed_tokens(inputs)
    mx.eval(h)

    # Build masks (model-specific)
    if cache is None:
        cache_list = [None] * len(inner.layers)
    else:
        cache_list = cache

    mask = None
    if hasattr(inner, 'fa_idx'):
        # Qwen3.5-style: different masks for attention vs SSM layers
        from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(h, cache_list[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache_list[inner.ssm_idx])
        mx.eval(fa_mask, ssm_mask)

        for layer, c in zip(inner.layers, cache_list):
            layer_mask = ssm_mask if getattr(layer, 'is_linear', False) else fa_mask
            h = layer(h, mask=layer_mask, cache=c)
            mx.eval(h)
    else:
        # Standard transformer: single attention mask
        try:
            from mlx_lm.models.base import create_attention_mask
            mask = create_attention_mask(h, cache_list[0] if cache_list else None)
        except ImportError:
            mask = None

        if mask is not None:
            mx.eval(mask)

        for layer, c in zip(inner.layers, cache_list):
            if mask is not None:
                h = layer(h, mask=mask, cache=c)
            else:
                h = layer(h, cache=c)
            mx.eval(h)

    # Final norm
    h = inner.norm(h)
    mx.eval(h)

    # Output projection
    if hasattr(lm, 'lm_head'):
        logits = lm.lm_head(h)
    elif hasattr(inner, 'embed_tokens') and hasattr(inner.embed_tokens, 'as_linear'):
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = h

    return logits


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
        kv_cache_per_layer_mb: KV cache size per layer in MB.
        num_kv_layers: Number of layers with KV cache (0 = num_layers).

    Returns:
        Dict with standard_gb, streaming_gb, and savings_pct.
    """
    if num_kv_layers == 0:
        num_kv_layers = num_layers

    layer_size_gb = model_size_gb / num_layers
    kv_total_gb = (kv_cache_per_layer_mb * num_kv_layers) / 1024

    # Standard: all weights + KV cache in memory simultaneously
    standard = model_size_gb + kv_total_gb

    # Streaming: one layer active + embeddings/head (~10%) + full KV cache
    embedding_overhead = model_size_gb * 0.10
    streaming = layer_size_gb + embedding_overhead + kv_total_gb

    return {
        "standard_gb": round(standard, 1),
        "streaming_gb": round(streaming, 1),
        "savings_pct": round((1 - streaming / max(standard, 0.1)) * 100),
    }
