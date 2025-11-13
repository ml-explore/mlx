# ABOUTME: Provides paged key/value cache management utilities for attention.
# ABOUTME: Implements block allocation, reference counting, and copy-on-write.

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.core.fast as mx_fast

_NATIVE_PAGED_ATTENTION = getattr(mx_fast, "_paged_attention_impl", None)
_NATIVE_PAGED_KV_WRITE = getattr(mx_fast, "_paged_kv_write_impl", None)
_PAGED_ATTENTION_PREWARM = getattr(mx_fast, "_paged_attention_prewarm", None)
_NATIVE_PAGED_PREFILL = getattr(mx_fast, "_paged_prefill_impl", None)
_PAGED_PREFILL_PREWARM = getattr(mx_fast, "_paged_prefill_prewarm", None)
_NATIVE_PAGED_OVERLAY = getattr(mx_fast, "_paged_attention_with_overlay_impl", None)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() not in {"", "0", "false", "off", "no"}


if _env_flag("MLX_PAGED_DISABLE_NATIVE"):
    logging.warning(
        "MLX_PAGED_DISABLE_NATIVE set; forcing reference paged attention/prefill."
    )
    _NATIVE_PAGED_ATTENTION = None
    _NATIVE_PAGED_PREFILL = None
    _NATIVE_PAGED_OVERLAY = None

_SUPPORTED_QUANT_BITS = (4, 8)
_QUANT_TARGETS = frozenset({"k", "v"})


def _overlay_length_value(overlay_len, k_overlay) -> int:
    if overlay_len is not None:
        try:
            return int(overlay_len)
        except Exception:
            return int(mx.array(overlay_len, dtype=mx.int32))
    if k_overlay.ndim >= 4:
        return int(k_overlay.shape[0])
    return 1


def _paged_attention_overlay_fallback(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    *,
    layer_idx,
    kv_head_mapping=None,
    scale=None,
    k_overlay,
    v_overlay,
    overlay_len=None,
    v_q_cache=None,
    v_scale_cache=None,
    v_zero_cache=None,
    quant_bits=None,
    quant_group_size=None,
    quant_groups_per_head=None,
    quant_symmetric=None,
    stream=None,
):
    quant_enabled = quant_bits not in (None, 0)
    if quant_enabled:
        logging.debug(
            "paged_attention overlay fallback running in float mode despite quantized KV inputs"
        )
    overlay_tokens = _overlay_length_value(overlay_len, k_overlay)
    if overlay_tokens <= 0:
        return mx_fast.paged_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            layer_idx=layer_idx,
            kv_head_mapping=kv_head_mapping,
            scale=scale,
            v_q_cache=v_q_cache,
            v_scale_cache=v_scale_cache,
            v_zero_cache=v_zero_cache,
            quant_bits=quant_bits,
            quant_group_size=quant_group_size,
            quant_groups_per_head=quant_groups_per_head,
            quant_symmetric=quant_symmetric,
        )
    if k_overlay.ndim == 3:
        overlay_k = mx.expand_dims(k_overlay, axis=0)
        overlay_v = mx.expand_dims(v_overlay, axis=0)
    elif k_overlay.ndim == 4:
        overlay_k = k_overlay
        overlay_v = v_overlay
    else:
        raise ValueError("overlay tensors must be rank-3 or rank-4")
    overlay_tokens = min(overlay_tokens, int(overlay_k.shape[0]))
    batch, num_heads, _, head_dim = q.shape
    layer_idx = 0 if layer_idx is None else layer_idx
    layer_k = k_cache[layer_idx] if k_cache.ndim == 5 else k_cache
    layer_v = v_cache[layer_idx] if v_cache.ndim == 5 else v_cache
    num_kv_heads = int(layer_k.shape[0])
    block_size = int(layer_k.shape[2])
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if kv_head_mapping is not None:
        mapping = [int(x) for x in kv_head_mapping.tolist()]
    elif num_kv_heads == num_heads:
        mapping = list(range(num_heads))
    elif num_kv_heads == 1:
        mapping = [0 for _ in range(num_heads)]
    else:
        mapping = [(h * num_kv_heads) // num_heads for h in range(num_heads)]

    overlay_k = overlay_k[:overlay_tokens]
    overlay_v = overlay_v[:overlay_tokens]
    overlay_k_batch = mx.transpose(overlay_k, (1, 2, 0, 3))
    overlay_v_batch = mx.transpose(overlay_v, (1, 2, 0, 3))
    overlay_k_batch = overlay_k_batch.astype(layer_k.dtype)
    overlay_v_batch = overlay_v_batch.astype(layer_v.dtype)

    head_outputs: List[List[mx.array]] = [
        [mx.zeros((head_dim,), dtype=q.dtype) for _ in range(num_heads)]
        for _ in range(batch)
    ]
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        base_k, base_v = _materialize_sequence(
            seq_len, block_size, block_tables[b], layer_k, layer_v
        )
        extra_k = overlay_k_batch[b]
        extra_v = overlay_v_batch[b]
        k_seq = (
            base_k
            if extra_k.shape[1] == 0
            else mx.concatenate([base_k, extra_k], axis=1)
        )
        v_seq = (
            base_v
            if extra_v.shape[1] == 0
            else mx.concatenate([base_v, extra_v], axis=1)
        )
        total_tokens = int(k_seq.shape[1])
        if total_tokens == 0:
            continue
        for head_idx in range(num_heads):
            kv_head = mapping[head_idx]
            q_vec = q[b, head_idx, 0]
            k_head = k_seq[kv_head, :total_tokens]
            v_head = v_seq[kv_head, :total_tokens]
            scores = mx.sum(q_vec * k_head, axis=-1) * scale
            weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            context = mx.sum(mx.expand_dims(weights, axis=-1) * v_head, axis=0)
            head_outputs[b][head_idx] = context.astype(q.dtype)
    if not head_outputs:
        return q * 0
    per_batch = [mx.stack(heads, axis=0) for heads in head_outputs]
    stacked = mx.stack(per_batch, axis=0)
    return mx.expand_dims(stacked, axis=2)


if _NATIVE_PAGED_OVERLAY is None:
    logging.debug("Installing reference fallback for paged overlay attention")

    def _overlay_dispatch(*args, **kwargs):
        return _paged_attention_overlay_fallback(*args, **kwargs)

    mx_fast._paged_attention_with_overlay_impl = _overlay_dispatch
else:

    def _overlay_dispatch(*args, **kwargs):
        try:
            return _NATIVE_PAGED_OVERLAY(*args, **kwargs)
        except RuntimeError:
            logging.warning(
                "Native paged overlay attention failed; falling back to reference path",
                exc_info=True,
            )
            return _paged_attention_overlay_fallback(*args, **kwargs)

    mx_fast._paged_attention_with_overlay_impl = _overlay_dispatch


@dataclass(frozen=True)
class QuantSpec:
    """Describes on-device quantization for paged KV storage."""

    bits: int
    group_size: int
    targets: Tuple[str, ...] = ("v",)
    symmetric: bool = False
    storage_dtype: mx.Dtype = mx.uint8
    scale_dtype: mx.Dtype = mx.float16

    def __post_init__(self) -> None:
        if self.bits not in _SUPPORTED_QUANT_BITS:
            raise ValueError(f"bits must be one of {_SUPPORTED_QUANT_BITS}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        bad = [t for t in self.targets if t not in _QUANT_TARGETS]
        if bad:
            raise ValueError(f"unsupported quantization targets: {bad}")

    def requires(self, target: str) -> bool:
        return target in self.targets


@dataclass
class PagedKVQuantLayout:
    """Metadata describing quantized buffer layouts for paged attention."""

    bits: int
    group_size: int
    storage_dtype: mx.Dtype
    scale_dtype: mx.Dtype
    value_shape: Tuple[int, ...]
    scale_shape: Tuple[int, ...]
    zero_shape: Tuple[int, ...]
    bytes_per_token: int
    groups_per_head: int


def _bytes_per_token(head_dim: int, bits: int, group_size: int) -> int:
    padded_dim = _groups_per_head(head_dim, group_size) * group_size
    return math.ceil(padded_dim * bits / 8)


def _groups_per_head(head_dim: int, group_size: int) -> int:
    return math.ceil(head_dim / group_size)


class KVBlockManager:
    _POOL_COUNTER = 1

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks: int,
        dtype: mx.Dtype,
        kv_quantization: Optional[QuantSpec] = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype
        self.quant_spec = kv_quantization
        self._quant_layout: Optional[PagedKVQuantLayout] = None

        self.k = mx.zeros(
            (num_layers, num_kv_heads, max_blocks, block_size, head_dim), dtype=dtype
        )
        self.v = mx.zeros_like(self.k)
        self._pool_id = KVBlockManager._POOL_COUNTER
        KVBlockManager._POOL_COUNTER += 1
        self._manager_epoch = 0

        if _PAGED_ATTENTION_PREWARM is not None:
            try:
                _PAGED_ATTENTION_PREWARM(block_size, dtype)
            except RuntimeError:
                pass

        if _PAGED_PREFILL_PREWARM is not None:
            try:
                _PAGED_PREFILL_PREWARM(block_size, dtype)
            except RuntimeError:
                pass

        (
            self.v_q,
            self.v_scale,
            self.v_zero,
        ) = self._initialize_quant_buffers(kv_quantization)

        self._free_blocks: deque[int] = deque(range(max_blocks))
        self._ref_counts: List[int] = [0 for _ in range(max_blocks)]
        self._block_tables: Dict[int, List[int]] = {}
        self._context_lens: Dict[int, int] = {}
        self._staged_context_lens: Dict[int, int] = {}
        self._prefill_active: set[int] = set()

        self.max_blocks_per_sequence = max_blocks

    def new_sequence(self, seq_id: int, prompt_len: int) -> None:
        if seq_id in self._block_tables:
            raise ValueError(f"sequence {seq_id} already exists")

        needed = max(1, math.ceil(max(prompt_len, 0) / self.block_size))
        if needed > len(self._free_blocks):
            raise RuntimeError("insufficient blocks for new sequence")

        table = [-1 for _ in range(self.max_blocks_per_sequence)]
        for idx in range(needed):
            block_id = self._allocate_block()
            table[idx] = block_id
            self._ref_counts[block_id] = 1

        self._block_tables[seq_id] = table
        self._context_lens[seq_id] = prompt_len
        self._bump_manager_epoch()

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        if child_seq_id in self._block_tables:
            raise ValueError(f"sequence {child_seq_id} already exists")
        parent_table = self._block_tables.get(parent_seq_id)
        if parent_table is None:
            raise ValueError(f"parent sequence {parent_seq_id} missing")

        child_table = list(parent_table)
        for block_id in child_table:
            if block_id >= 0:
                self._ref_counts[block_id] += 1

        self._block_tables[child_seq_id] = child_table
        self._context_lens[child_seq_id] = self._context_lens[parent_seq_id]
        self._bump_manager_epoch()

    def write_prefill(
        self,
        seq_id: int,
        layer_idx: int,
        k_chunk: mx.array,
        v_chunk: mx.array,
        start_pos: int,
        *,
        commit: bool = True,
    ) -> None:
        self._write_tokens(
            seq_id, layer_idx, k_chunk, v_chunk, start_pos, commit=commit
        )

    def append_decode_token(
        self,
        seq_id: int,
        layer_idx: int,
        k_token: mx.array,
        v_token: mx.array,
    ) -> None:
        if k_token.ndim == 1:
            k_token = mx.expand_dims(k_token, 0)
        if k_token.ndim == 2:
            k_token = mx.expand_dims(k_token, 1)
        if v_token.ndim == 1:
            v_token = mx.expand_dims(v_token, 0)
        if v_token.ndim == 2:
            v_token = mx.expand_dims(v_token, 1)

        position = self._context_lens.get(seq_id, 0)
        self._write_tokens(seq_id, layer_idx, k_token, v_token, position)

    def _write_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        k_chunk: mx.array,
        v_chunk: mx.array,
        start_pos: int,
        *,
        commit: bool = True,
    ) -> None:
        tokens = int(k_chunk.shape[1])
        if tokens == 0:
            return

        end_pos = start_pos + tokens
        first_block = start_pos // self.block_size
        last_block = math.ceil(end_pos / self.block_size)
        for block_idx in range(first_block, last_block):
            self._ensure_block(seq_id, block_idx, copy_existing=True)

        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")
        block_row = mx.array(table, dtype=mx.int32)

        quant_enabled = self.v_q is not None and self.quant_spec is not None
        native_supported = _NATIVE_PAGED_KV_WRITE is not None and (
            not quant_enabled or self._quant_write_supported()
        )
        if native_supported:
            try:
                k_tokens = self._prepare_chunk(k_chunk)
                v_tokens = self._prepare_chunk(v_chunk)
                kwargs = {}
                if quant_enabled and self._quant_layout is not None:
                    kwargs = {
                        "v_q_cache": self.v_q[layer_idx],
                        "v_scale_cache": self.v_scale[layer_idx],
                        "v_zero_cache": self.v_zero[layer_idx],
                        "quant_bits": self._quant_layout.bits,
                        "quant_group_size": self._quant_layout.group_size,
                        "quant_bytes_per_token": self._quant_layout.bytes_per_token,
                        "quant_groups_per_head": self._quant_layout.groups_per_head,
                        "quant_symmetric": bool(self.quant_spec.symmetric),
                    }
                _NATIVE_PAGED_KV_WRITE(
                    self.k[layer_idx],
                    self.v[layer_idx],
                    block_row,
                    start_pos,
                    k_tokens,
                    v_tokens,
                    **kwargs,
                )
                self._update_context_length(seq_id, end_pos, commit=commit)
                return
            except RuntimeError:
                pass

        self._write_prefill_python(seq_id, layer_idx, k_chunk, v_chunk, start_pos)
        if self.v_q is not None and self.quant_spec is not None:
            self._write_prefill_python_quantized(
                seq_id,
                layer_idx,
                v_chunk,
                start_pos,
                block_row,
            )

        self._update_context_length(seq_id, end_pos, commit=commit)

    def table(self, seq_id: int) -> Tuple[mx.array, int]:
        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")
        block_ids = mx.array(table, dtype=mx.int32)
        ctx_len = self._context_lens.get(seq_id, 0)
        return block_ids, ctx_len

    def batch_tables(
        self,
        seq_ids: Sequence[int],
        context_override: Optional[Sequence[int]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Return block tables/context lengths for the provided sequence ids."""
        if not seq_ids:
            raise ValueError("batch_tables expects at least one sequence id")
        tables = mx.full(
            (len(seq_ids), self.max_blocks_per_sequence),
            -1,
            dtype=mx.int32,
        )
        ctx = mx.zeros((len(seq_ids),), dtype=mx.int32)
        for idx, seq_id in enumerate(seq_ids):
            block_table = self._block_tables.get(seq_id)
            if block_table is None:
                raise ValueError(f"sequence {seq_id} missing")
            tables[idx] = mx.array(block_table, dtype=mx.int32)
            if context_override is not None:
                ctx[idx] = int(context_override[idx])
            else:
                ctx[idx] = int(self._context_lens.get(seq_id, 0))
        return tables, ctx

    def prepare_prefill_view(self, seq_id: int, chunk_len: int) -> Tuple[int, int]:
        base_len = self._context_lens.get(seq_id, 0)
        if chunk_len <= 0:
            return base_len, base_len
        staged = self._staged_context_lens.get(seq_id, base_len)
        start = staged
        end = start + chunk_len
        first_block = start // self.block_size
        last_block = math.ceil(end / self.block_size)
        for block_idx in range(first_block, last_block):
            self._ensure_block(seq_id, block_idx, copy_existing=True)
        self._staged_context_lens[seq_id] = end
        self._prefill_active.add(seq_id)
        return base_len, end

    def commit_prefill(self, seq_id: int) -> None:
        staged = self._staged_context_lens.pop(seq_id, None)
        if staged is None:
            self._prefill_active.discard(seq_id)
            return
        self._context_lens[seq_id] = max(self._context_lens.get(seq_id, 0), staged)
        self._prefill_active.discard(seq_id)

    def is_prefill_active(self, seq_id: int) -> bool:
        return seq_id in self._prefill_active

    def free(self, seq_id: int) -> None:
        table = self._block_tables.pop(seq_id, None)
        if table is None:
            return
        for block_id in table:
            if block_id >= 0:
                self._ref_counts[block_id] -= 1
                if self._ref_counts[block_id] == 0:
                    self._release_block(block_id)
        self._context_lens.pop(seq_id, None)
        self._bump_manager_epoch()

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("no available KV blocks")
        return self._free_blocks.popleft()

    def _release_block(self, block_id: int) -> None:
        self.k[:, :, block_id] = 0
        self.v[:, :, block_id] = 0
        if self.v_q is not None:
            self.v_q[:, :, block_id] = 0
        if self.v_scale is not None:
            self.v_scale[:, :, block_id] = 0
        if self.v_zero is not None:
            self.v_zero[:, :, block_id] = 0
        self._free_blocks.append(block_id)

    def _ensure_block(self, seq_id: int, block_idx: int, copy_existing: bool) -> int:
        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")

        if block_idx >= self.max_blocks_per_sequence:
            raise RuntimeError("block index exceeds per-sequence capacity")

        block_id = table[block_idx]
        if block_id < 0:
            block_id = self._allocate_block()
            table[block_idx] = block_id
            self._ref_counts[block_id] = 1
            return block_id

        if copy_existing and self._ref_counts[block_id] > 1:
            new_block = self._allocate_block()
            self.k[:, :, new_block] = self.k[:, :, block_id]
            self.v[:, :, new_block] = self.v[:, :, block_id]
            if self.v_q is not None:
                self.v_q[:, :, new_block] = self.v_q[:, :, block_id]
            if self.v_scale is not None:
                self.v_scale[:, :, new_block] = self.v_scale[:, :, block_id]
            if self.v_zero is not None:
                self.v_zero[:, :, new_block] = self.v_zero[:, :, block_id]
            self._ref_counts[block_id] -= 1
            self._ref_counts[new_block] = 1
            table[block_idx] = new_block
            self._bump_manager_epoch()
            return new_block

        return block_id

    def _prepare_chunk(self, chunk: mx.array) -> mx.array:
        if chunk.ndim != 3:
            raise ValueError("expected chunk with shape [kv_heads, tokens, head_dim]")
        prepared = chunk.astype(self.dtype) if chunk.dtype != self.dtype else chunk
        return mx.swapaxes(prepared, 0, 1)

    def _write_prefill_python(
        self,
        seq_id: int,
        layer_idx: int,
        k_chunk: mx.array,
        v_chunk: mx.array,
        start_pos: int,
    ) -> None:
        tokens = int(k_chunk.shape[1])
        for t in range(tokens):
            position = start_pos + t
            block_idx = position // self.block_size
            offset = position % self.block_size
            block_id = self._ensure_block(seq_id, block_idx, copy_existing=True)
            self.k[layer_idx, :, block_id, offset] = k_chunk[:, t]
            self.v[layer_idx, :, block_id, offset] = v_chunk[:, t]

    def _write_prefill_python_quantized(
        self,
        seq_id: int,
        layer_idx: int,
        v_chunk: mx.array,
        start_pos: int,
        block_row: mx.array,
    ) -> None:
        if self.v_q is None or self.quant_spec is None:
            return

        value_chunk, scale_chunk, zero_chunk = _quantize_v_chunk(
            v_chunk, self.quant_spec
        )
        tokens = int(value_chunk.shape[0])
        for t in range(tokens):
            logical_pos = start_pos + t
            block_idx = logical_pos // self.block_size
            row = logical_pos % self.block_size
            block_id = int(block_row[block_idx].item())
            if block_id < 0:
                continue
            self.v_q[layer_idx, :, block_id, row] = value_chunk[t]
            self.v_scale[layer_idx, :, block_id, row] = scale_chunk[t]
            self.v_zero[layer_idx, :, block_id, row] = zero_chunk[t]

    def _update_context_length(
        self, seq_id: int, end_pos: int, *, commit: bool
    ) -> None:
        if commit:
            self._context_lens[seq_id] = max(self._context_lens.get(seq_id, 0), end_pos)
            self._staged_context_lens.pop(seq_id, None)
            self._prefill_active.discard(seq_id)
            return

        staged_base = self._staged_context_lens.get(
            seq_id, self._context_lens.get(seq_id, 0)
        )
        self._staged_context_lens[seq_id] = max(staged_base, end_pos)

    def quantized_value_layout(self) -> Optional[PagedKVQuantLayout]:
        return self._quant_layout

    def snapshot_blocks(self, seq_id: int, seq_len: int) -> List[int]:
        """Return physical block ids covering seq_len tokens for seq_id."""
        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")
        if seq_len <= 0:
            return []
        blocks_needed = math.ceil(seq_len / self.block_size)
        result: List[int] = []
        for idx in range(min(blocks_needed, len(table))):
            block_id = table[idx]
            if block_id < 0:
                break
            result.append(block_id)
        return result

    def ensure_decode_capacity(self, seq_ids: Sequence[int], tokens: int = 1) -> None:
        tokens = max(1, int(tokens))
        for seq_id in seq_ids:
            position = self._context_lens.get(seq_id, 0)
            target = max(0, position + tokens - 1)
            current_block = position // self.block_size if self.block_size else 0
            target_block = target // self.block_size if self.block_size else 0
            for block_idx in range(current_block, target_block + 1):
                self._ensure_block(seq_id, block_idx, copy_existing=True)

    def decode_write_targets(
        self, seq_ids: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        block_ids: List[int] = []
        offsets: List[int] = []
        for seq_id in seq_ids:
            position = self._context_lens.get(seq_id, 0)
            block_idx = position // self.block_size
            offset = position % self.block_size
            table = self._block_tables.get(seq_id, [])
            block_id = table[block_idx] if block_idx < len(table) else -1
            block_ids.append(block_id)
            offsets.append(offset)
        return block_ids, offsets

    def bump_decode_lengths(self, seq_ids: Sequence[int], delta: int = 1) -> None:
        for seq_id in seq_ids:
            self._context_lens[seq_id] = self._context_lens.get(seq_id, 0) + delta

    def _ensure_decode_block(self, seq_id: int) -> None:
        position = self._context_lens.get(seq_id, 0)
        block_idx = position // self.block_size
        self._ensure_block(seq_id, block_idx, copy_existing=True)

    def pool_id(self) -> int:
        return self._pool_id

    def mapping_epoch(self) -> int:
        return self._manager_epoch

    def _bump_manager_epoch(self) -> None:
        self._manager_epoch = (self._manager_epoch + 1) % (1 << 30)

    def reuse_prefix(self, seq_id: int, block_ids: Sequence[int], seq_len: int) -> None:
        """Attach existing blocks to seq_id and bump refcounts."""
        if not block_ids:
            return
        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")
        for idx, block_id in enumerate(block_ids):
            if block_id < 0 or block_id >= self.max_blocks:
                raise ValueError(f"invalid block id {block_id}")
            table[idx] = block_id
            self._ref_counts[block_id] += 1
        self._context_lens[seq_id] = max(self._context_lens.get(seq_id, 0), seq_len)
        if block_ids:
            self._bump_manager_epoch()

    def _initialize_quant_buffers(
        self, quant_spec: Optional[QuantSpec]
    ) -> Tuple[Optional[mx.array], Optional[mx.array], Optional[mx.array]]:
        if quant_spec is None:
            return None, None, None

        if not quant_spec.requires("v"):
            # Only V quantization is supported in the initial plumbing stage.
            return None, None, None

        bytes_per_token = _bytes_per_token(
            self.head_dim, quant_spec.bits, quant_spec.group_size
        )
        groups = _groups_per_head(self.head_dim, quant_spec.group_size)

        value_shape = (
            self.num_layers,
            self.num_kv_heads,
            self.max_blocks,
            self.block_size,
            bytes_per_token,
        )
        scale_shape = (
            self.num_layers,
            self.num_kv_heads,
            self.max_blocks,
            self.block_size,
            groups,
        )
        zero_shape = scale_shape

        v_q = mx.zeros(value_shape, dtype=quant_spec.storage_dtype)
        v_scale = mx.zeros(scale_shape, dtype=quant_spec.scale_dtype)
        v_zero = mx.zeros(zero_shape, dtype=quant_spec.scale_dtype)

        self._quant_layout = PagedKVQuantLayout(
            bits=quant_spec.bits,
            group_size=quant_spec.group_size,
            storage_dtype=quant_spec.storage_dtype,
            scale_dtype=quant_spec.scale_dtype,
            value_shape=value_shape,
            scale_shape=scale_shape,
            zero_shape=zero_shape,
            bytes_per_token=bytes_per_token,
            groups_per_head=groups,
        )

        return v_q, v_scale, v_zero

    def _quant_write_supported(self) -> bool:
        if self.quant_spec is None or self._quant_layout is None or self.v_q is None:
            return False
        if self.quant_spec.bits not in _SUPPORTED_QUANT_BITS:
            return False
        if self.quant_spec.storage_dtype != mx.uint8:
            return False
        if self.quant_spec.scale_dtype != mx.float16:
            return False
        return True

    def _quant_attention_kwargs(
        self, layer_idx: Optional[int] = None
    ) -> Dict[str, object]:
        if (
            not self._quant_write_supported()
            or self._quant_layout is None
            or self.v_q is None
            or self.v_scale is None
            or self.v_zero is None
        ):
            return {}
        layout = self._quant_layout
        v_q = self.v_q if layer_idx is None else self.v_q[layer_idx]
        v_scale = self.v_scale if layer_idx is None else self.v_scale[layer_idx]
        v_zero = self.v_zero if layer_idx is None else self.v_zero[layer_idx]
        return {
            "v_q_cache": v_q,
            "v_scale_cache": v_scale,
            "v_zero_cache": v_zero,
            "quant_bits": layout.bits,
            "quant_group_size": layout.group_size,
            "quant_bytes_per_token": layout.bytes_per_token,
            "quant_groups_per_head": layout.groups_per_head,
            "quant_symmetric": bool(self.quant_spec.symmetric),
        }


def _quantize_v_chunk(
    v_chunk: mx.array, spec: QuantSpec
) -> Tuple[mx.array, mx.array, mx.array]:
    if v_chunk.ndim != 3:
        raise ValueError("expected value chunk shaped [kv_heads, tokens, head_dim]")

    kv_heads, tokens, head_dim = v_chunk.shape
    bits = spec.bits
    levels = float((1 << bits) - 1)
    group_size = spec.group_size
    groups = _groups_per_head(head_dim, group_size)
    padded_dim = groups * group_size

    tokens_first = mx.swapaxes(v_chunk, 0, 1)
    if padded_dim > head_dim:
        pad = padded_dim - head_dim
        tokens_first = mx.pad(tokens_first, ((0, 0), (0, 0), (0, pad)))

    grouped = mx.reshape(tokens_first, (tokens, kv_heads, groups, group_size))
    grouped_fp = grouped.astype(mx.float32)

    if spec.symmetric:
        offset = float(1 << (bits - 1))
        max_vals = mx.max(grouped_fp, axis=-1)
        min_vals = mx.min(grouped_fp, axis=-1)
        max_abs = mx.maximum(mx.abs(max_vals), mx.abs(min_vals))
        valid = max_abs > 0
        scale = mx.where(valid, max_abs / offset, mx.ones_like(max_abs))
        zero = mx.ones_like(scale) * offset
        q = mx.round(grouped_fp / scale[..., None]) + zero[..., None]
    else:
        min_vals = mx.min(grouped_fp, axis=-1)
        max_vals = mx.max(grouped_fp, axis=-1)
        span = max_vals - min_vals
        valid = span > 0
        scale = mx.where(valid, span / levels, mx.ones_like(span))
        zero = mx.where(
            valid,
            mx.round(-min_vals / scale),
            -min_vals,
        )
        q = mx.round(grouped_fp / scale[..., None] + zero[..., None])

    q = mx.clip(q, 0.0, levels).astype(mx.int32)
    flat = mx.reshape(q, (tokens, kv_heads, groups * group_size))

    values_per_byte = 8 // bits
    pad_features = (-flat.shape[-1]) % values_per_byte
    if pad_features:
        flat = mx.pad(flat, ((0, 0), (0, 0), (0, pad_features)))

    flat_u8 = flat.astype(mx.uint8)
    if bits == 8:
        packed = flat_u8
    else:
        low = flat_u8[..., ::2]
        high = flat_u8[..., 1::2]
        shift = mx.zeros_like(high) + 4
        packed = mx.bitwise_or(low, mx.left_shift(high, shift))

    scale = scale.astype(spec.scale_dtype)
    zero = zero.astype(spec.scale_dtype)
    return packed.astype(spec.storage_dtype), scale, zero


def _dequantize_v_chunk(
    packed: mx.array,
    scale: mx.array,
    zero: mx.array,
    spec: QuantSpec,
) -> mx.array:
    if packed.ndim != 3:
        raise ValueError("expected packed chunk shaped [tokens, kv_heads, bytes]")
    if scale.shape[:2] != packed.shape[:2] or zero.shape[:2] != packed.shape[:2]:
        raise ValueError("scale/zero must align with packed values")

    bits = spec.bits
    group_size = spec.group_size
    groups = scale.shape[-1]

    if bits == 8:
        flat = packed.astype(mx.float32)
    else:
        packed_u8 = packed.astype(mx.uint8)
        low = mx.bitwise_and(packed_u8, mx.array(0x0F, dtype=mx.uint8))
        shift = mx.zeros_like(packed_u8) + 4
        high = mx.bitwise_and(
            mx.right_shift(packed_u8, shift), mx.array(0x0F, dtype=mx.uint8)
        )
        pairs = mx.stack([low, high], axis=-1)
        flat = mx.reshape(pairs, (packed.shape[0], packed.shape[1], -1)).astype(
            mx.float32
        )

    needed = groups * group_size
    if flat.shape[-1] > needed:
        flat = flat[..., :needed]
    elif flat.shape[-1] < needed:
        pad = needed - flat.shape[-1]
        flat = mx.pad(flat, ((0, 0), (0, 0), (0, pad)))

    values = mx.reshape(flat, (packed.shape[0], packed.shape[1], groups, group_size))
    scale = scale.astype(mx.float32)
    zero = zero.astype(mx.float32)
    recon = (values - zero[..., None]) * scale[..., None]
    return mx.reshape(recon, (packed.shape[0], packed.shape[1], groups * group_size))


def _materialize_sequence(
    seq_len: int,
    block_size: int,
    block_row: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
) -> tuple[mx.array, mx.array]:
    tokens_remaining = seq_len
    k_blocks: List[mx.array] = []
    v_blocks: List[mx.array] = []

    for idx in range(block_row.shape[0]):
        if tokens_remaining <= 0:
            break
        block_id = int(block_row[idx].item())
        if block_id < 0:
            continue

        take = min(block_size, tokens_remaining)
        k_blocks.append(k_cache[:, block_id, :take])
        v_blocks.append(v_cache[:, block_id, :take])
        tokens_remaining -= take

    if not k_blocks:
        empty_shape = (k_cache.shape[0], 0, k_cache.shape[-1])
        return (
            mx.zeros(empty_shape, dtype=k_cache.dtype),
            mx.zeros(empty_shape, dtype=v_cache.dtype),
        )

    k_seq = k_blocks[0] if len(k_blocks) == 1 else mx.concatenate(k_blocks, axis=1)
    v_seq = v_blocks[0] if len(v_blocks) == 1 else mx.concatenate(v_blocks, axis=1)
    return k_seq, v_seq


def _paged_attention_reference(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    layer_idx: int | None = None,
    kv_head_mapping: Optional[mx.array] = None,
    rope_freqs: Optional[mx.array] = None,
    scale: Optional[float] = None,
    causal: bool = True,
    attn_mask: Optional[mx.array] = None,
    v_q_cache: Optional[mx.array] = None,
    v_scale_cache: Optional[mx.array] = None,
    v_zero_cache: Optional[mx.array] = None,
    quant_bits: Optional[int] = None,
    quant_group_size: Optional[int] = None,
    quant_groups_per_head: Optional[int] = None,
    quant_symmetric: Optional[bool] = None,
    quant_bytes_per_token: Optional[int] = None,
    **_unused_kwargs,
) -> mx.array:
    del rope_freqs
    del causal
    del attn_mask
    del quant_bytes_per_token

    batch, num_heads, q_len, head_dim = q.shape
    if q_len != 1:
        raise ValueError("paged_attention reference expects decode queries with Lq=1")

    layer_idx = 0 if layer_idx is None else layer_idx
    layer_k = k_cache[layer_idx] if k_cache.ndim == 5 else k_cache
    layer_v = v_cache[layer_idx] if v_cache.ndim == 5 else v_cache

    num_kv_heads = layer_k.shape[0]
    block_size = layer_k.shape[2]
    result = mx.zeros_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if kv_head_mapping is not None:
        mapping = [int(x) for x in kv_head_mapping.tolist()]
    elif num_kv_heads == num_heads:
        mapping = list(range(num_heads))
    elif num_kv_heads == 1:
        mapping = [0 for _ in range(num_heads)]
    else:
        mapping = [(h * num_kv_heads) // num_heads for h in range(num_heads)]

    if _NATIVE_PAGED_ATTENTION is not None:
        try:
            return mx_fast._paged_attention_impl(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                layer_idx=layer_idx,
                kv_head_mapping=kv_head_mapping,
                scale=scale,
                v_q_cache=v_q_cache,
                v_scale_cache=v_scale_cache,
                v_zero_cache=v_zero_cache,
                quant_bits=quant_bits,
                quant_group_size=quant_group_size,
                quant_groups_per_head=quant_groups_per_head,
                quant_symmetric=quant_symmetric,
            )
        except RuntimeError as exc:
            logging.warning(
                "paged_attention native kernel failed: %s q=%s tables=%s ctx=%s layer=%s",
                exc,
                q.shape,
                block_tables.shape,
                context_lens.shape,
                layer_idx,
            )

    for b in range(batch):
        seq_len = int(context_lens[b].item())
        if seq_len == 0:
            continue

        k_seq, v_seq = _materialize_sequence(
            seq_len, block_size, block_tables[b], layer_k, layer_v
        )

        for head_idx in range(num_heads):
            kv_head = mapping[head_idx]
            q_vec = q[b, head_idx, 0]
            k_head = k_seq[kv_head, :seq_len]
            v_head = v_seq[kv_head, :seq_len]

            scores = mx.sum(q_vec * k_head, axis=-1) * scale
            weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            context = mx.sum(mx.expand_dims(weights, axis=-1) * v_head, axis=0)
            result[b, head_idx, 0] = context.astype(q.dtype)

    return result


def prefill_into_kv(
    manager: "KVBlockManager",
    seq_id: int,
    chunk_iterator: Iterable[Tuple[Sequence[mx.array], Sequence[mx.array]]],
    on_chunk_committed: Optional[Callable[[], None]] = None,
) -> None:
    """Write per-chunk layer K/V outputs into the KV cache."""

    start_pos = manager._context_lens.get(seq_id, 0)
    for k_layers, v_layers in chunk_iterator:
        if len(k_layers) != manager.num_layers or len(v_layers) != manager.num_layers:
            raise ValueError("chunk does not match manager.num_layers")
        if not k_layers:
            continue
        chunk_tokens = k_layers[0].shape[1]
        if chunk_tokens == 0:
            continue
        for layer_idx, (k_chunk, v_chunk) in enumerate(zip(k_layers, v_layers)):
            manager.write_prefill(seq_id, layer_idx, k_chunk, v_chunk, start_pos)
        start_pos += chunk_tokens
        if on_chunk_committed is not None:
            on_chunk_committed()


def paged_prefill(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    base_lens: Optional[mx.array],
    context_lens: mx.array,
    *,
    kv_head_mapping: Optional[mx.array] = None,
    scale: Optional[float] = None,
    layer_idx: Optional[int] = None,
    **kwargs,
) -> mx.array:
    """Dispatch paged prefill to the native kernel when available."""

    layer = 0 if layer_idx is None else layer_idx
    effective_base = base_lens if base_lens is not None else context_lens
    quant_kw = {
        "v_q_cache": kwargs.get("v_q_cache"),
        "v_scale_cache": kwargs.get("v_scale_cache"),
        "v_zero_cache": kwargs.get("v_zero_cache"),
        "quant_bits": kwargs.get("quant_bits"),
        "quant_group_size": kwargs.get("quant_group_size"),
        "quant_groups_per_head": kwargs.get("quant_groups_per_head"),
        "quant_symmetric": kwargs.get("quant_symmetric"),
    }
    if quant_kw["v_q_cache"] is not None and quant_kw["v_q_cache"].ndim == 5:
        quant_kw["v_q_cache"] = quant_kw["v_q_cache"][layer]
    if quant_kw["v_scale_cache"] is not None and quant_kw["v_scale_cache"].ndim == 5:
        quant_kw["v_scale_cache"] = quant_kw["v_scale_cache"][layer]
    if quant_kw["v_zero_cache"] is not None and quant_kw["v_zero_cache"].ndim == 5:
        quant_kw["v_zero_cache"] = quant_kw["v_zero_cache"][layer]
    layer_k = k_cache[layer] if k_cache.ndim == 5 else k_cache
    layer_v = v_cache[layer] if v_cache.ndim == 5 else v_cache
    native_prefill = _NATIVE_PAGED_PREFILL
    if native_prefill is not None:
        try:
            return native_prefill(
                q,
                k_cache,
                v_cache,
                block_tables,
                effective_base,
                context_lens,
                layer,
                kv_head_mapping=kv_head_mapping,
                scale=scale,
                **quant_kw,
            )
        except RuntimeError as exc:
            logging.warning(
                "paged_prefill native kernel failed: %s q=%s tables=%s ctx=%s layer=%s",
                exc,
                q.shape,
                block_tables.shape,
                context_lens.shape,
                layer,
            )
    return _paged_prefill_reference(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        base_lens=base_lens,
        kv_head_mapping=kv_head_mapping,
        layer_idx=layer,
        **kwargs,
    )


def _paged_prefill_reference(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    *,
    scale: float,
    base_lens: Optional[mx.array] = None,
    kv_head_mapping: Optional[mx.array] = None,
    **kwargs,
) -> mx.array:
    """Fallback Multi-Q prefill that reuses the decode kernel."""

    total_tokens = int(q.shape[2])
    if total_tokens == 0:
        return q
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("layer_idx", 0)
    batch = int(context_lens.shape[0]) if context_lens.ndim else 1
    if base_lens is not None:
        base_list = [int(x) for x in base_lens.tolist()]
    else:
        base_list = [int(x) for x in context_lens.tolist()]
    outputs = []
    for offset in range(total_tokens):
        q_slice = q[:, :, offset : offset + 1, :]
        if base_lens is not None:
            override = [base_list[b] + offset + 1 for b in range(batch)]
            lens_tensor = mx.array(override, dtype=mx.int32)
        else:
            lens_tensor = context_lens
        outputs.append(
            mx_fast.paged_attention(
                q_slice,
                k_cache,
                v_cache,
                block_tables,
                lens_tensor,
                scale=scale,
                kv_head_mapping=kv_head_mapping,
                **call_kwargs,
            )
        )
    return mx.concatenate(outputs, axis=2) if len(outputs) > 1 else outputs[0]


__all__ = [
    "KVBlockManager",
    "QuantSpec",
    "PagedKVQuantLayout",
    "_dequantize_v_chunk",
    "prefill_into_kv",
    "_paged_prefill_reference",
    "paged_prefill",
]

if not hasattr(mx_fast, "paged_attention"):
    mx_fast.paged_attention = _paged_attention_reference
mx_fast.paged_prefill = paged_prefill
