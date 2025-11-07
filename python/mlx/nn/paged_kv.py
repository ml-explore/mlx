# ABOUTME: Provides paged key/value cache management utilities for attention.
# ABOUTME: Implements block allocation, reference counting, and copy-on-write.

from collections import deque
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx

_NATIVE_PAGED_ATTENTION = getattr(mx.fast, "_paged_attention_impl", None)
_NATIVE_PAGED_KV_WRITE = getattr(mx.fast, "_paged_kv_write_impl", None)
_PAGED_ATTENTION_PREWARM = getattr(mx.fast, "_paged_attention_prewarm", None)

class KVBlockManager:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks: int,
        dtype: mx.Dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype

        self.k = mx.zeros(
            (num_layers, num_kv_heads, max_blocks, block_size, head_dim), dtype=dtype
        )
        self.v = mx.zeros_like(self.k)

        if _PAGED_ATTENTION_PREWARM is not None:
            try:
                _PAGED_ATTENTION_PREWARM(block_size, dtype)
            except RuntimeError:
                pass

        self._free_blocks: deque[int] = deque(range(max_blocks))
        self._ref_counts: List[int] = [0 for _ in range(max_blocks)]
        self._block_tables: Dict[int, List[int]] = {}
        self._context_lens: Dict[int, int] = {}

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

    def write_prefill(
        self,
        seq_id: int,
        layer_idx: int,
        k_chunk: mx.array,
        v_chunk: mx.array,
        start_pos: int,
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

        if _NATIVE_PAGED_KV_WRITE is not None:
            try:
                k_tokens = self._prepare_chunk(k_chunk)
                v_tokens = self._prepare_chunk(v_chunk)
                _NATIVE_PAGED_KV_WRITE(
                    self.k[layer_idx],
                    self.v[layer_idx],
                    block_row,
                    start_pos,
                    k_tokens,
                    v_tokens,
                )
                self._context_lens[seq_id] = max(
                    self._context_lens.get(seq_id, 0), end_pos
                )
                return
            except RuntimeError:
                pass

        self._write_prefill_python(
            seq_id, layer_idx, k_chunk, v_chunk, start_pos
        )
        self._context_lens[seq_id] = max(
            self._context_lens.get(seq_id, 0), end_pos
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
        if v_token.ndim == 1:
            v_token = mx.expand_dims(v_token, 0)

        position = self._context_lens.get(seq_id, 0)
        block_idx = position // self.block_size
        offset = position % self.block_size

        block_id = self._ensure_block(seq_id, block_idx, copy_existing=True)
        self.k[layer_idx, :, block_id, offset] = k_token
        self.v[layer_idx, :, block_id, offset] = v_token

        self._context_lens[seq_id] = position + 1

    def table(self, seq_id: int) -> Tuple[mx.array, int]:
        table = self._block_tables.get(seq_id)
        if table is None:
            raise ValueError(f"sequence {seq_id} missing")
        block_ids = mx.array(table, dtype=mx.int32)
        ctx_len = self._context_lens.get(seq_id, 0)
        return block_ids, ctx_len

    def batch_tables(self, seq_ids: Sequence[int]) -> Tuple[mx.array, mx.array]:
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
            ctx[idx] = int(self._context_lens.get(seq_id, 0))
        return tables, ctx

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

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("no available KV blocks")
        return self._free_blocks.popleft()

    def _release_block(self, block_id: int) -> None:
        self.k[:, :, block_id] = 0
        self.v[:, :, block_id] = 0
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
            self._ref_counts[block_id] -= 1
            self._ref_counts[new_block] = 1
            table[block_idx] = new_block
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


def _paged_attention_reference(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    layer_idx: int,
    kv_head_mapping: Optional[mx.array] = None,
    rope_freqs: Optional[mx.array] = None,
    scale: Optional[float] = None,
    causal: bool = True,
    attn_mask: Optional[mx.array] = None,
) -> mx.array:
    del rope_freqs
    del causal
    del attn_mask

    batch, num_heads, q_len, head_dim = q.shape
    if q_len != 1:
        raise ValueError("paged_attention reference expects decode queries with Lq=1")

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
            return _NATIVE_PAGED_ATTENTION(
                q,
                layer_k,
                layer_v,
                block_tables,
                context_lens,
                0,
                kv_head_mapping=(kv_head_mapping if kv_head_mapping is not None else None),
                scale=float(scale) if scale is not None else None,
            )
        except RuntimeError:
            pass

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


if not hasattr(mx.fast, "paged_attention"):
    mx.fast.paged_attention = _paged_attention_reference


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


__all__ = ["KVBlockManager", "prefill_into_kv"]
