# ABOUTME: Validates paged KV allocator semantics and copy-on-write behavior.
# ABOUTME: Asserts paged attention fallback matches dense attention outputs.

import math
import unittest

import mlx.core as mx
import mlx.core.fast as mx_fast
import mlx.nn.paged_kv as paged_kv
import mlx_tests
import numpy as np
import pytest
from mlx.nn.paged_kv import (
    KVBlockManager,
    QuantSpec,
    _dequantize_v_chunk,
    prefill_into_kv,
)


def _dense_reference(q, dense_k, dense_v, context_lens, scale, kv_head_mapping=None):
    batch, num_heads, _, _ = q.shape
    num_kv_heads = dense_k.shape[1]
    out = mx.zeros_like(q)
    if kv_head_mapping is not None:
        mapping = [int(x) for x in kv_head_mapping.tolist()]
    elif num_kv_heads == num_heads:
        mapping = list(range(num_heads))
    elif num_kv_heads == 1:
        mapping = [0 for _ in range(num_heads)]
    else:
        mapping = [(h * num_kv_heads) // num_heads for h in range(num_heads)]
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        if seq_len == 0:
            continue
        for h in range(num_heads):
            kv_head = mapping[h]
            k_head = dense_k[b, kv_head, :seq_len]
            v_head = dense_v[b, kv_head, :seq_len]
            q_vec = q[b, h, 0]
            scores = mx.sum(q_vec * k_head, axis=-1) * scale
            weights = mx.softmax(scores.astype(mx.float32))
            out[b, h, 0] = mx.sum(mx.expand_dims(weights, axis=-1) * v_head, axis=0)
    return out


def _build_paged_cache(dense_k, dense_v, seq_lens, block_size):
    batch, kv_heads, max_len, head_dim = dense_k.shape
    blocks_per_seq = math.ceil(max_len / block_size)
    total_blocks = batch * blocks_per_seq
    k_cache = mx.zeros(
        (1, kv_heads, total_blocks, block_size, head_dim), dtype=dense_k.dtype
    )
    v_cache = mx.zeros_like(k_cache)
    block_tables = mx.full((batch, blocks_per_seq), -1, dtype=mx.int32)
    for b in range(batch):
        tokens = seq_lens[b]
        for t in range(tokens):
            block_idx = t // block_size
            block_id = b * blocks_per_seq + block_idx
            block_tables[b, block_idx] = block_id
            offset = t % block_size
            k_cache[0, :, block_id, offset] = dense_k[b, :, t]
            v_cache[0, :, block_id, offset] = dense_v[b, :, t]
    return k_cache, v_cache, block_tables


def _apply_rope_sequence(tensor, base_offsets):
    batch, heads, seq, dim = tensor.shape
    rotated = []
    for pos in range(seq):
        token = tensor[:, :, pos : pos + 1, :]
        offsets = mx.array([base + pos for base in base_offsets], dtype=mx.int32)
        rotated.append(
            mx.fast.rope(
                token,
                dim,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=offsets,
            )
        )
    return mx.concatenate(rotated, axis=2)


def _apply_rope_decode(tokens, offsets):
    dim = tokens.shape[-1]
    pieces = []
    for b, offset in enumerate(offsets):
        pieces.append(
            mx.fast.rope(
                tokens[b : b + 1],
                dim,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=int(offset),
            )
        )
    return mx.concatenate(pieces, axis=0)


_UNSUPPORTED_MSG = "configuration not supported"


def _cpu_write_batch(k_cache, v_cache, block_tables, context_lens, k_batch, v_batch):
    block_size = k_cache.shape[2]
    for seq_idx in range(k_batch.shape[0]):
        pos = int(context_lens[seq_idx])
        block_idx = pos // block_size
        row = pos % block_size
        block_id = int(block_tables[seq_idx, block_idx])
        if block_id < 0:
            continue
        k_cache[:, block_id, row, :] = k_batch[seq_idx]
        v_cache[:, block_id, row, :] = v_batch[seq_idx]


def _cpu_write_layers_batch(
    k_cache, v_cache, block_tables, context_lens, k_layers, v_layers
):
    for layer_idx in range(k_layers.shape[0]):
        _cpu_write_batch(
            k_cache[layer_idx],
            v_cache[layer_idx],
            block_tables,
            context_lens,
            k_layers[layer_idx],
            v_layers[layer_idx],
        )


def _cpu_write_layers_tokens(
    k_cache, v_cache, block_tables, context_lens, k_tokens, v_tokens
):
    layers, steps = k_tokens.shape[:2]
    for step in range(steps):
        _cpu_write_layers_batch(
            k_cache,
            v_cache,
            block_tables,
            context_lens + step,
            k_tokens[:, step],
            v_tokens[:, step],
        )


def _write_batch_with_fallback(
    k_cache, v_cache, block_tables, context_lens, k_batch, v_batch
):
    try:
        mx.fast.paged_kv_write_batch(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_batch,
            v_batch,
        )
        return k_cache, v_cache
    except RuntimeError as exc:
        if _UNSUPPORTED_MSG not in str(exc):
            raise
    k_np = np.array(k_cache)
    v_np = np.array(v_cache)
    _cpu_write_batch(
        k_np,
        v_np,
        np.array(block_tables),
        np.array(context_lens),
        np.array(k_batch),
        np.array(v_batch),
    )
    return mx.array(k_np, dtype=k_cache.dtype), mx.array(v_np, dtype=v_cache.dtype)


def _write_layers_batch_with_fallback(
    k_cache, v_cache, block_tables, context_lens, k_layers, v_layers
):
    try:
        mx.fast.paged_kv_write_layers_batch(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_layers,
            v_layers,
        )
        return k_cache, v_cache
    except RuntimeError as exc:
        if _UNSUPPORTED_MSG not in str(exc):
            raise
    k_np = np.array(k_cache)
    v_np = np.array(v_cache)
    _cpu_write_layers_batch(
        k_np,
        v_np,
        np.array(block_tables),
        np.array(context_lens),
        np.array(k_layers),
        np.array(v_layers),
    )
    return mx.array(k_np, dtype=k_cache.dtype), mx.array(v_np, dtype=v_cache.dtype)


def _write_layers_tokens_with_fallback(
    k_cache, v_cache, block_tables, context_lens, k_tokens, v_tokens
):
    try:
        mx.fast.paged_kv_write_layers_tokens(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_tokens,
            v_tokens,
        )
        return k_cache, v_cache
    except RuntimeError as exc:
        if _UNSUPPORTED_MSG not in str(exc):
            raise
    k_np = np.array(k_cache)
    v_np = np.array(v_cache)
    _cpu_write_layers_tokens(
        k_np,
        v_np,
        np.array(block_tables),
        np.array(context_lens),
        np.array(k_tokens),
        np.array(v_tokens),
    )
    return mx.array(k_np, dtype=k_cache.dtype), mx.array(v_np, dtype=v_cache.dtype)


@pytest.mark.skipif(mx is None, reason="mlx.core not available")
def test_overlay_fallback_matches_dense_reference():
    import mlx.nn.paged_kv as paged_kv_mod

    mx.set_default_device(mx.cpu)
    manager = paged_kv_mod.KVBlockManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        max_blocks=4,
        dtype=mx.float16,
    )
    seq_id = 0
    manager.new_sequence(seq_id=seq_id, prompt_len=1)
    block_tables, context_lens = manager.batch_tables([seq_id])
    overlay_k = mx.full(
        (1, 1, manager.num_kv_heads, manager.head_dim), 0.5, dtype=mx.float16
    )
    overlay_v = overlay_k + 1.0
    queries = mx.ones((1, 1, 1, manager.head_dim), dtype=mx.float16)

    ref_k_np = np.array(manager.k, copy=True)
    ref_v_np = np.array(manager.v, copy=True)
    block_id = int(block_tables[0, 0].item())
    offset = int(context_lens[0].item())
    ref_k_np[0, :, block_id, offset] = np.array(overlay_k[0, 0])
    ref_v_np[0, :, block_id, offset] = np.array(overlay_v[0, 0])
    ref_k = mx.array(ref_k_np)
    ref_v = mx.array(ref_v_np)
    dense = mx.fast.paged_attention(
        queries,
        ref_k,
        ref_v,
        block_tables,
        context_lens + 1,
        layer_idx=0,
        scale=1.0,
    )

    fallback = paged_kv_mod._paged_attention_overlay_fallback(
        queries,
        manager.k,
        manager.v,
        block_tables,
        context_lens,
        layer_idx=0,
        k_overlay=overlay_k,
        v_overlay=overlay_v,
        scale=1.0,
    )

    assert mx.allclose(dense, fallback, atol=1e-4)

    max_blocks = block_tables.shape[1]
    value_bytes = 2
    v_q_dummy = mx.zeros(
        (1, manager.num_kv_heads, max_blocks, manager.block_size, value_bytes),
        dtype=mx.uint8,
    )
    v_scale_dummy = mx.ones(
        (1, manager.num_kv_heads, max_blocks, manager.block_size, 1),
        dtype=mx.float16,
    )
    v_zero_dummy = mx.zeros_like(v_scale_dummy)
    fallback_quant = paged_kv_mod._paged_attention_overlay_fallback(
        queries,
        manager.k,
        manager.v,
        block_tables,
        context_lens,
        layer_idx=0,
        k_overlay=overlay_k,
        v_overlay=overlay_v,
        scale=1.0,
        v_q_cache=v_q_dummy,
        v_scale_cache=v_scale_dummy,
        v_zero_cache=v_zero_dummy,
        quant_bits=4,
        quant_group_size=4,
        quant_groups_per_head=1,
        quant_symmetric=False,
    )

    assert mx.allclose(dense, fallback_quant, atol=1e-4)


@pytest.mark.skipif(mx is None, reason="mlx.core not available")
def test_overlay_fallback_does_not_touch_kv_cache():
    import mlx.nn.paged_kv as paged_kv_mod

    mx.set_default_device(mx.cpu)
    manager = paged_kv_mod.KVBlockManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        block_size=4,
        max_blocks=4,
        dtype=mx.float16,
    )
    seq_id = 0
    manager.new_sequence(seq_id=seq_id, prompt_len=2)
    block_tables, context_lens = manager.batch_tables([seq_id])
    overlay_k = mx.full(
        (2, 1, manager.num_kv_heads, manager.head_dim),
        0.25,
        dtype=mx.float16,
    )
    overlay_v = overlay_k + 0.5
    queries = mx.ones((1, 4, 1, manager.head_dim), dtype=mx.float16)
    ref_k_np = np.array(manager.k, copy=True)
    ref_v_np = np.array(manager.v, copy=True)
    base_len = int(context_lens[0].item())
    for token_idx in range(overlay_k.shape[0]):
        absolute = base_len + token_idx
        block_idx = absolute // manager.block_size
        offset = absolute % manager.block_size
        block_id = int(block_tables[0, block_idx].item())
        ref_k_np[0, :, block_id, offset] = np.array(overlay_k[token_idx, 0])
        ref_v_np[0, :, block_id, offset] = np.array(overlay_v[token_idx, 0])
    ref_k = mx.array(ref_k_np)
    ref_v = mx.array(ref_v_np)
    dense = mx_fast.paged_attention(
        queries,
        ref_k,
        ref_v,
        block_tables,
        context_lens + overlay_k.shape[0],
        layer_idx=0,
        scale=1.0,
    )

    original_write = paged_kv_mod.mx_fast.paged_kv_write_layers_tokens

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("paged_kv_write_layers_tokens should not be called")

    paged_kv_mod.mx_fast.paged_kv_write_layers_tokens = _unexpected
    try:
        fallback = paged_kv_mod._paged_attention_overlay_fallback(
            queries,
            manager.k,
            manager.v,
            block_tables,
            context_lens,
            layer_idx=0,
            k_overlay=overlay_k,
            v_overlay=overlay_v,
            scale=1.0,
        )
    finally:
        paged_kv_mod.mx_fast.paged_kv_write_layers_tokens = original_write

    assert mx.allclose(dense, fallback, atol=1e-4)


class TestKVBlockManager(mlx_tests.MLXTestCase):
    def test_new_sequence_allocates_expected_blocks(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            max_blocks=8,
            dtype=mx.float16,
        )

        manager.new_sequence(seq_id=0, prompt_len=3)

        block_ids, ctx_len = manager.table(0)
        self.assertEqual(ctx_len, 3)
        self.assertEqual(block_ids.shape, (manager.max_blocks_per_sequence,))
        active = block_ids[:2]
        self.assertTrue(mx.array_equal(active, mx.array([0, 1], dtype=mx.int32)))

    def test_write_prefill_populates_blocks_in_order(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            max_blocks=4,
            dtype=mx.float16,
        )

        manager.new_sequence(seq_id=0, prompt_len=3)

        k_chunk = mx.reshape(
            mx.arange(12, dtype=mx.float16), (1, 3, 4)
        )  # [kv_heads, tokens, dim]
        v_chunk = k_chunk + 100
        manager.write_prefill(
            seq_id=0, layer_idx=0, k_chunk=k_chunk, v_chunk=v_chunk, start_pos=0
        )

        block_ids, _ = manager.table(0)
        first_block = int(block_ids[0].item())
        second_block = int(block_ids[1].item())

        k_store = manager.k[0, 0]
        v_store = manager.v[0, 0]
        self.assertTrue(
            mx.array_equal(
                k_store[first_block, 0], mx.array([0, 1, 2, 3], dtype=mx.float16)
            )
        )
        self.assertTrue(
            mx.array_equal(
                k_store[first_block, 1], mx.array([4, 5, 6, 7], dtype=mx.float16)
            )
        )
        self.assertTrue(
            mx.array_equal(
                k_store[second_block, 0], mx.array([8, 9, 10, 11], dtype=mx.float16)
            )
        )
        self.assertTrue(
            mx.array_equal(
                v_store[first_block, 0],
                mx.array([100, 101, 102, 103], dtype=mx.float16),
            )
        )

    def test_fork_performs_copy_on_write_on_tail_block(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=6,
            dtype=mx.float16,
        )

        manager.new_sequence(seq_id=0, prompt_len=2)
        manager.fork(parent_seq_id=0, child_seq_id=1)

        k_token = mx.array([[1.0, 2.0]], dtype=mx.float16)
        v_token = mx.array([[3.0, 4.0]], dtype=mx.float16)
        manager.append_decode_token(
            seq_id=1, layer_idx=0, k_token=k_token, v_token=v_token
        )

        parent_blocks, _ = manager.table(0)
        child_blocks, _ = manager.table(1)

        self.assertEqual(parent_blocks[0].item(), child_blocks[0].item())
        self.assertNotEqual(parent_blocks[1].item(), child_blocks[1].item())

    def test_batch_tables_serializes_requested_sequences(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=4,
            dtype=mx.float16,
        )
        manager.new_sequence(seq_id=0, prompt_len=3)
        manager.new_sequence(seq_id=1, prompt_len=1)

        tables, ctx = manager.batch_tables([1, 0])
        self.assertEqual(tables.shape, (2, manager.max_blocks_per_sequence))
        self.assertTrue(mx.array_equal(ctx, mx.array([1, 3], dtype=mx.int32)))

    def test_quant_buffers_allocated_for_values(self):
        spec = QuantSpec(bits=4, group_size=2)
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            block_size=4,
            max_blocks=6,
            dtype=mx.float16,
            kv_quantization=spec,
        )
        self.assertIsNotNone(manager.v_q)
        self.assertIsNotNone(manager.v_scale)
        self.assertIsNotNone(manager.v_zero)
        layout = manager.quantized_value_layout()
        self.assertIsNotNone(layout)
        self.assertEqual(layout.bits, 4)
        self.assertEqual(layout.group_size, 2)
        self.assertEqual(layout.value_shape[-1], math.ceil(8 * 4 / 8))
        self.assertEqual(layout.scale_shape[-1], 4)
        self.assertEqual(layout.zero_shape[-1], 4)

    def test_quant_buffers_copy_on_write(self):
        spec = QuantSpec(bits=4, group_size=2)
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            max_blocks=4,
            dtype=mx.float16,
            kv_quantization=spec,
        )
        manager.new_sequence(seq_id=0, prompt_len=2)
        block_ids, _ = manager.table(0)
        first_block = int(block_ids[0].item())
        # Fill quantized buffer for block 0 with sentinel values.
        sentinel = mx.full(
            manager.v_q[:, :, first_block].shape,
            7,
            dtype=manager.v_q.dtype,
        )
        manager.v_q[:, :, first_block] = sentinel

        manager.fork(parent_seq_id=0, child_seq_id=1)

        k_chunk = mx.zeros((1, 2, 4), dtype=mx.float16)
        v_chunk = mx.zeros_like(k_chunk)
        manager.write_prefill(
            seq_id=1, layer_idx=0, k_chunk=k_chunk, v_chunk=v_chunk, start_pos=0
        )

        child_blocks, _ = manager.table(1)
        new_block = int(child_blocks[0].item())
        self.assertNotEqual(new_block, first_block)
        self.assertTrue(
            mx.array_equal(
                manager.v_q[:, :, first_block],
                sentinel,
            )
        )


class TestPagedKVWriteBatch(mlx_tests.MLXTestCase):
    @unittest.skipUnless(mx.default_device() == mx.gpu, "GPU device required")
    def test_batch_kernel_matches_scalar_writer(self):
        kv_heads = 2
        max_blocks = 9
        block_size = 4
        head_dim = 8
        batch = 3

        k_cache = mx.zeros(
            (kv_heads, max_blocks, block_size, head_dim), dtype=mx.float16
        )
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            dtype=mx.int32,
        )
        context_lens = mx.array([0, 5, 9], dtype=mx.int32)
        k_batch = mx.reshape(
            mx.arange(batch * kv_heads * head_dim, dtype=mx.float16),
            (batch, kv_heads, head_dim),
        )
        v_batch = k_batch + 1000

        # Reference: manual scatter into the same cache layout.
        k_ref_np = np.array(k_cache)
        v_ref_np = np.array(v_cache)
        block_tables_np = np.array(block_tables, dtype=np.int32)
        context_np = np.array(context_lens, dtype=np.int32)
        k_batch_np = np.array(k_batch)
        v_batch_np = np.array(v_batch)
        for seq_idx in range(batch):
            pos = int(context_np[seq_idx])
            block_idx = pos // block_size
            row = pos % block_size
            block_id = int(block_tables_np[seq_idx, block_idx])
            k_ref_np[:, block_id, row, :] = k_batch_np[seq_idx]
            v_ref_np[:, block_id, row, :] = v_batch_np[seq_idx]
        k_ref = mx.array(k_ref_np, dtype=k_cache.dtype)
        v_ref = mx.array(v_ref_np, dtype=v_cache.dtype)

        k_cache, v_cache = _write_batch_with_fallback(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_batch,
            v_batch,
        )
        mx.eval(k_cache, v_cache, k_ref, v_ref)
        self.assertTrue(bool(mx.allclose(k_cache, k_ref)))
        self.assertTrue(bool(mx.allclose(v_cache, v_ref)))

    @unittest.skipUnless(mx.default_device() == mx.gpu, "GPU device required")
    @pytest.mark.skip(reason="paged_kv_write_batch not graph-safe under mx.compile yet")
    def test_batch_kernel_compiles_with_mx_compile(self):
        kv_heads = 1
        max_blocks = 3
        block_size = 2
        head_dim = 4
        batch = 2

        k_cache = mx.zeros(
            (kv_heads, max_blocks, block_size, head_dim), dtype=mx.float16
        )
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.array([[0, 1], [2, -1]], dtype=mx.int32)
        context_lens = mx.array([0, 1], dtype=mx.int32)
        k_batch = mx.reshape(
            mx.arange(batch * kv_heads * head_dim, dtype=mx.float16),
            (batch, kv_heads, head_dim),
        )
        v_batch = k_batch + 10

        def writer(kc, vc, tables, lens, kb, vb):
            kc_out = mx.array(kc)
            vc_out = mx.array(vc)
            mx.fast.paged_kv_write_batch(kc_out, vc_out, tables, lens, kb, vb)
            return kc_out, vc_out

        compiled = mx.compile(writer)
        eager_out = writer(
            k_cache, v_cache, block_tables, context_lens, k_batch, v_batch
        )
        compiled_out = compiled(
            k_cache, v_cache, block_tables, context_lens, k_batch, v_batch
        )
        mx.eval(*eager_out, *compiled_out)
        self.assertTrue(bool(mx.allclose(eager_out[0], compiled_out[0])))
        self.assertTrue(bool(mx.allclose(eager_out[1], compiled_out[1])))

    @unittest.skipUnless(mx.default_device() == mx.gpu, "GPU device required")
    def test_layers_batch_matches_per_layer_writes(self):
        layers = 3
        kv_heads = 2
        max_blocks = 4
        block_size = 2
        head_dim = 4
        batch = 2

        k_cache = mx.zeros(
            (layers, kv_heads, max_blocks, block_size, head_dim), dtype=mx.float16
        )
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.array(
            [
                [0, 1, -1, -1],
                [2, 3, -1, -1],
            ],
            dtype=mx.int32,
        )
        context_lens = mx.array([0, 1], dtype=mx.int32)

        layer_batches = []
        layer_values = []
        for l in range(layers):
            base = l * 100
            data = mx.reshape(
                mx.arange(batch * kv_heads * head_dim, dtype=mx.float16) + base,
                (batch, kv_heads, head_dim),
            )
            layer_batches.append(data)
            layer_values.append(data + 500)
        k_layers = mx.stack(layer_batches, axis=0)
        v_layers = mx.stack(layer_values, axis=0)

        ref_k = mx.array(k_cache)
        ref_v = mx.array(v_cache)
        for layer_idx in range(layers):
            ref_k[layer_idx], ref_v[layer_idx] = _write_batch_with_fallback(
                ref_k[layer_idx],
                ref_v[layer_idx],
                block_tables,
                context_lens,
                k_layers[layer_idx],
                v_layers[layer_idx],
            )

        k_cache, v_cache = _write_layers_batch_with_fallback(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_layers,
            v_layers,
        )
        mx.eval(k_cache, v_cache, ref_k, ref_v)
        self.assertTrue(bool(mx.allclose(k_cache, ref_k)))
        self.assertTrue(bool(mx.allclose(v_cache, ref_v)))

    @unittest.skipUnless(mx.default_device() == mx.gpu, "GPU device required")
    def test_layers_tokens_writer_matches_step_writes(self):
        layers = 2
        steps = 3
        kv_heads = 2
        max_blocks = 6
        block_size = 2
        head_dim = 4
        batch = 2

        k_cache = mx.zeros(
            (layers, kv_heads, max_blocks, block_size, head_dim), dtype=mx.float16
        )
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=mx.int32,
        )
        context_lens = mx.array([0, 1], dtype=mx.int32)

        rng = np.random.default_rng(42)
        tokens_shape = (layers, steps, batch, kv_heads, head_dim)
        k_tokens = mx.array(rng.standard_normal(tokens_shape).astype(np.float16))
        v_tokens = k_tokens + 0.5

        ref_k = mx.array(k_cache)
        ref_v = mx.array(v_cache)
        for step in range(steps):
            k_step = k_tokens[:, step]
            v_step = v_tokens[:, step]
            ref_k, ref_v = _write_layers_batch_with_fallback(
                ref_k,
                ref_v,
                block_tables,
                context_lens + step,
                k_step,
                v_step,
            )

        k_cache, v_cache = _write_layers_tokens_with_fallback(
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_tokens,
            v_tokens,
        )
        mx.eval(k_cache, v_cache, ref_k, ref_v)
        self.assertTrue(bool(mx.allclose(k_cache, ref_k)))
        self.assertTrue(bool(mx.allclose(v_cache, ref_v)))

    def test_python_quantization_roundtrip(self):
        spec = QuantSpec(bits=4, group_size=2)
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=4,
            max_blocks=4,
            dtype=mx.float16,
            kv_quantization=spec,
        )
        manager.new_sequence(seq_id=0, prompt_len=0)
        block_ids, _ = manager.table(0)
        block_id = int(block_ids[0].item())
        block_row = mx.array([block_id], dtype=mx.int32)
        v_chunk = mx.array(
            [
                [[-1.0, -0.5, 0.25, 0.75]],
            ]
        )  # [kv_heads, tokens, head_dim]
        manager._write_prefill_python_quantized(
            seq_id=0,
            layer_idx=0,
            v_chunk=v_chunk,
            start_pos=0,
            block_row=block_row,
        )
        packed = manager.v_q[0, 0, block_id, 0]
        scale = manager.v_scale[0, 0, block_id, 0]
        zero = manager.v_zero[0, 0, block_id, 0]
        recon = _dequantize_v_chunk(
            packed[None, None, :],
            scale[None, None, :],
            zero[None, None, :],
            spec,
        )
        recon = recon[0, 0, : v_chunk.shape[-1]]
        self.assertTrue(bool(mx.allclose(recon, v_chunk[0, 0], atol=5e-1).item()))

    def test_quantized_native_write_receives_args(self):
        spec = QuantSpec(bits=4, group_size=2)
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            max_blocks=4,
            dtype=mx.float16,
            kv_quantization=spec,
        )
        manager.new_sequence(seq_id=0, prompt_len=0)

        k_chunk = mx.ones((1, 2, 4), dtype=mx.float16)
        v_chunk = mx.arange(8, dtype=mx.float16).reshape(1, 2, 4)
        calls = {}

        original = paged_kv._NATIVE_PAGED_KV_WRITE

        def fake_writer(*args, **kwargs):
            calls["kwargs"] = kwargs
            raise RuntimeError("force python fallback")

        paged_kv._NATIVE_PAGED_KV_WRITE = fake_writer
        try:
            manager.write_prefill(
                seq_id=0,
                layer_idx=0,
                k_chunk=k_chunk,
                v_chunk=v_chunk,
                start_pos=0,
            )
        finally:
            paged_kv._NATIVE_PAGED_KV_WRITE = original

        self.assertIn("kwargs", calls)
        quant_kwargs = calls["kwargs"]
        self.assertEqual(quant_kwargs["quant_bits"], 4)
        self.assertEqual(quant_kwargs["quant_group_size"], 2)
        self.assertEqual(quant_kwargs["quant_bytes_per_token"], math.ceil(4 * 4 / 8))
        self.assertFalse(quant_kwargs["quant_symmetric"])
        block_ids, _ = manager.table(0)
        block_id = int(block_ids[0].item())
        stored = manager.v_q[0, 0, block_id, 0]
        self.assertFalse(bool(mx.array_equal(stored, mx.zeros_like(stored)).item()))


class TestPagedAttentionReference(mlx_tests.MLXTestCase):
    def test_paged_attention_matches_dense_reference(self):
        q_heads = 2
        kv_heads = 2
        head_dim = 4
        batch = 2
        block_size = 2

        q = mx.arange(batch * q_heads * head_dim, dtype=mx.float32).reshape(
            batch, q_heads, 1, head_dim
        )
        dense_k = mx.arange(batch * kv_heads * 3 * head_dim, dtype=mx.float32).reshape(
            batch, kv_heads, 3, head_dim
        )
        dense_v = dense_k + 500
        scale = 1.0 / math.sqrt(head_dim)
        context_lens = mx.array([3, 2], dtype=mx.int32)

        dense_out = _dense_reference(q, dense_k, dense_v, context_lens, scale)

        k_cache = mx.zeros((1, kv_heads, 4, block_size, head_dim), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.full((batch, 4), -1, dtype=mx.int32)

        # Populate paged cache with dense data
        for b in range(batch):
            tokens = context_lens[b].item()
            for t in range(tokens):
                block_idx = t // block_size
                block_id = b * 2 + block_idx
                offset = t % block_size
                block_tables[b, block_idx] = block_id
                k_cache[0, :, block_id, offset] = dense_k[b, :, t]
                v_cache[0, :, block_id, offset] = dense_v[b, :, t]

        paged_out = mx.fast.paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            layer_idx=0,
            kv_head_mapping=None,
            rope_freqs=None,
            scale=scale,
        )

        self.assertTrue(mx.allclose(paged_out, dense_out, rtol=1e-4, atol=1e-4))


class TestPagedAttentionRobustness(mlx_tests.MLXTestCase):
    def test_long_context_parity(self):
        head_dim = 32
        q_heads = 4
        kv_heads = 4
        dtype = mx.float16
        seq_templates = [
            lambda bs: [bs - 1, bs + 1],
            lambda bs: [2 * bs - 1, 2 * bs + 1],
            lambda bs: [bs, max(1, bs // 2), 8 * bs + 7],
        ]
        for block_size in (16, 32, 64):
            for tpl in seq_templates:
                seq_lens = [max(1, s) for s in tpl(block_size)]
                batch = len(seq_lens)
                max_len = max(seq_lens)
                q = mx.random.normal((batch, q_heads, 1, head_dim), dtype=dtype)
                dense_k = mx.random.normal(
                    (batch, kv_heads, max_len, head_dim), dtype=dtype
                )
                dense_v = mx.random.normal(
                    (batch, kv_heads, max_len, head_dim), dtype=dtype
                )
                k_cache, v_cache, block_tables = _build_paged_cache(
                    dense_k, dense_v, seq_lens, block_size
                )
                context_lens = mx.array(seq_lens, dtype=mx.int32)
                scale = 1.0 / math.sqrt(head_dim)
                dense_out = _dense_reference(q, dense_k, dense_v, context_lens, scale)
                paged_out = mx.fast.paged_attention(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    layer_idx=0,
                    kv_head_mapping=None,
                    rope_freqs=None,
                    scale=scale,
                )
                self.assertTrue(mx.allclose(paged_out, dense_out, rtol=5e-3, atol=5e-3))

    def test_gqa_mapping_matches_dense(self):
        block_size = 32
        seq_lens = [block_size + 3, block_size * 2 - 5, 7]
        q_heads = 8
        kv_heads = 2
        head_dim = 32
        batch = len(seq_lens)
        max_len = max(seq_lens)
        q = mx.random.normal((batch, q_heads, 1, head_dim), dtype=mx.float16)
        dense_k = mx.random.normal(
            (batch, kv_heads, max_len, head_dim), dtype=mx.float16
        )
        dense_v = mx.random.normal(
            (batch, kv_heads, max_len, head_dim), dtype=mx.float16
        )
        k_cache, v_cache, block_tables = _build_paged_cache(
            dense_k, dense_v, seq_lens, block_size
        )
        context_lens = mx.array(seq_lens, dtype=mx.int32)
        scale = 1.0 / math.sqrt(head_dim)
        dense_out = _dense_reference(q, dense_k, dense_v, context_lens, scale)
        paged_out = mx.fast.paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            layer_idx=0,
            kv_head_mapping=None,
            rope_freqs=None,
            scale=scale,
        )
        self.assertTrue(mx.allclose(paged_out, dense_out, rtol=5e-3, atol=5e-3))

    def test_copy_on_write_tail_block(self):
        from mlx.nn.paged_kv import KVBlockManager

        block_size = 8
        kv_heads = 2
        head_dim = 4
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_blocks=8,
            dtype=mx.float16,
        )
        manager.new_sequence(seq_id=0, prompt_len=block_size)
        k_chunk = mx.ones((kv_heads, block_size, head_dim), dtype=mx.float16)
        v_chunk = 2 * k_chunk
        manager.write_prefill(0, 0, k_chunk, v_chunk, 0)
        manager.fork(0, 1)

        k_token = mx.full((kv_heads, head_dim), 7.0, dtype=mx.float16)
        v_token = mx.full((kv_heads, head_dim), 9.0, dtype=mx.float16)
        manager.append_decode_token(1, 0, k_token, v_token)

        parent_blocks, _ = manager.table(0)
        child_blocks, _ = manager.table(1)
        self.assertEqual(parent_blocks[0].item(), child_blocks[0].item())
        self.assertNotEqual(parent_blocks[1].item(), child_blocks[1].item())

        self.assertEqual(parent_blocks[1].item(), -1)
        self.assertGreaterEqual(child_blocks[1].item(), 0)
        child_tail = manager.k[0, 0, int(child_blocks[1].item()), 0]
        self.assertTrue(mx.array_equal(child_tail, k_token[0]))

    def test_rope_offsets_match_dense(self):
        block_size = 32
        seq_lens = [25, 41]
        batch = len(seq_lens)
        q_heads = 4
        kv_heads = 4
        head_dim = 64
        max_len = max(seq_lens)
        dense_k_raw = mx.random.normal(
            (batch, kv_heads, max_len, head_dim), dtype=mx.float32
        )
        dense_v = mx.random.normal(
            (batch, kv_heads, max_len, head_dim), dtype=mx.float32
        )
        k_rot = _apply_rope_sequence(dense_k_raw, [0] * batch)
        q_raw = mx.random.normal((batch, q_heads, 1, head_dim), dtype=mx.float32)
        q_rot = _apply_rope_decode(q_raw, seq_lens)

        k_cache, v_cache, block_tables = _build_paged_cache(
            k_rot, dense_v, seq_lens, block_size
        )
        context_lens = mx.array(seq_lens, dtype=mx.int32)
        scale = 1.0 / math.sqrt(head_dim)
        dense_out = _dense_reference(q_rot, k_rot, dense_v, context_lens, scale)
        paged_out = mx.fast.paged_attention(
            q=q_rot,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            layer_idx=0,
            kv_head_mapping=None,
            rope_freqs=None,
            scale=scale,
        )
        self.assertTrue(mx.allclose(paged_out, dense_out, rtol=5e-3, atol=5e-3))


class TestPrefillHelpers(mlx_tests.MLXTestCase):
    def test_prefill_into_kv_writes_chunks_and_yields(self):
        manager = KVBlockManager(
            num_layers=2,
            num_kv_heads=1,
            head_dim=2,
            block_size=4,
            max_blocks=8,
            dtype=mx.float32,
        )
        seq_id = 9
        manager.new_sequence(seq_id=seq_id, prompt_len=0)

        def chunk_iter():
            k_chunk1 = [
                mx.ones((1, 2, 2), dtype=mx.float32),
                2 * mx.ones((1, 2, 2), dtype=mx.float32),
            ]
            v_chunk1 = [
                3 * mx.ones((1, 2, 2), dtype=mx.float32),
                4 * mx.ones((1, 2, 2), dtype=mx.float32),
            ]
            yield k_chunk1, v_chunk1
            k_chunk2 = [
                5 * mx.ones((1, 1, 2), dtype=mx.float32),
                6 * mx.ones((1, 1, 2), dtype=mx.float32),
            ]
            v_chunk2 = [
                7 * mx.ones((1, 1, 2), dtype=mx.float32),
                8 * mx.ones((1, 1, 2), dtype=mx.float32),
            ]
            yield k_chunk2, v_chunk2

        yield_calls: list[bool] = []
        prefill_into_kv(manager, seq_id, chunk_iter(), lambda: yield_calls.append(True))
        self.assertEqual(len(yield_calls), 2)

        block_ids, ctx_len = manager.table(seq_id)
        self.assertEqual(ctx_len, 3)
        first_block = int(block_ids[0].item())
        stored = manager.k[0, 0, first_block]
        self.assertTrue(mx.array_equal(stored[0], mx.ones((2,), dtype=mx.float32)))
        self.assertTrue(mx.array_equal(stored[1], mx.ones((2,), dtype=mx.float32)))
        self.assertTrue(mx.array_equal(stored[2], 5 * mx.ones((2,), dtype=mx.float32)))


@unittest.skipUnless(hasattr(mx.fast, "paged_attention"), "paged attention unavailable")
class PagedPrefillReferenceTests(mlx_tests.MLXTestCase):
    def test_reference_matches_decode_loop(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=8,
            dtype=mx.float32,
        )
        manager.new_sequence(seq_id=0, prompt_len=0)
        k_chunk = mx.reshape(mx.arange(8, dtype=mx.float32), (1, 4, 2))
        v_chunk = k_chunk + 100
        manager.write_prefill(
            seq_id=0, layer_idx=0, k_chunk=k_chunk, v_chunk=v_chunk, start_pos=0
        )
        tables, ctx = manager.batch_tables([0])

        q = mx.reshape(mx.arange(8, dtype=mx.float32), (1, 1, 4, 2))
        scale = 0.5

        base_lens = mx.array([0], dtype=mx.int32)
        reference = paged_kv._paged_prefill_reference(
            q,
            manager.k[0],
            manager.v[0],
            tables,
            ctx,
            scale=scale,
            base_lens=base_lens,
        )

        pieces = []
        for idx in range(4):
            lens_override = mx.array([idx + 1], dtype=mx.int32)
            piece = mx.fast.paged_attention(
                q[:, :, idx : idx + 1, :],
                manager.k[0],
                manager.v[0],
                tables,
                lens_override,
                scale=scale,
            )
            pieces.append(piece)
        expected = mx.concatenate(pieces, axis=2)

        np.testing.assert_allclose(
            np.array(reference), np.array(expected), rtol=1e-5, atol=1e-6
        )


class PagedPrefillDispatchTests(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        self._orig_native = getattr(paged_kv, "_NATIVE_PAGED_PREFILL", None)
        self._orig_entry = getattr(mx.fast, "paged_prefill", None)

    def tearDown(self):
        if self._orig_native is None:
            if hasattr(paged_kv, "_NATIVE_PAGED_PREFILL"):
                delattr(paged_kv, "_NATIVE_PAGED_PREFILL")
        else:
            paged_kv._NATIVE_PAGED_PREFILL = self._orig_native
        if self._orig_entry is None:
            if hasattr(mx.fast, "paged_prefill"):
                delattr(mx.fast, "paged_prefill")
        else:
            mx.fast.paged_prefill = self._orig_entry
        super().tearDown()

    def _make_inputs(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=4,
            dtype=mx.float32,
        )
        seq_id = 0
        manager.new_sequence(seq_id=seq_id, prompt_len=0)
        k_chunk = mx.reshape(mx.arange(8, dtype=mx.float32), (1, 4, 2))
        v_chunk = k_chunk + 50.0
        manager.write_prefill(
            seq_id=seq_id,
            layer_idx=0,
            k_chunk=k_chunk,
            v_chunk=v_chunk,
            start_pos=0,
        )
        block_tables, context_lens = manager.batch_tables([seq_id])
        base_lens = mx.array([0], dtype=mx.int32)
        q = mx.reshape(mx.arange(8, dtype=mx.float32), (1, 1, 4, 2))
        return q, manager.k[0], manager.v[0], block_tables, base_lens, context_lens

    def test_calls_native_impl_when_available(self):
        q, k_cache, v_cache, tables, base_lens, context_lens = self._make_inputs()
        captured = {}

        def _fake_native(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return mx.zeros_like(q)

        paged_kv._NATIVE_PAGED_PREFILL = _fake_native

        out = mx.fast.paged_prefill(
            q,
            k_cache,
            v_cache,
            tables,
            base_lens,
            context_lens,
            kv_head_mapping=None,
            scale=0.5,
        )

        self.assertTrue(mx.array_equal(out, mx.zeros_like(q)))
        self.assertIn("args", captured)
        self.assertIs(captured["args"][4], base_lens)
        self.assertIs(captured["args"][5], context_lens)


if __name__ == "__main__":
    unittest.main()
