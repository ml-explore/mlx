# ABOUTME: Validates paged KV allocator semantics and copy-on-write behavior.
# ABOUTME: Asserts paged attention fallback matches dense attention outputs.

import math
import unittest

import mlx.core as mx
import mlx_tests
from mlx.nn.paged_kv import KVBlockManager, prefill_into_kv


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
    k_cache = mx.zeros((1, kv_heads, total_blocks, block_size, head_dim), dtype=dense_k.dtype)
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
            mx.array_equal(k_store[first_block, 0], mx.array([0, 1, 2, 3], dtype=mx.float16))
        )
        self.assertTrue(
            mx.array_equal(k_store[first_block, 1], mx.array([4, 5, 6, 7], dtype=mx.float16))
        )
        self.assertTrue(
            mx.array_equal(k_store[second_block, 0], mx.array([8, 9, 10, 11], dtype=mx.float16))
        )
        self.assertTrue(
            mx.array_equal(
                v_store[first_block, 0], mx.array([100, 101, 102, 103], dtype=mx.float16)
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
        manager.append_decode_token(seq_id=1, layer_idx=0, k_token=k_token, v_token=v_token)

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
        self.assertEqual(int(tables[0, 0].item()), manager.table(1)[0][0].item())


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
                self.assertTrue(mx.allclose(paged_out, dense_out, rtol=1e-3, atol=1e-3))

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
        self.assertTrue(mx.allclose(paged_out, dense_out, rtol=1e-3, atol=1e-3))

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
        self.assertTrue(mx.allclose(paged_out, dense_out, rtol=1e-3, atol=1e-3))


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



if __name__ == "__main__":
    unittest.main()
