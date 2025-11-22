# ABOUTME: Validates paged KV allocator semantics and copy-on-write behavior.
# ABOUTME: Asserts paged attention fallback matches dense attention outputs.

import math
import unittest

import mlx.core as mx
import mlx_tests


def _dense_reference(q, dense_k, dense_v, context_lens, scale):
    batch, num_heads, _, _ = q.shape
    num_kv_heads = dense_k.shape[1]
    out = mx.zeros_like(q)
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        if seq_len == 0:
            continue
        for h in range(num_heads):
            kv_head = h if num_kv_heads > 1 else 0
            k_head = dense_k[b, kv_head, :seq_len]
            v_head = dense_v[b, kv_head, :seq_len]
            q_vec = q[b, h, 0]
            scores = mx.sum(q_vec * k_head, axis=-1) * scale
            weights = mx.softmax(scores.astype(mx.float32))
            out[b, h, 0] = mx.sum(mx.expand_dims(weights, axis=-1) * v_head, axis=0)
    return out


class TestKVBlockManager(mlx_tests.MLXTestCase):
    def test_new_sequence_allocates_expected_blocks(self):
        from mlx.nn.paged_kv import KVBlockManager

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
        from mlx.nn.paged_kv import KVBlockManager

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
        from mlx.nn.paged_kv import KVBlockManager

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


if __name__ == "__main__":
    unittest.main()
