# ABOUTME: Validates native paged prefill/attention kernels on GPU via subprocess.
# ABOUTME: Ensures Metal fast-paths run without crashing under minimal workloads.

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="mlx.core not available for paged tests")
mx_fast = pytest.importorskip(
    "mlx.core.fast", reason="mlx.core.fast not available for paged tests"
)


def _gpu_available() -> bool:
    try:
        device = mlx.default_device()
    except RuntimeError:
        return False
    device_type = getattr(device, "type", None)
    if device_type is None:
        return False
    name = getattr(device_type, "name", "")
    return name == "gpu"


@pytest.mark.skipif(
    not hasattr(mx_fast, "_paged_prefill_impl"), reason="native paged_prefill missing"
)
@pytest.mark.skipif(not _gpu_available(), reason="GPU device not available")
def test_paged_prefill_and_attention_do_not_crash(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("MLX_PAGED_DISABLE_NATIVE", None)
    pythonpath = str(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    script = textwrap.dedent(
        """
        import math
        import mlx.core as mx
        from mlx.nn.paged_kv import KVBlockManager

        mx.set_default_device(mx.gpu)
        with mx.stream(mx.gpu):
            manager = KVBlockManager(
                num_layers=1,
                num_kv_heads=1,
                head_dim=1,
                block_size=1,
                max_blocks=8,
                dtype=mx.float16,
            )
            seq_id = 1
            manager.new_sequence(seq_id, prompt_len=1)
            k_chunk = mx.ones(
                (manager.num_kv_heads, 1, manager.head_dim), dtype=mx.float16
            )
            v_chunk = mx.ones_like(k_chunk)
            manager.write_prefill(seq_id, 0, k_chunk, v_chunk, start_pos=0)
            block_tables, context_lens = manager.batch_tables([seq_id])
            base_lens = mx.zeros_like(context_lens)
            queries = mx.ones((1, 1, 1, 1), dtype=mx.float16)

            mx.fast._paged_prefill_impl(
                queries,
                manager.k,
                manager.v,
                block_tables,
                base_lens,
                context_lens,
                layer_idx=0,
                scale=1.0,
            )

            manager.commit_prefill(seq_id)
            decode_tables, decode_ctx = manager.batch_tables([seq_id])
            decode_q = mx.ones((1, 1, 1, 1), dtype=mx.float16)

            mx.fast._paged_attention_impl(
                decode_q,
                manager.k,
                manager.v,
                decode_tables,
                decode_ctx,
                layer_idx=0,
                scale=1.0,
            )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert (
        result.returncode == 0
    ), f"paged kernels crashed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.skipif(
    not hasattr(mx_fast, "_paged_attention_with_overlay_impl"),
    reason="overlay paged attention missing",
)
@pytest.mark.skipif(not _gpu_available(), reason="GPU device not available")
def test_paged_attention_overlay_matches_dense_write():
    mlx.set_default_device(mlx.gpu)
    batch = 2
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    block_size = 4
    max_blocks_per_seq = 1
    total_blocks = batch

    base_context = mlx.array([2, 1], dtype=mlx.int32)
    block_tables_np = [[0], [1]]
    block_tables = mlx.array(block_tables_np, dtype=mlx.int32)

    def make_cache_np():
        data = np.zeros(
            (num_kv_heads, total_blocks, block_size, head_dim), dtype=np.float16
        )
        for kv in range(num_kv_heads):
            for blk in range(total_blocks):
                for row in range(block_size):
                    for col in range(head_dim):
                        data[kv, blk, row, col] = (
                            (kv + 1) * 0.1 + blk * 0.01 + row * 0.001 + col * 0.0001
                        )
        return data

    k_cache_np = make_cache_np()
    v_cache_np = make_cache_np()
    k_cache = mlx.array(k_cache_np)
    v_cache = mlx.array(v_cache_np)

    overlay_k = mlx.arange(batch * num_kv_heads * head_dim, dtype=mlx.float16)
    overlay_k = overlay_k.reshape(batch, num_kv_heads, head_dim)
    overlay_v = overlay_k + 0.5

    # Build reference caches with the overlay row written into storage
    k_written_np = np.copy(k_cache_np)
    v_written_np = np.copy(v_cache_np)
    for b in range(batch):
        block_id = block_tables_np[b][0]
        row_idx = int(base_context[b].item())
        k_written_np[0, block_id, row_idx] = np.array(overlay_k[b, 0], dtype=np.float16)
        v_written_np[0, block_id, row_idx] = np.array(overlay_v[b, 0], dtype=np.float16)

    k_written = mlx.array(k_written_np)
    v_written = mlx.array(v_written_np)

    q = mlx.arange(batch * num_heads * head_dim, dtype=mlx.float16)
    q = q.reshape(batch, num_heads, 1, head_dim)

    reference = mx_fast._paged_attention_impl(
        q,
        k_written,
        v_written,
        block_tables,
        base_context + 1,
        layer_idx=0,
        scale=1.0,
    )

    overlay_out = mx_fast._paged_attention_with_overlay_impl(
        q,
        k_cache,
        v_cache,
        block_tables,
        base_context,
        layer_idx=0,
        k_overlay=overlay_k,
        v_overlay=overlay_v,
        scale=1.0,
    )

    assert mlx.allclose(reference, overlay_out, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(mx_fast, "_paged_attention_with_overlay_impl"),
    reason="overlay paged attention missing",
)
@pytest.mark.skipif(not _gpu_available(), reason="GPU device not available")
def test_paged_attention_overlay_multi_token_stack_matches_dense():
    mlx.set_default_device(mlx.gpu)
    batch = 2
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    block_size = 4
    max_blocks_per_seq = 1
    overlay_len = 2

    base_context = mlx.array([1, 2], dtype=mlx.int32)
    block_tables_np = [[0], [1]]
    block_tables = mlx.array(block_tables_np, dtype=mlx.int32)

    def _make_cache():
        data = np.zeros(
            (num_kv_heads, len(block_tables_np), block_size, head_dim),
            dtype=np.float16,
        )
        for kv in range(num_kv_heads):
            for blk in range(len(block_tables_np)):
                for row in range(block_size):
                    for col in range(head_dim):
                        data[kv, blk, row, col] = (
                            kv * 0.01 + blk * 0.001 + row * 0.0001 + col * 0.00001
                        )
        return data

    k_cache_np = _make_cache()
    v_cache_np = _make_cache()
    k_cache = mlx.array(k_cache_np)
    v_cache = mlx.array(v_cache_np)

    overlay_k = mlx.arange(
        overlay_len * batch * num_kv_heads * head_dim, dtype=mlx.float16
    ).reshape(overlay_len, batch, num_kv_heads, head_dim)
    overlay_v = overlay_k + 0.25

    # Build reference caches with overlay rows written sequentially.
    k_written = np.copy(k_cache_np)
    v_written = np.copy(v_cache_np)
    ref_context = np.array(base_context, dtype=np.int32)
    for step in range(overlay_len):
        for b in range(batch):
            logical = int(ref_context[b])
            block_idx = logical // block_size
            row = logical % block_size
            block_id = block_tables_np[b][block_idx]
            k_written[0, block_id, row] = np.array(
                overlay_k[step, b, 0], dtype=np.float16
            )
            v_written[0, block_id, row] = np.array(
                overlay_v[step, b, 0], dtype=np.float16
            )
            ref_context[b] += 1

    q = mlx.arange(batch * num_heads * head_dim, dtype=mlx.float16).reshape(
        batch, num_heads, 1, head_dim
    )

    reference = mx_fast._paged_attention_impl(
        q,
        mlx.array(k_written),
        mlx.array(v_written),
        block_tables,
        mlx.array(ref_context, dtype=mlx.int32),
        layer_idx=0,
        scale=1.0,
    )

    overlay_out = mx_fast._paged_attention_with_overlay_impl(
        q,
        k_cache,
        v_cache,
        block_tables,
        base_context,
        layer_idx=0,
        k_overlay=overlay_k,
        v_overlay=overlay_v,
        scale=1.0,
    )

    assert mlx.allclose(reference, overlay_out, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(mx_fast, "_paged_attention_with_overlay_impl"),
    reason="overlay paged attention missing",
)
@pytest.mark.skipif(not _gpu_available(), reason="GPU device not available")
def test_paged_attention_overlay_compiles_with_mx_compile():
    mlx.set_default_device(mlx.gpu)
    batch = 1
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    block_size = 2
    max_blocks_per_seq = 1

    block_tables = mlx.zeros((batch, max_blocks_per_seq), dtype=mlx.int32)
    context = mlx.ones((batch,), dtype=mlx.int32)

    k_cache = mlx.arange(
        num_kv_heads * max_blocks_per_seq * block_size * head_dim, dtype=mlx.float16
    ).reshape(num_kv_heads, max_blocks_per_seq, block_size, head_dim)
    v_cache = k_cache + 0.25
    overlay_k = mlx.full((batch, num_kv_heads, head_dim), 0.5, dtype=mlx.float16)
    overlay_v = overlay_k + 0.5
    queries = mlx.arange(batch * num_heads * head_dim, dtype=mlx.float16).reshape(
        batch, num_heads, 1, head_dim
    )

    def decode(
        q,
        k_pool,
        v_pool,
        tables,
        ctx,
        overlay_key,
        overlay_value,
    ):
        return mx_fast._paged_attention_with_overlay_impl(
            q,
            k_pool,
            v_pool,
            tables,
            ctx,
            layer_idx=0,
            scale=1.0,
            k_overlay=overlay_key,
            v_overlay=overlay_value,
        )

    compiled = mlx.compile(decode)

    eager_out = decode(
        queries,
        k_cache,
        v_cache,
        block_tables,
        context,
        overlay_k,
        overlay_v,
    )
    compiled_out = compiled(
        queries,
        k_cache,
        v_cache,
        block_tables,
        context,
        overlay_k,
        overlay_v,
    )

    compiled_out_second = compiled(
        queries * 1.1,
        k_cache,
        v_cache,
        block_tables,
        context,
        overlay_k,
        overlay_v,
    )

    assert mlx.allclose(eager_out, compiled_out, atol=1e-4)
    assert compiled_out_second.shape == eager_out.shape
