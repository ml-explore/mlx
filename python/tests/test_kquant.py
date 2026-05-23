# Copyright © 2026 Apple Inc.

import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np

Q8_0_GROUP = 32
Q8_0_BLOCK_BYTES = 34  # fp16 d (2) + int8 q[32] (32)
Q8_0_D_OFFSET = 0
Q8_0_Q_OFFSET = 2


def _quantize_q8_0_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array (length must be a multiple of 32) to Q8_0
    packed wire bytes. Returns uint8 array of length len(row) * 34/32."""
    assert row.ndim == 1
    assert row.size % Q8_0_GROUP == 0, "Q8_0 requires K % 32 == 0"
    n_blocks = row.size // Q8_0_GROUP
    out = np.zeros(n_blocks * Q8_0_BLOCK_BYTES, dtype=np.uint8)
    blocks = row.reshape(n_blocks, Q8_0_GROUP).astype(np.float32)
    for b in range(n_blocks):
        block = blocks[b]
        amax = float(np.max(np.abs(block)))
        if amax == 0.0:
            d = np.float32(0.0)
            q = np.zeros(Q8_0_GROUP, dtype=np.int8)
        else:
            d = np.float32(amax / 127.0)
            q = np.clip(np.round(block / d), -127, 127).astype(np.int8)
        d_fp16 = np.float16(d)
        base = b * Q8_0_BLOCK_BYTES
        out[base + Q8_0_D_OFFSET : base + Q8_0_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q8_0_Q_OFFSET : base + Q8_0_Q_OFFSET + Q8_0_GROUP] = np.frombuffer(
            q.tobytes(), dtype=np.uint8
        )
    return out


def _quantize_q8_0_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % Q8_0_GROUP == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q8_0_BLOCK_BYTES // Q8_0_GROUP
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q8_0_row(W[i])
    return out


def _dequantize_q8_0_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q8_0_BLOCK_BYTES // Q8_0_GROUP
    n_blocks_per_row = in_dim // Q8_0_GROUP
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        row = W_q[i]
        for b in range(n_blocks_per_row):
            base = b * Q8_0_BLOCK_BYTES
            d_fp16 = np.frombuffer(
                row[base + Q8_0_D_OFFSET : base + Q8_0_D_OFFSET + 2].tobytes(),
                dtype=np.float16,
            )[0]
            q_int8 = np.frombuffer(
                row[base + Q8_0_Q_OFFSET : base + Q8_0_Q_OFFSET + Q8_0_GROUP].tobytes(),
                dtype=np.int8,
            )
            out[i, b * Q8_0_GROUP : (b + 1) * Q8_0_GROUP] = float(
                d_fp16
            ) * q_int8.astype(np.float32)
    return out


def _scales_placeholder() -> mx.array:
    """KQuant accepts a placeholder scales tensor; pass a 1-byte dummy."""
    return mx.zeros((1,), dtype=mx.uint8)


def _kquant_matmul(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q8_0 (gs=32, bits=8)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="kquant",
        kquant_type="q8_0",
    )


Q4_0_GROUP = 32
Q4_0_BLOCK_BYTES = 18  # fp16 d (2) + uint8 qs[16]
Q4_0_D_OFFSET = 0
Q4_0_QS_OFFSET = 2


def _quantize_q4_0_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array (length must be a multiple of 32) to Q4_0
    packed wire bytes. Returns uint8 array of length len(row) * 18/32.

    Symmetric quantization: scale = amax / -8.0;
    q = round(weight / scale) + 8 clipped to [0, 15]; pack split-half.
    """
    assert row.ndim == 1
    assert row.size % Q4_0_GROUP == 0, "Q4_0 requires K % 32 == 0"
    n_blocks = row.size // Q4_0_GROUP
    out = np.zeros(n_blocks * Q4_0_BLOCK_BYTES, dtype=np.uint8)
    blocks = row.reshape(n_blocks, Q4_0_GROUP).astype(np.float32)
    for b in range(n_blocks):
        block = blocks[b]
        # Use the value with largest magnitude (signed) -- pick the entry
        # with greatest |x|, keep its sign for the scale.
        idx = int(np.argmax(np.abs(block)))
        amax_signed = float(block[idx])
        if amax_signed == 0.0:
            d = np.float32(0.0)
            q = np.full(Q4_0_GROUP, 8, dtype=np.uint8)
        else:
            d = np.float32(amax_signed / -8.0)
            inv_d = 1.0 / float(d)
            q = np.clip(np.round(block * inv_d).astype(np.int32) + 8, 0, 15).astype(
                np.uint8
            )
        # Split-half pack: qs[j] = q[j] | (q[j+16] << 4) for j in [0, 16).
        qs = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            qs[j] = np.uint8((int(q[j]) & 0x0F) | ((int(q[j + 16]) & 0x0F) << 4))
        d_fp16 = np.float16(d)
        base = b * Q4_0_BLOCK_BYTES
        out[base + Q4_0_D_OFFSET : base + Q4_0_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q4_0_QS_OFFSET : base + Q4_0_QS_OFFSET + 16] = qs
    return out


def _quantize_q4_0_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % Q4_0_GROUP == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q4_0_BLOCK_BYTES // Q4_0_GROUP
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q4_0_row(W[i])
    return out


def _dequantize_q4_0_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q4_0 dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q4_0_BLOCK_BYTES // Q4_0_GROUP
    n_blocks_per_row = in_dim // Q4_0_GROUP
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        row = W_q[i]
        for b in range(n_blocks_per_row):
            base = b * Q4_0_BLOCK_BYTES
            d = float(
                np.frombuffer(
                    row[base + Q4_0_D_OFFSET : base + Q4_0_D_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            qs = np.frombuffer(
                row[base + Q4_0_QS_OFFSET : base + Q4_0_QS_OFFSET + 16].tobytes(),
                dtype=np.uint8,
            )
            for j in range(16):
                x0 = (int(qs[j]) & 0x0F) - 8
                x1 = (int(qs[j]) >> 4) - 8
                out[i, b * Q4_0_GROUP + j] = d * x0
                out[i, b * Q4_0_GROUP + j + 16] = d * x1
    return out


def _kquant_matmul_q4_0(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q4_0 (gs=32, bits=4)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="kquant",
        kquant_type="q4_0",
    )


Q4_1_GROUP = 32
Q4_1_BLOCK_BYTES = 20  # fp16 d (2) + fp16 m (2) + uint8 qs[16]
Q4_1_D_OFFSET = 0
Q4_1_M_OFFSET = 2
Q4_1_QS_OFFSET = 4


def _quantize_q4_1_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array to Q4_1 packed wire bytes (asymmetric, 4-bit)."""
    assert row.ndim == 1
    assert row.size % Q4_1_GROUP == 0, "Q4_1 requires K % 32 == 0"
    n_blocks = row.size // Q4_1_GROUP
    out = np.zeros(n_blocks * Q4_1_BLOCK_BYTES, dtype=np.uint8)
    blocks = row.reshape(n_blocks, Q4_1_GROUP).astype(np.float32)
    for b in range(n_blocks):
        block = blocks[b]
        mn = float(block.min())
        mx_ = float(block.max())
        if mx_ == mn:
            d = np.float32(0.0)
            m = np.float32(mn)
            q = np.zeros(Q4_1_GROUP, dtype=np.uint8)
        else:
            d = np.float32((mx_ - mn) / 15.0)
            m = np.float32(mn)
            q = np.clip(np.round((block - mn) / d), 0, 15).astype(np.uint8)
        qs = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            qs[j] = np.uint8((int(q[j]) & 0x0F) | ((int(q[j + 16]) & 0x0F) << 4))
        d_fp16 = np.float16(d)
        m_fp16 = np.float16(m)
        base = b * Q4_1_BLOCK_BYTES
        out[base + Q4_1_D_OFFSET : base + Q4_1_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q4_1_M_OFFSET : base + Q4_1_M_OFFSET + 2] = np.frombuffer(
            m_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q4_1_QS_OFFSET : base + Q4_1_QS_OFFSET + 16] = qs
    return out


def _quantize_q4_1_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % Q4_1_GROUP == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q4_1_BLOCK_BYTES // Q4_1_GROUP
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q4_1_row(W[i])
    return out


def _dequantize_q4_1_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q4_1 dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q4_1_BLOCK_BYTES // Q4_1_GROUP
    n_blocks_per_row = in_dim // Q4_1_GROUP
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        row = W_q[i]
        for b in range(n_blocks_per_row):
            base = b * Q4_1_BLOCK_BYTES
            d = float(
                np.frombuffer(
                    row[base + Q4_1_D_OFFSET : base + Q4_1_D_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            m = float(
                np.frombuffer(
                    row[base + Q4_1_M_OFFSET : base + Q4_1_M_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            qs = np.frombuffer(
                row[base + Q4_1_QS_OFFSET : base + Q4_1_QS_OFFSET + 16].tobytes(),
                dtype=np.uint8,
            )
            for j in range(16):
                x0 = int(qs[j]) & 0x0F
                x1 = int(qs[j]) >> 4
                out[i, b * Q4_1_GROUP + j] = d * x0 + m
                out[i, b * Q4_1_GROUP + j + 16] = d * x1 + m
    return out


def _kquant_matmul_q4_1(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="kquant",
        kquant_type="q4_1",
    )


Q5_0_GROUP = 32
Q5_0_BLOCK_BYTES = 22  # fp16 d (2) + uint8 qh[4] + uint8 qs[16]
Q5_0_D_OFFSET = 0
Q5_0_QH_OFFSET = 2
Q5_0_QS_OFFSET = 6


def _quantize_q5_0_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array to Q5_0 packed wire bytes (symmetric, 5-bit)."""
    assert row.ndim == 1
    assert row.size % Q5_0_GROUP == 0, "Q5_0 requires K % 32 == 0"
    n_blocks = row.size // Q5_0_GROUP
    out = np.zeros(n_blocks * Q5_0_BLOCK_BYTES, dtype=np.uint8)
    blocks = row.reshape(n_blocks, Q5_0_GROUP).astype(np.float32)
    for b in range(n_blocks):
        block = blocks[b]
        # Scale from the entry with greatest |x|; range is [-16, 15].
        idx = int(np.argmax(np.abs(block)))
        amax_signed = float(block[idx])
        if amax_signed == 0.0:
            d = np.float32(0.0)
            q = np.full(Q5_0_GROUP, 16, dtype=np.uint8)
        else:
            d = np.float32(amax_signed / -16.0)
            inv_d = 1.0 / float(d)
            q = np.clip(np.round(block * inv_d).astype(np.int32) + 16, 0, 31).astype(
                np.uint8
            )
        # qh: bit j (j<32) = (q[j] >> 4) & 1, but packed into the 4-byte
        # layout shared with Q5_1: low 16 bits hold high-bits of weights
        # 0..15 (bit j); high 16 bits hold high-bits of weights 16..31
        # (bit (j-16)+16 = j matches the dequant ((qh >> j) << 4) & 0x10
        # for the low half and ((qh >> (j+12))) & 0x10 for the high half).
        qh = np.uint32(0)
        for j in range(16):
            qh |= np.uint32((int(q[j]) >> 4) & 1) << j
        for j in range(16):
            qh |= np.uint32((int(q[j + 16]) >> 4) & 1) << (j + 16)
        # qs: low 4 bits of q[j] in low nibble of qs[j]; q[j+16] in high nibble.
        qs = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            qs[j] = np.uint8((int(q[j]) & 0x0F) | ((int(q[j + 16]) & 0x0F) << 4))
        d_fp16 = np.float16(d)
        base = b * Q5_0_BLOCK_BYTES
        out[base + Q5_0_D_OFFSET : base + Q5_0_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q5_0_QH_OFFSET : base + Q5_0_QH_OFFSET + 4] = np.frombuffer(
            qh.tobytes(), dtype=np.uint8
        )
        out[base + Q5_0_QS_OFFSET : base + Q5_0_QS_OFFSET + 16] = qs
    return out


def _quantize_q5_0_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % Q5_0_GROUP == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q5_0_BLOCK_BYTES // Q5_0_GROUP
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q5_0_row(W[i])
    return out


def _dequantize_q5_0_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q5_0 dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q5_0_BLOCK_BYTES // Q5_0_GROUP
    n_blocks_per_row = in_dim // Q5_0_GROUP
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        row = W_q[i]
        for b in range(n_blocks_per_row):
            base = b * Q5_0_BLOCK_BYTES
            d = float(
                np.frombuffer(
                    row[base + Q5_0_D_OFFSET : base + Q5_0_D_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            qh_bytes = row[base + Q5_0_QH_OFFSET : base + Q5_0_QH_OFFSET + 4]
            qh = (
                int(qh_bytes[0])
                | (int(qh_bytes[1]) << 8)
                | (int(qh_bytes[2]) << 16)
                | (int(qh_bytes[3]) << 24)
            )
            qs = np.frombuffer(
                row[base + Q5_0_QS_OFFSET : base + Q5_0_QS_OFFSET + 16].tobytes(),
                dtype=np.uint8,
            )
            for j in range(16):
                xh_0 = ((qh >> j) << 4) & 0x10
                xh_1 = ((qh >> (j + 12))) & 0x10
                x0 = (int(qs[j]) & 0x0F) | xh_0
                x1 = (int(qs[j]) >> 4) | xh_1
                out[i, b * Q5_0_GROUP + j] = d * (x0 - 16)
                out[i, b * Q5_0_GROUP + j + 16] = d * (x1 - 16)
    return out


def _kquant_matmul_q5_0(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=32,
        bits=5,
        mode="kquant",
        kquant_type="q5_0",
    )


Q5_1_GROUP = 32
Q5_1_BLOCK_BYTES = 24  # fp16 d (2) + fp16 m (2) + uint32 qh (4) + uint8 qs[16]
Q5_1_D_OFFSET = 0
Q5_1_M_OFFSET = 2
Q5_1_QH_OFFSET = 4
Q5_1_QS_OFFSET = 8


def _quantize_q5_1_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array (length must be a multiple of 32) to Q5_1
    packed wire bytes. Returns uint8 array of length len(row) * 24/32."""
    assert row.ndim == 1
    assert row.size % Q5_1_GROUP == 0, "Q5_1 requires K % 32 == 0"
    n_blocks = row.size // Q5_1_GROUP
    out = np.zeros(n_blocks * Q5_1_BLOCK_BYTES, dtype=np.uint8)
    blocks = row.reshape(n_blocks, Q5_1_GROUP).astype(np.float32)
    for b in range(n_blocks):
        block = blocks[b]
        mn = float(block.min())
        mx_ = float(block.max())
        if mx_ == mn:
            d = np.float32(0.0)
            m = np.float32(mn)
            q = np.zeros(Q5_1_GROUP, dtype=np.uint8)
        else:
            d = np.float32((mx_ - mn) / 31.0)
            m = np.float32(mn)
            q = np.clip(np.round((block - mn) / d), 0, 31).astype(np.uint8)
        # Pack 5th bit of each weight into qh (uint32, bit j = (q[j] >> 4) & 1).
        qh = np.uint32(0)
        for j in range(Q5_1_GROUP):
            qh |= np.uint32((int(q[j]) >> 4) & 1) << j
        # Pack low 4 bits: qs[j] = (q[j] & 0xF) | ((q[j+16] & 0xF) << 4).
        qs = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            qs[j] = np.uint8((int(q[j]) & 0x0F) | ((int(q[j + 16]) & 0x0F) << 4))
        d_fp16 = np.float16(d)
        m_fp16 = np.float16(m)
        base = b * Q5_1_BLOCK_BYTES
        out[base + Q5_1_D_OFFSET : base + Q5_1_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q5_1_M_OFFSET : base + Q5_1_M_OFFSET + 2] = np.frombuffer(
            m_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q5_1_QH_OFFSET : base + Q5_1_QH_OFFSET + 4] = np.frombuffer(
            qh.tobytes(), dtype=np.uint8
        )
        out[base + Q5_1_QS_OFFSET : base + Q5_1_QS_OFFSET + 16] = qs
    return out


def _quantize_q5_1_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % Q5_1_GROUP == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q5_1_BLOCK_BYTES // Q5_1_GROUP
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q5_1_row(W[i])
    return out


def _dequantize_q5_1_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q5_1 dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q5_1_BLOCK_BYTES // Q5_1_GROUP
    n_blocks_per_row = in_dim // Q5_1_GROUP
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        row = W_q[i]
        for b in range(n_blocks_per_row):
            base = b * Q5_1_BLOCK_BYTES
            d = float(
                np.frombuffer(
                    row[base + Q5_1_D_OFFSET : base + Q5_1_D_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            m = float(
                np.frombuffer(
                    row[base + Q5_1_M_OFFSET : base + Q5_1_M_OFFSET + 2].tobytes(),
                    dtype=np.float16,
                )[0]
            )
            qh = int(
                np.frombuffer(
                    row[base + Q5_1_QH_OFFSET : base + Q5_1_QH_OFFSET + 4].tobytes(),
                    dtype=np.uint32,
                )[0]
            )
            qs = np.frombuffer(
                row[base + Q5_1_QS_OFFSET : base + Q5_1_QS_OFFSET + 16].tobytes(),
                dtype=np.uint8,
            )
            for j in range(16):
                xh_0 = ((qh >> j) << 4) & 0x10
                xh_1 = ((qh >> (j + 12))) & 0x10
                x0 = (int(qs[j]) & 0x0F) | xh_0
                x1 = (int(qs[j]) >> 4) | xh_1
                out[i, b * Q5_1_GROUP + j] = d * x0 + m
                out[i, b * Q5_1_GROUP + j + 16] = d * x1 + m
    return out


def _kquant_matmul_q5_1(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q5_1 (gs=32, bits=5)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=32,
        bits=5,
        mode="kquant",
        kquant_type="q5_1",
    )


# Reasonable per-dtype tolerances for Q8_0 matmul vs fp32 reference.
# Q8_0 itself is <=1/127 relative quantization error per block; fp16
# accumulation in the kernel adds another ~1e-4 relative.
_MATMUL_REL_TOL = {
    mx.float32: 1e-4,
    mx.float16: 5e-3,
    mx.bfloat16: 5e-2,  # bf16 has ~7-bit mantissa -> looser tolerance
}


class _KQuantCodecTestMixin:
    """Shared test helpers and standard test methods for K-quant codecs.

    Concrete test classes inherit from this mixin and set the following
    class attributes to wire up codec-specific quantize/dequantize/matmul:

        quantize_matrix   -- staticmethod: (W: np.ndarray) -> np.ndarray
        dequantize_matrix -- staticmethod: (W_q: np.ndarray, in_dim: int) -> np.ndarray
        matmul_fn         -- staticmethod: (x: mx.array, W_q: np.ndarray) -> mx.array
        group_size        -- int (32 or 256)
        bits              -- int
        block_bytes       -- int (wire-format bytes per block)
        kquant_type       -- str (e.g. "q8_0", "q4_k")
    """

    # Default dimensions -- overridden by gs=256 codecs.
    general_out_dim = 64
    general_in_dim = 768
    qmm_t_shapes = ((4, 64, 1024), (17, 48, 1024), (64, 64, 2048))
    qmm_n_shapes = ((8, 64, 1024), (17, 64, 1024))

    def _check_dequant_via_one_hot(self, out_dim, in_dim, dtype):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.5
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, in_dim)
        cols = [0, 1, in_dim // 2, in_dim - 1]
        for col in cols:
            x_np = np.zeros(in_dim, dtype=np.float32)
            x_np[col] = 1.0
            x = mx.array(x_np)
            if dtype != mx.float32:
                x = x.astype(dtype)
            y = self.matmul_fn(x, W_q)
            mx.eval(y)
            y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
            ref = W_ref[:, col]
            denom = max(1e-8, float(np.max(np.abs(ref))))
            rel = float(np.max(np.abs(y_np - ref))) / denom
            self.assertLess(
                rel,
                1e-2,
                msg=f"dtype={dtype} col={col}: rel={rel:.3e}",
            )

    def _check_random_matmul(self, out_dim, in_dim, dtype):
        rng = np.random.default_rng(7)
        W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.3
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, in_dim)
        x_np = rng.standard_normal((in_dim,)).astype(np.float32)
        ref = W_ref @ x_np

        x = mx.array(x_np)
        if dtype != mx.float32:
            x = x.astype(dtype)
        y = self.matmul_fn(x, W_q)
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)

        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
        tol = _MATMUL_REL_TOL[out_dtype]
        self.assertLess(
            rel,
            tol,
            msg=f"out={out_dim} in={in_dim} dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
        )

    def _check_qmm_t(self, M, N, K, dtype):
        rng = np.random.default_rng(7)
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, K)
        X_np = rng.standard_normal((M, K)).astype(np.float32)
        ref = X_np @ W_ref.T
        x = mx.array(X_np)
        if dtype != mx.float32:
            x = x.astype(dtype)
        y = self.matmul_fn(x, W_q)
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
        tol = _MATMUL_REL_TOL[out_dtype]
        self.assertLess(
            rel,
            tol,
            msg=f"M={M} N={N} K={K} dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
        )

    def _check_qmm_n(self, M, N, K, dtype):
        rng = np.random.default_rng(7)
        W = rng.standard_normal((K, N)).astype(np.float32) * 0.3
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, N)
        X_np = rng.standard_normal((M, K)).astype(np.float32)
        ref = X_np @ W_ref
        x = mx.array(X_np)
        if dtype != mx.float32:
            x = x.astype(dtype)
        y = mx.quantized_matmul(
            x,
            mx.array(W_q),
            scales=_scales_placeholder(),
            biases=None,
            transpose=False,
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
        tol = _MATMUL_REL_TOL[out_dtype]
        self.assertLess(
            rel,
            tol,
            msg=f"M={M} N={N} K={K} dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
        )

    def test_dequantize_via_one_hot(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_dequant_via_one_hot(8, 1024, dtype)

    def test_dequantize(self):
        """mx.dequantize(mode='kquant') reproduces the codec reference."""
        rng = np.random.default_rng(42)
        N, K = 64, 1024
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, K)
        # Per-element cast precision: fp32 is exact, fp16 keeps ~10 bits,
        # bf16 keeps ~7 bits of mantissa.
        tol_by_dtype = {
            mx.float32: 1e-4,
            mx.float16: 1e-3,
            mx.bfloat16: 5e-3,
        }
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                y = mx.dequantize(
                    mx.array(W_q),
                    scales=_scales_placeholder(),
                    biases=None,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode="kquant",
                    kquant_type=self.kquant_type,
                    dtype=dtype,
                )
                mx.eval(y)
                y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                denom = max(1e-8, float(np.max(np.abs(W_ref))))
                rel = float(np.max(np.abs(y_np - W_ref))) / denom
                tol = tol_by_dtype[dtype]
                self.assertLess(
                    rel,
                    tol,
                    msg=f"dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
                )

    def test_random_matmul_fast_path(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_random_matmul(64, 1024, dtype)

    def test_random_matmul_general_path(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_random_matmul(
                    self.general_out_dim, self.general_in_dim, dtype
                )

    def test_random_matmul_small_n(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_random_matmul(8, 1024, dtype)

    def test_qmm_t(self):
        for M, N, K in self.qmm_t_shapes:
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(M=M, N=N, K=K, dtype=dtype):
                    self._check_qmm_t(M, N, K, dtype)

    def test_qmm_n(self):
        for M, N, K in self.qmm_n_shapes:
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(M=M, N=N, K=K, dtype=dtype):
                    self._check_qmm_n(M, N, K, dtype)

    def test_qmm_n_large_k(self):
        """transpose=False with K>=2048 exercises splitk/large-reduction paths."""
        K = 2048
        N = max(256, self.group_size)
        for M in (1, 8):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(M=M, N=N, K=K, dtype=dtype):
                    self._check_qmm_n(M, N, K, dtype)

    def test_non_multiple_of_8_output_dim(self):
        """Non-aligned N (not divisible by 8) exercises edge-tile handling.

        Only tests transpose=True (qmv and qmm_t) where N is the output dim.
        For transpose=False, N is the quantized axis and must be block-aligned.
        """
        K = 1024
        for N in (13, 33):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(N=N, dtype=dtype):
                    self._check_qmm_t(1, N, K, dtype)
                    self._check_qmm_t(10, N, K, dtype)

    def test_gather_qmm(self):
        """Indexed transpose=True matmul (MoE): y[b] = x[lhs[b]] @ W[rhs[b]].T.

        Exercises both gather_qmv (M=1) and gather_qmm_t (M=32) paths.
        """
        rng = np.random.default_rng(42)
        E = 4  # number of experts
        B = 8  # batch positions
        K = 1024  # multiple of both 32 and 256
        N = 64
        Ws_q = []
        Ws_ref = []
        for _ in range(E):
            W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
            W_q = self.quantize_matrix(W)
            Ws_q.append(W_q)
            Ws_ref.append(self.dequantize_matrix(W_q, K))
        w_stacked = np.stack(Ws_q, axis=0)  # (E, N, bytes_per_row)
        rhs_idx = rng.integers(0, E, size=B).astype(np.uint32)
        lhs_idx = np.arange(B, dtype=np.uint32)
        for M in (1, 32):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(M=M, dtype=dtype):
                    X = rng.standard_normal((B, M, K)).astype(np.float32)
                    ref = np.zeros((B, M, N), dtype=np.float32)
                    for b in range(B):
                        ref[b] = X[b] @ Ws_ref[rhs_idx[b]].T
                    x = mx.array(X)
                    if dtype != mx.float32:
                        x = x.astype(dtype)
                    y = mx.gather_qmm(
                        x,
                        mx.array(w_stacked),
                        scales=mx.zeros((E, 1, 1), dtype=mx.uint8),
                        biases=None,
                        lhs_indices=mx.array(lhs_idx),
                        rhs_indices=mx.array(rhs_idx),
                        transpose=True,
                        group_size=self.group_size,
                        bits=self.bits,
                        mode="kquant",
                        kquant_type=self.kquant_type,
                    )
                    mx.eval(y)
                    y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                    denom = max(1e-8, float(np.max(np.abs(ref))))
                    rel = float(np.max(np.abs(y_np - ref))) / denom
                    out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
                    tol = _MATMUL_REL_TOL[out_dtype]
                    self.assertLess(
                        rel,
                        tol,
                        msg=f"M={M} dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
                    )

    def test_gather_qvm(self):
        """Indexed transpose=False matmul (MoE): y[b] = x[lhs[b]] @ W[rhs[b]].

        Exercises the gather_qmm_n path for KQuant (transpose=False, small M).
        """
        rng = np.random.default_rng(43)
        E = 4
        B = 8
        K = 256
        N = 512
        Ws_q = []
        Ws_ref = []
        for _ in range(E):
            W = rng.standard_normal((K, N)).astype(np.float32) * 0.3
            W_q = self.quantize_matrix(W)
            Ws_q.append(W_q)
            Ws_ref.append(self.dequantize_matrix(W_q, N))
        w_stacked = np.stack(Ws_q, axis=0)
        rhs_idx = rng.integers(0, E, size=B).astype(np.uint32)
        lhs_idx = np.arange(B, dtype=np.uint32)
        for M in (1, 2):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(M=M, dtype=dtype):
                    X = rng.standard_normal((B, M, K)).astype(np.float32)
                    ref = np.zeros((B, M, N), dtype=np.float32)
                    for b in range(B):
                        ref[b] = X[b] @ Ws_ref[rhs_idx[b]]
                    x = mx.array(X)
                    if dtype != mx.float32:
                        x = x.astype(dtype)
                    y = mx.gather_qmm(
                        x,
                        mx.array(w_stacked),
                        scales=mx.zeros((E, 1, 1), dtype=mx.uint8),
                        biases=None,
                        lhs_indices=mx.array(lhs_idx),
                        rhs_indices=mx.array(rhs_idx),
                        transpose=False,
                        group_size=self.group_size,
                        bits=self.bits,
                        mode="kquant",
                        kquant_type=self.kquant_type,
                    )
                    mx.eval(y)
                    y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                    denom = max(1e-8, float(np.max(np.abs(ref))))
                    rel = float(np.max(np.abs(y_np - ref))) / denom
                    out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
                    tol = _MATMUL_REL_TOL[out_dtype]
                    self.assertLess(
                        rel,
                        tol,
                        msg=f"M={M} dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
                    )

    def test_batched_qmm(self):
        """Batched (B>1) matmul with 3D weight: out[b] = x[b] @ W[b].T (or @ W[b]).

        Exercises the batched dispatch path (non_batched=false, B>1) across
        the qmv decode (M=1), qmm_t prefill (M=64), and qmm_n (transpose=False)
        kernels. K=1024 is divisible by every codec's group size.
        """
        rng = np.random.default_rng(13)
        B = 3
        K = 1024
        # For transpose=False the inner dim of the call is the second axis of
        # the (out, in) quantized weight; that must be a multiple of group_size.
        N_t = 128 if self.group_size <= 32 else self.group_size
        N_n = max(128, self.group_size)

        cases = [
            (1, True, N_t),  # qmv decode path
            (64, True, N_t),  # qmm_t prefill path (M >= vector_limit)
            (4, False, N_n),  # qmm_n path (transpose=False, small M)
        ]

        for M, transpose, N in cases:
            for dtype in (mx.float32, mx.bfloat16):
                with self.subTest(M=M, transpose=transpose, N=N, dtype=dtype):
                    Ws_q = []
                    Ws_ref = []
                    for _ in range(B):
                        if transpose:
                            W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
                            inner = K
                        else:
                            W = rng.standard_normal((K, N)).astype(np.float32) * 0.3
                            inner = N
                        W_q = self.quantize_matrix(W)
                        Ws_q.append(W_q)
                        Ws_ref.append(self.dequantize_matrix(W_q, inner))
                    w_stacked = np.stack(Ws_q, axis=0)

                    X = rng.standard_normal((B, M, K)).astype(np.float32)
                    ref = np.zeros((B, M, N), dtype=np.float32)
                    for b in range(B):
                        if transpose:
                            ref[b] = X[b] @ Ws_ref[b].T
                        else:
                            ref[b] = X[b] @ Ws_ref[b]

                    x = mx.array(X)
                    if dtype != mx.float32:
                        x = x.astype(dtype)
                    y = mx.quantized_matmul(
                        x,
                        mx.array(w_stacked),
                        scales=mx.zeros((B, 1, 1), dtype=mx.uint8),
                        biases=None,
                        transpose=transpose,
                        group_size=self.group_size,
                        bits=self.bits,
                        mode="kquant",
                        kquant_type=self.kquant_type,
                    )
                    mx.eval(y)
                    y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                    denom = max(1e-8, float(np.max(np.abs(ref))))
                    rel = float(np.max(np.abs(y_np - ref))) / denom
                    out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
                    tol = _MATMUL_REL_TOL[out_dtype]
                    self.assertLess(
                        rel,
                        tol,
                        msg=(
                            f"M={M} transpose={transpose} N={N} dtype={dtype}: "
                            f"rel={rel:.3e} tol={tol:.0e}"
                        ),
                    )

    def test_partial_block_weight_rejected(self):
        bad_w = mx.zeros((4, self.block_bytes // 2), dtype=mx.uint8)
        x = mx.zeros((1, self.group_size // 2), dtype=mx.float16)
        scales = _scales_placeholder()
        with self.assertRaisesRegex((RuntimeError, ValueError), "whole number of"):
            mx.quantized_matmul(
                x,
                bad_w,
                scales=scales,
                biases=None,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode="kquant",
                kquant_type=self.kquant_type,
            )

    def test_wrong_weight_dtype_rejected(self):
        w = mx.zeros((4, self.block_bytes), dtype=mx.uint32)
        x = mx.zeros((1, self.group_size), dtype=mx.float16)
        scales = _scales_placeholder()
        with self.assertRaisesRegex((RuntimeError, ValueError), "must be uint8"):
            mx.quantized_matmul(
                x,
                w,
                scales=scales,
                biases=None,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode="kquant",
                kquant_type=self.kquant_type,
            )

    # Tolerance for round-trip relative error per codec. Codecs absent
    # from this dict have no MLX encoder yet and skip the round-trip test.
    # K-codec entries are added as each encoder lands. Tolerances reflect
    # the codec's inherent quantization error on N(0, 0.3^2) input.
    _ENCODE_ROUND_TRIP_TOL = {
        "q8_0": 5e-3,
        "q4_0": 8e-2,
        "q4_1": 8e-2,
        "q5_0": 4e-2,
        "q5_1": 4e-2,
        "q6_k": 2e-2,
        "q4_k": 8e-2,
        "q5_k": 4e-2,
        "q3_k": 1.3e-1,
        "q2_k": 2.6e-1,
    }

    def test_quantize_round_trip(self):
        """mx.quantize(mode='kquant') -> mx.dequantize round-trip within tol."""
        if self.kquant_type not in self._ENCODE_ROUND_TRIP_TOL:
            self.skipTest(f"MLX encoder for {self.kquant_type} not implemented")
        rng = np.random.default_rng(42)
        N, K = 8, 1024
        W_np = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        tol = self._ENCODE_ROUND_TRIP_TOL[self.kquant_type]
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                w = mx.array(W_np).astype(dtype)
                wq, scales = mx.quantize(
                    w,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode="kquant",
                    kquant_type=self.kquant_type,
                )
                mx.eval(wq, scales)
                self.assertEqual(wq.dtype, mx.uint8)
                bytes_per_row = K * self.block_bytes // self.group_size
                self.assertEqual(wq.shape, (N, bytes_per_row))
                w_dec = mx.dequantize(
                    wq,
                    scales=scales,
                    biases=None,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode="kquant",
                    kquant_type=self.kquant_type,
                    dtype=mx.float32,
                )
                mx.eval(w_dec)
                y_np = np.asarray(w_dec).astype(np.float32)
                denom = max(1e-8, float(np.max(np.abs(W_np))))
                rel = float(np.max(np.abs(y_np - W_np))) / denom
                self.assertLess(
                    rel,
                    tol,
                    msg=f"dtype={dtype}: rel={rel:.3e} tol={tol:.0e}",
                )

    def test_quantize_missing_type_rejected(self):
        """mx.quantize(mode='kquant') without kquant_type must raise."""
        w = mx.zeros((4, 64), dtype=mx.float32)
        with self.assertRaisesRegex((RuntimeError, ValueError), "kquant_type"):
            mx.quantize(w, group_size=32, bits=8, mode="kquant")

    # Codecs that consume an imatrix during encoding. Q8_0 is symmetric
    # amax-only and ignores imatrix; that's tested by the identity test
    # below (with-imatrix output equals without-imatrix output).
    _IMATRIX_AWARE_CODECS = {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k"}

    def test_quantize_imatrix_none_matches_no_arg(self):
        """imatrix=None must produce byte-identical output to omitting the arg.
        Both code paths must converge."""
        if self.kquant_type not in self._ENCODE_ROUND_TRIP_TOL:
            self.skipTest(f"MLX encoder for {self.kquant_type} not implemented")
        rng = np.random.default_rng(11)
        K = max(self.group_size, 256)
        W = rng.standard_normal((4, K)).astype(np.float32) * 0.3
        wq_a, _ = mx.quantize(
            mx.array(W),
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )
        wq_b, _ = mx.quantize(
            mx.array(W),
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
            imatrix=None,
        )
        mx.eval(wq_a, wq_b)
        np.testing.assert_array_equal(
            np.asarray(wq_a),
            np.asarray(wq_b),
            err_msg=f"{self.kquant_type}: imatrix=None must equal no-imatrix path",
        )

    def test_quantize_imatrix_improves_high_importance(self):
        """A spiky imatrix should lower error on the high-importance range.

        The spike spans a super-block boundary (cols 200..400 crosses the
        256 boundary), so the imatrix path is exercised across multiple
        super-blocks per row rather than only sub-blocks within one.
        """
        if self.kquant_type not in self._IMATRIX_AWARE_CODECS:
            self.skipTest(f"{self.kquant_type} does not consume imatrix during encode")
        rng = np.random.default_rng(7)
        K = 1024
        W = rng.standard_normal((8, K)).astype(np.float32) * 0.3
        imat_np = np.ones(K, dtype=np.float32)
        imat_np[200:400] = 10.0  # crosses super-block boundary at 256
        imat = mx.array(imat_np)
        wq_no, _ = mx.quantize(
            mx.array(W),
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )
        wq_im, _ = mx.quantize(
            mx.array(W),
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
            imatrix=imat,
        )
        mx.eval(wq_no, wq_im)
        w_no = mx.dequantize(
            wq_no,
            scales=mx.zeros((1,), dtype=mx.uint8),
            biases=None,
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
            dtype=mx.float32,
        )
        w_im = mx.dequantize(
            wq_im,
            scales=mx.zeros((1,), dtype=mx.uint8),
            biases=None,
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
            dtype=mx.float32,
        )
        mx.eval(w_no, w_im)
        w_no_np = np.asarray(w_no)
        w_im_np = np.asarray(w_im)
        err_no_hi = float(np.mean(np.abs(W[:, 200:400] - w_no_np[:, 200:400])))
        err_im_hi = float(np.mean(np.abs(W[:, 200:400] - w_im_np[:, 200:400])))
        self.assertLess(
            err_im_hi,
            err_no_hi,
            msg=(
                f"{self.kquant_type}: imatrix should improve high-importance "
                f"err: no-imatrix={err_no_hi:.4e}, with-imatrix={err_im_hi:.4e}"
            ),
        )

    def test_quantize_imatrix_wrong_shape_rejected(self):
        if self.kquant_type not in self._ENCODE_ROUND_TRIP_TOL:
            self.skipTest(f"MLX encoder for {self.kquant_type} not implemented")
        K = max(self.group_size, 256)
        w = mx.zeros((4, K), dtype=mx.float32)
        bad_im = mx.zeros((K - 1,), dtype=mx.float32)
        with self.assertRaisesRegex((RuntimeError, ValueError), "imatrix shape"):
            wq, _ = mx.quantize(
                w,
                group_size=self.group_size,
                bits=self.bits,
                mode="kquant",
                kquant_type=self.kquant_type,
                imatrix=bad_im,
            )
            mx.eval(wq)

    def test_quantize_imatrix_wrong_dtype_rejected(self):
        if self.kquant_type not in self._ENCODE_ROUND_TRIP_TOL:
            self.skipTest(f"MLX encoder for {self.kquant_type} not implemented")
        K = max(self.group_size, 256)
        w = mx.zeros((4, K), dtype=mx.float32)
        bad_im = mx.zeros((K,), dtype=mx.float16)
        with self.assertRaisesRegex(
            (RuntimeError, ValueError), "imatrix must be float32"
        ):
            wq, _ = mx.quantize(
                w,
                group_size=self.group_size,
                bits=self.bits,
                mode="kquant",
                kquant_type=self.kquant_type,
                imatrix=bad_im,
            )
            mx.eval(wq)


class TestKQuant(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q8_0 (mode='kquant', gs=32, bits=8).

    Validates the full dispatch chain Python -> ops.cpp -> Metal kernel
    (kq_q8_0_qmv / kq_q8_0_qmv_fast).
    """

    quantize_matrix = staticmethod(_quantize_q8_0_matrix)
    dequantize_matrix = staticmethod(_dequantize_q8_0_matrix)
    matmul_fn = staticmethod(_kquant_matmul)
    group_size = 32
    bits = 8
    block_bytes = Q8_0_BLOCK_BYTES
    kquant_type = "q8_0"

    def test_random_matmul_large(self):
        # 4096 x 2048 fp16 -- closer to a real generation-time linear layer.
        self._check_random_matmul(4096, 2048, mx.float16)

    def test_quantize_bit_exact(self):
        """Q8_0 has no importance weighting (_ref == _impl), so the Metal
        encoder must produce byte-identical output to the Python reference
        encoder for fp32 inputs."""
        rng = np.random.default_rng(42)
        N, K = 8, 1024
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        wq_ref = _quantize_q8_0_matrix(W)
        wq, _ = mx.quantize(
            mx.array(W),
            group_size=32,
            bits=8,
            mode="kquant",
            kquant_type="q8_0",
        )
        mx.eval(wq)
        wq_metal = np.asarray(wq)
        np.testing.assert_array_equal(
            wq_metal,
            wq_ref,
            err_msg="Q8_0 Metal encoder must match Python reference bit-exactly",
        )

    def test_quantize_all_zero_block(self):
        """A block of all zeros must produce d=0 and qs=0 (no NaN)."""
        W = np.zeros((1, 64), dtype=np.float32)
        wq, _ = mx.quantize(
            mx.array(W),
            group_size=32,
            bits=8,
            mode="kquant",
            kquant_type="q8_0",
        )
        mx.eval(wq)
        wq_bytes = np.asarray(wq).flatten()
        # 2 blocks x 34 bytes = 68 bytes; all should be zero.
        self.assertEqual(wq_bytes.shape, (68,))
        self.assertTrue((wq_bytes == 0).all())

    def test_qvm_routes_to_qmm_n(self):
        """transpose=False with small M previously threw NYI; should now route
        through qmm_n and produce correct results matching the qmm_n path.
        """
        rng = np.random.default_rng(7)
        K, N = 1024, 64
        for M in (1, 2, 3):
            W = rng.standard_normal((K, N)).astype(np.float32) * 0.3
            W_q = self.quantize_matrix(W)
            W_ref = self.dequantize_matrix(W_q, N)
            X_np = rng.standard_normal((M, K)).astype(np.float32)
            ref = X_np @ W_ref
            x = mx.array(X_np).astype(mx.float16)
            y = mx.quantized_matmul(
                x,
                mx.array(W_q),
                scales=_scales_placeholder(),
                biases=None,
                transpose=False,
                group_size=self.group_size,
                bits=self.bits,
                mode="kquant",
                kquant_type=self.kquant_type,
            )
            mx.eval(y)
            y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
            denom = max(1e-8, float(np.max(np.abs(ref))))
            rel = float(np.max(np.abs(y_np - ref))) / denom
            self.assertLess(
                rel, _MATMUL_REL_TOL[mx.float16], msg=f"M={M}: rel={rel:.3e}"
            )

    def test_unknown_codec_rejected(self):
        # An unknown kquant_type string must be rejected -- this is the
        # post-refactor analog of the old (gs, bits) lookup miss.
        w = mx.zeros((4, 34), dtype=mx.uint8)
        x = mx.zeros((1, 32), dtype=mx.float16)
        scales = _scales_placeholder()
        with self.assertRaisesRegex((RuntimeError, ValueError), "Unknown kquant_type"):
            mx.quantized_matmul(
                x,
                w,
                scales=scales,
                biases=None,
                transpose=True,
                group_size=32,
                bits=8,
                mode="kquant",
                kquant_type="q9_z",  # not a real codec
            )

    def test_missing_kquant_type_rejected(self):
        # mode='kquant' without a kquant_type string must error -- there is
        # no fallback because (gs, bits) does not uniquely identify a codec.
        w = mx.zeros((4, 34), dtype=mx.uint8)
        x = mx.zeros((1, 32), dtype=mx.float16)
        scales = _scales_placeholder()
        with self.assertRaisesRegex(
            (RuntimeError, ValueError), "kquant mode requires kquant_type"
        ):
            mx.quantized_matmul(
                x,
                w,
                scales=scales,
                biases=None,
                transpose=True,
                group_size=32,
                bits=8,
                mode="kquant",
            )

    def test_nax_k_not_aligned_falls_back_to_alu(self):
        """NAX requires K % 64 == 0. With K=96 (multiple of 32 but not 64) the
        kquant dispatch must take the ALU path and still produce correct
        results. Catches the case where the NAX gate is missing the K%64
        guard or where the ALU fallback regresses."""
        rng = np.random.default_rng(0)
        K = 96
        M, N = 4, 32
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.2
        W_q = self.quantize_matrix(W)
        W_ref = self.dequantize_matrix(W_q, K)
        x_np = rng.standard_normal((M, K)).astype(np.float32)
        ref = x_np @ W_ref.T
        x = mx.array(x_np).astype(mx.float16)
        y = mx.quantized_matmul(
            x,
            mx.array(W_q),
            scales=_scales_placeholder(),
            biases=None,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        self.assertLess(rel, _MATMUL_REL_TOL[mx.float16])


class TestKQuantQ4_0(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q4_0 (mode='kquant', gs=32, bits=4).

    Q4_0 is the simplest 4-bit codec: per-block fp16 scale, symmetric
    nibble pack with -8 centering, no min. Same threading geometry as
    Q8_0 -- split-half nibble layout means each thread's 8-weight slice
    falls entirely in either the low or high nibble half.
    """

    quantize_matrix = staticmethod(_quantize_q4_0_matrix)
    dequantize_matrix = staticmethod(_dequantize_q4_0_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q4_0)
    group_size = 32
    bits = 4
    block_bytes = Q4_0_BLOCK_BYTES
    kquant_type = "q4_0"


class TestKQuantQ4_1(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q4_1 (mode='kquant', gs=32, bits=4).

    Q4_1 is asymmetric: per-block fp16 (d, m) envelope and a 4-bit nibble
    pack with split-half layout. Shares (group_size=32, bits=4) with Q4_0;
    the codec-name dispatch is the only thing distinguishing them.
    """

    quantize_matrix = staticmethod(_quantize_q4_1_matrix)
    dequantize_matrix = staticmethod(_dequantize_q4_1_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q4_1)
    group_size = 32
    bits = 4
    block_bytes = Q4_1_BLOCK_BYTES
    kquant_type = "q4_1"


class TestKQuantQ5_0(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q5_0 (mode='kquant', gs=32, bits=5).

    Q5_0 is symmetric: per-block fp16 d, no min, 5-bit weights centered
    at 16. qh extraction is bit-identical to Q5_1's; the (gs, bits) tuple
    collides with Q5_1 -- only the codec name distinguishes them.
    """

    quantize_matrix = staticmethod(_quantize_q5_0_matrix)
    dequantize_matrix = staticmethod(_dequantize_q5_0_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q5_0)
    group_size = 32
    bits = 5
    block_bytes = Q5_0_BLOCK_BYTES
    kquant_type = "q5_0"


class TestKQuantQ5_1(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q5_1 (mode='kquant', gs=32, bits=5).

    Validates the full dispatch chain Python -> ops.cpp -> Metal kernel
    (kq_q5_1_qmv / kq_q5_1_qmv_fast). Q5_1 is asymmetric (d, m) with
    5-bit unsigned weights; the kernel uses factored accumulation
    (d * sum(x*q) + m * sum(x)) so the m-term is row-independent.
    """

    quantize_matrix = staticmethod(_quantize_q5_1_matrix)
    dequantize_matrix = staticmethod(_dequantize_q5_1_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q5_1)
    group_size = 32
    bits = 5
    block_bytes = Q5_1_BLOCK_BYTES
    kquant_type = "q5_1"

    def test_random_matmul_large(self):
        # 4096 x 2048 fp16 -- closer to a real generation-time linear layer.
        self._check_random_matmul(4096, 2048, mx.float16)


QK_K = 256
K_SCALE_SIZE = 12
Q4_K_BLOCK_BYTES = 144  # fp16 d (2) + fp16 dmin (2) + scales[12] + qs[128]
Q4_K_D_OFFSET = 0
Q4_K_DMIN_OFFSET = 2
Q4_K_SCALES_OFFSET = 4
Q4_K_QS_OFFSET = 16


def _unpack_scale_min_q4k(scales12: np.ndarray):
    """Unpack 6-bit sub-scale/min pairs from the 12-byte field. Returns (sc8, mn8),
    each shape (n_blocks, 8), uint8 in [0, 63]."""
    s = scales12.astype(np.uint8)
    n = s.shape[0]
    sc = np.empty((n, 8), dtype=np.uint8)
    mn = np.empty((n, 8), dtype=np.uint8)
    sc[:, 0:4] = s[:, 0:4] & 0x3F
    mn[:, 0:4] = s[:, 4:8] & 0x3F
    sc[:, 4:8] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


def _pack_scale_min_q4k(sc6: np.ndarray, mn6: np.ndarray) -> np.ndarray:
    """Inverse of _unpack_scale_min_q4k for a single super-block. sc6/mn6 are
    length-8 arrays in [0, 63]; returns 12-byte uint8 array."""
    out = np.zeros(12, dtype=np.uint8)
    for j in range(4):
        out[j] = (sc6[j] & 0x3F) | (((sc6[j + 4] >> 4) & 0x3) << 6)
        out[j + 4] = (mn6[j] & 0x3F) | (((mn6[j + 4] >> 4) & 0x3) << 6)
        out[j + 8] = (sc6[j + 4] & 0x0F) | ((mn6[j + 4] & 0x0F) << 4)
    return out


def _quantize_q4_k_row(row: np.ndarray) -> np.ndarray:
    """Quantize a 1D fp32 array (length must be a multiple of 256) to Q4_K
    packed wire bytes. Simple non-optimal scheme: per sub-block, sc = (max-min)/15
    and the_min = -min (>= 0 since q is non-negative). Quantize sc and the_min
    to 6 bits via a per-super-block envelope (d, dmin). Sufficient for tests
    because the kernel is compared against a dequant of the same wire bytes."""
    assert row.ndim == 1
    assert row.size % QK_K == 0, "Q4_K requires K % 256 == 0"
    n_super = row.size // QK_K
    out = np.zeros(n_super * Q4_K_BLOCK_BYTES, dtype=np.uint8)
    for sb_idx in range(n_super):
        super_block = (
            row[sb_idx * QK_K : (sb_idx + 1) * QK_K].astype(np.float32).reshape(8, 32)
        )
        sc_local = np.zeros(8, dtype=np.float32)
        mn_local = np.zeros(8, dtype=np.float32)
        q4 = np.zeros((8, 32), dtype=np.uint8)
        for j in range(8):
            block = super_block[j]
            mn_j = float(block.min())
            mx_j = float(block.max())
            if mx_j == mn_j:
                sc_local[j] = 0.0
                mn_local[j] = -mn_j  # the_min = -block.min() >= 0 if min <= 0
                q4[j, :] = 0
            else:
                sc_local[j] = (mx_j - mn_j) / 15.0
                mn_local[j] = -mn_j
                q4[j, :] = np.clip(
                    np.round((block - mn_j) / sc_local[j]), 0, 15
                ).astype(np.uint8)
        # Quantize sc and mn to 6-bit. mn_local can be negative if block.min > 0;
        # in that case the encoding cannot represent it (dmin_mn6 >= 0), so
        # dequant will reconstruct y ~= d*sc*q + 0 which still spans [min..max]
        # because q in [0, 15] and sc covers the full range. Edge cases (all
        # negative blocks with positive mn_local clamped) are handled below.
        max_sc = float(sc_local.max())
        # Clamp mn_local to >= 0 (Q4_K's dmin*mn6 term is non-negative); blocks
        # whose values are entirely > 0 get mn_local < 0, which we round to 0.
        mn_clamped = np.maximum(mn_local, 0.0)
        max_mn = float(mn_clamped.max())
        if max_sc <= 0.0:
            d = np.float32(0.0)
            sc6 = np.zeros(8, dtype=np.uint8)
        else:
            d = np.float32(max_sc / 63.0)
            sc6 = np.clip(np.round(sc_local / d), 0, 63).astype(np.uint8)
        if max_mn <= 0.0:
            dmin = np.float32(0.0)
            mn6 = np.zeros(8, dtype=np.uint8)
        else:
            dmin = np.float32(max_mn / 63.0)
            mn6 = np.clip(np.round(mn_clamped / dmin), 0, 63).astype(np.uint8)

        scales12 = _pack_scale_min_q4k(sc6, mn6)

        # Pack qs[128]: qs[p*32 + l] = q4[2p, l] | (q4[2p+1, l] << 4)
        qs = np.zeros(128, dtype=np.uint8)
        for p in range(4):
            for l in range(32):
                qs[p * 32 + l] = (q4[2 * p, l] & 0x0F) | (
                    (q4[2 * p + 1, l] & 0x0F) << 4
                )

        d_fp16 = np.float16(d)
        dmin_fp16 = np.float16(dmin)
        base = sb_idx * Q4_K_BLOCK_BYTES
        out[base + Q4_K_D_OFFSET : base + Q4_K_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q4_K_DMIN_OFFSET : base + Q4_K_DMIN_OFFSET + 2] = np.frombuffer(
            dmin_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q4_K_SCALES_OFFSET : base + Q4_K_SCALES_OFFSET + K_SCALE_SIZE] = (
            scales12
        )
        out[base + Q4_K_QS_OFFSET : base + Q4_K_QS_OFFSET + 128] = qs
    return out


def _quantize_q4_k_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % QK_K == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q4_K_BLOCK_BYTES // QK_K
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q4_k_row(W[i])
    return out


def _dequantize_q4_k_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q4_K dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q4_K_BLOCK_BYTES // QK_K
    n_blocks_per_row = in_dim // QK_K
    blocks = W_q.reshape(out_dim * n_blocks_per_row, Q4_K_BLOCK_BYTES)
    n_blocks = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc8, mn8 = _unpack_scale_min_q4k(blocks[:, 4:16])
    qs = blocks[:, 16 : 16 + 128]

    sub_scale = d[:, None] * sc8.astype(np.float32)
    sub_min = dmin[:, None] * mn8.astype(np.float32)

    qs_g = qs.reshape(n_blocks, 4, 32)
    low_nib = (qs_g & 0x0F).astype(np.float32)
    high_nib = (qs_g >> 4).astype(np.float32)
    sub_q = np.stack([low_nib, high_nib], axis=2).reshape(n_blocks, 8, 32)

    out_flat = (
        (sub_scale[:, :, None] * sub_q - sub_min[:, :, None])
        .reshape(n_blocks * QK_K)
        .astype(np.float32)
    )
    return out_flat.reshape(out_dim, in_dim)


def _kquant_matmul_q4_k(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q4_K (gs=256, bits=4)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=256,
        bits=4,
        mode="kquant",
        kquant_type="q4_k",
    )


Q5_K_BLOCK_BYTES = 176  # Q4_K + qh[32]
Q5_K_D_OFFSET = 0
Q5_K_DMIN_OFFSET = 2
Q5_K_SCALES_OFFSET = 4
Q5_K_QH_OFFSET = 16
Q5_K_QS_OFFSET = 48


def _quantize_q5_k_row(row: np.ndarray) -> np.ndarray:
    """Quantize fp32 row to Q5_K wire bytes. Same scheme as Q4_K but with q
    in [0, 31] instead of [0, 15]; the high bit goes into qh[32] under the
    transposed scheme: bit `sb` of qh[l] = high bit of weight `sb*32 + l`."""
    assert row.ndim == 1
    assert row.size % QK_K == 0, "Q5_K requires K % 256 == 0"
    n_super = row.size // QK_K
    out = np.zeros(n_super * Q5_K_BLOCK_BYTES, dtype=np.uint8)
    for sb_idx in range(n_super):
        super_block = (
            row[sb_idx * QK_K : (sb_idx + 1) * QK_K].astype(np.float32).reshape(8, 32)
        )
        sc_local = np.zeros(8, dtype=np.float32)
        mn_local = np.zeros(8, dtype=np.float32)
        q5 = np.zeros((8, 32), dtype=np.uint8)
        for j in range(8):
            block = super_block[j]
            mn_j = float(block.min())
            mx_j = float(block.max())
            if mx_j == mn_j:
                sc_local[j] = 0.0
                mn_local[j] = -mn_j
                q5[j, :] = 0
            else:
                sc_local[j] = (mx_j - mn_j) / 31.0
                mn_local[j] = -mn_j
                q5[j, :] = np.clip(
                    np.round((block - mn_j) / sc_local[j]), 0, 31
                ).astype(np.uint8)

        max_sc = float(sc_local.max())
        mn_clamped = np.maximum(mn_local, 0.0)
        max_mn = float(mn_clamped.max())
        if max_sc <= 0.0:
            d = np.float32(0.0)
            sc6 = np.zeros(8, dtype=np.uint8)
        else:
            d = np.float32(max_sc / 63.0)
            sc6 = np.clip(np.round(sc_local / d), 0, 63).astype(np.uint8)
        if max_mn <= 0.0:
            dmin = np.float32(0.0)
            mn6 = np.zeros(8, dtype=np.uint8)
        else:
            dmin = np.float32(max_mn / 63.0)
            mn6 = np.clip(np.round(mn_clamped / dmin), 0, 63).astype(np.uint8)

        scales12 = _pack_scale_min_q4k(sc6, mn6)

        # Low 4 bits -> qs[128] nibble pairs (same packing as Q4_K).
        # High bit -> qh[32] under the transposed scheme: for weight at
        # position [sb, l] (sub-block sb, within-sb l), bit sb of qh[l] is set.
        qs = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(32, dtype=np.uint8)
        for p in range(4):
            for l in range(32):
                low_even = q5[2 * p, l] & 0x0F
                low_odd = q5[2 * p + 1, l] & 0x0F
                qs[p * 32 + l] = low_even | (low_odd << 4)
        for sb in range(8):
            for l in range(32):
                bit = (int(q5[sb, l]) >> 4) & 1
                qh[l] |= bit << sb

        d_fp16 = np.float16(d)
        dmin_fp16 = np.float16(dmin)
        base = sb_idx * Q5_K_BLOCK_BYTES
        out[base + Q5_K_D_OFFSET : base + Q5_K_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q5_K_DMIN_OFFSET : base + Q5_K_DMIN_OFFSET + 2] = np.frombuffer(
            dmin_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q5_K_SCALES_OFFSET : base + Q5_K_SCALES_OFFSET + K_SCALE_SIZE] = (
            scales12
        )
        out[base + Q5_K_QH_OFFSET : base + Q5_K_QH_OFFSET + 32] = qh
        out[base + Q5_K_QS_OFFSET : base + Q5_K_QS_OFFSET + 128] = qs
    return out


def _quantize_q5_k_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % QK_K == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q5_K_BLOCK_BYTES // QK_K
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q5_k_row(W[i])
    return out


def _dequantize_q5_k_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q5_K dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q5_K_BLOCK_BYTES // QK_K
    n_blocks_per_row = in_dim // QK_K
    blocks = W_q.reshape(out_dim * n_blocks_per_row, Q5_K_BLOCK_BYTES)
    n_blocks = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc8, mn8 = _unpack_scale_min_q4k(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48 : 48 + 128]

    sub_scale = d[:, None] * sc8.astype(np.float32)
    sub_min = dmin[:, None] * mn8.astype(np.float32)

    qs_g = qs.reshape(n_blocks, 4, 32)
    low_nib = (qs_g & 0x0F).astype(np.uint8)
    high_nib = (qs_g >> 4).astype(np.uint8)
    low_bits = np.stack([low_nib, high_nib], axis=2).reshape(n_blocks, 8, 32)

    bit_sel = np.arange(8, dtype=np.uint8).reshape(1, 8, 1)
    high_bit = (qh[:, None, :] >> bit_sel) & 0x01

    q5 = (low_bits | (high_bit << 4)).astype(np.float32)
    out_flat = (
        (sub_scale[:, :, None] * q5 - sub_min[:, :, None])
        .reshape(n_blocks * QK_K)
        .astype(np.float32)
    )
    return out_flat.reshape(out_dim, in_dim)


def _kquant_matmul_q5_k(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q5_K (gs=256, bits=5)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=256,
        bits=5,
        mode="kquant",
        kquant_type="q5_k",
    )


# REVERSED field order vs Q4_K/Q5_K: payload first, envelope last.
Q6_K_BLOCK_BYTES = 210
Q6_K_QL_OFFSET = 0
Q6_K_QH_OFFSET = 128
Q6_K_SCALES_OFFSET = 192
Q6_K_D_OFFSET = 208


def _quantize_q6_k_row(row: np.ndarray) -> np.ndarray:
    """Quantize fp32 row to Q6_K wire bytes. Symmetric codec: 16 sub-blocks
    of 16 weights, signed int8 sub-block scales, no dmin. Simplified
    quantizer (positive sub-scales only) -- produces valid q6_K bytes that
    round-trip through the reference dequantizer."""
    assert row.ndim == 1
    assert row.size % QK_K == 0, "Q6_K requires K % 256 == 0"
    n_super = row.size // QK_K
    out = np.zeros(n_super * Q6_K_BLOCK_BYTES, dtype=np.uint8)
    for sb_idx in range(n_super):
        super_block = (
            row[sb_idx * QK_K : (sb_idx + 1) * QK_K]
            .astype(np.float32)
            .reshape(16, 16)  # 16 sub-blocks of 16 weights each
        )
        scale_local = np.zeros(16, dtype=np.float32)
        L_signed = np.zeros((16, 16), dtype=np.int32)  # in [-32, 31]
        for j in range(16):
            block = super_block[j]
            amax = float(np.max(np.abs(block)))
            if amax == 0.0:
                scale_local[j] = 0.0
                L_signed[j, :] = 0
            else:
                scale_local[j] = amax / 31.0
                L_signed[j, :] = np.clip(
                    np.round(block / scale_local[j]), -32, 31
                ).astype(np.int32)

        max_sc = float(np.max(np.abs(scale_local)))
        if max_sc == 0.0:
            d = np.float32(0.0)
            scales_i8 = np.zeros(16, dtype=np.int8)
        else:
            d = np.float32(max_sc / 127.0)
            scales_i8 = np.clip(np.round(scale_local / d), -127, 127).astype(np.int8)

        # Encode L as unsigned 6-bit: q6_unsigned = L_signed + 32 in [0, 63].
        # Layout: super_block[j, i] = row[j*16 + i], so weight at global
        # within-superblock index g = j*16 + i uses scales_i8[j] = scales_i8[g/16]
        # -- matches the dequant scale assignment.
        L_flat = (L_signed + 32).astype(np.uint8).reshape(QK_K)

        ql_out = np.zeros(128, dtype=np.uint8)
        qh_out = np.zeros(64, dtype=np.uint8)
        # Pack ql/qh:
        # 2 halves x 32 columns; quadrants 0/2 share ql[0..31] (low/high
        # nibble), quadrants 1/3 share ql[32..63]; all 4 share qh[0..31]
        # at shifts {0, 2, 4, 6}.
        for half in range(2):
            h = half * 128
            for l in range(32):
                ql_out[half * 64 + l + 0] = (L_flat[h + l + 0] & 0xF) | (
                    (L_flat[h + l + 64] & 0xF) << 4
                )
                ql_out[half * 64 + l + 32] = (L_flat[h + l + 32] & 0xF) | (
                    (L_flat[h + l + 96] & 0xF) << 4
                )
                qh_out[half * 32 + l] = (
                    (L_flat[h + l + 0] >> 4)
                    | ((L_flat[h + l + 32] >> 4) << 2)
                    | ((L_flat[h + l + 64] >> 4) << 4)
                    | ((L_flat[h + l + 96] >> 4) << 6)
                )

        d_fp16 = np.float16(d)
        base = sb_idx * Q6_K_BLOCK_BYTES
        out[base + Q6_K_QL_OFFSET : base + Q6_K_QL_OFFSET + 128] = ql_out
        out[base + Q6_K_QH_OFFSET : base + Q6_K_QH_OFFSET + 64] = qh_out
        out[base + Q6_K_SCALES_OFFSET : base + Q6_K_SCALES_OFFSET + 16] = np.frombuffer(
            scales_i8.tobytes(), dtype=np.uint8
        )
        out[base + Q6_K_D_OFFSET : base + Q6_K_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
    return out


def _quantize_q6_k_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % QK_K == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q6_K_BLOCK_BYTES // QK_K
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q6_k_row(W[i])
    return out


def _dequantize_q6_k_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q6_K dequantization."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q6_K_BLOCK_BYTES // QK_K
    n_blocks_per_row = in_dim // QK_K
    blocks = W_q.reshape(out_dim * n_blocks_per_row, Q6_K_BLOCK_BYTES)
    n_blocks = blocks.shape[0]

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208]
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc16 = scales.view(np.int8).astype(np.float32)

    ql_h = ql.reshape(n_blocks, 2, 64)
    qh_h = qh.reshape(n_blocks, 2, 32)
    sc_h = sc16.reshape(n_blocks, 2, 8)

    out = np.empty((n_blocks, 2, 128), dtype=np.float32)
    is_idx = np.arange(32) // 16
    for half_idx in range(2):
        ql_half = ql_h[:, half_idx, :]
        qh_half = qh_h[:, half_idx, :]
        sc_half = sc_h[:, half_idx, :]
        ql_lo = ql_half[:, 0:32]
        ql_lo32 = ql_half[:, 32:64]
        q1 = ((ql_lo & 0x0F) | (((qh_half >> 0) & 0x03) << 4)).astype(
            np.int8
        ) - np.int8(32)
        q2 = ((ql_lo32 & 0x0F) | (((qh_half >> 2) & 0x03) << 4)).astype(
            np.int8
        ) - np.int8(32)
        q3 = ((ql_lo >> 4) | (((qh_half >> 4) & 0x03) << 4)).astype(np.int8) - np.int8(
            32
        )
        q4 = ((ql_lo32 >> 4) | (((qh_half >> 6) & 0x03) << 4)).astype(
            np.int8
        ) - np.int8(32)
        for is_off, qq, out_slice in (
            (0, q1, slice(0, 32)),
            (2, q2, slice(32, 64)),
            (4, q3, slice(64, 96)),
            (6, q4, slice(96, 128)),
        ):
            scl = sc_half[:, is_off + is_idx]
            d_eff = d[:, None] * scl
            out[:, half_idx, out_slice] = d_eff * qq.astype(np.float32)

    return out.reshape(n_blocks * QK_K).astype(np.float32).reshape(out_dim, in_dim)


def _kquant_matmul_q6_k(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q6_K (gs=256, bits=6)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=256,
        bits=6,
        mode="kquant",
        kquant_type="q6_k",
    )


Q3_K_BLOCK_BYTES = 110  # hmask[32] + qs[64] + scales[12] + fp16 d (2)
Q3_K_HMASK_OFFSET = 0
Q3_K_QS_OFFSET = 32
Q3_K_SCALES_OFFSET = 96
Q3_K_D_OFFSET = 108


def _unpack_scale_q3k(scales12: np.ndarray) -> np.ndarray:
    """Unpack 4-word bit-shuffled Q3_K scales. Given the 12-byte `scales` field,
    returns a length-16 uint8 array of the per-sub-block scales in [0, 63]
    (subtract 32 for the signed [-32, 31] effective scale)."""
    s = scales12.astype(np.uint8)
    assert s.shape == (12,)
    out = np.zeros(16, dtype=np.uint8)
    for k in range(4):
        # quad 0: scales[0..3] -> low 4 from s[0..3], high 2 from s[8..11] bits 0-1
        out[k] = (s[k] & 0x0F) | ((s[8 + k] & 0x03) << 4)
        # quad 1: scales[4..7] -> low 4 from s[4..7], high 2 from s[8..11] bits 2-3
        out[k + 4] = (s[k + 4] & 0x0F) | (((s[8 + k] >> 2) & 0x03) << 4)
        # quad 2: scales[8..11] -> low 4 from s[0..3] high nibble, high 2 from s[8..11] bits 4-5
        out[k + 8] = ((s[k] >> 4) & 0x0F) | (((s[8 + k] >> 4) & 0x03) << 4)
        # quad 3: scales[12..15] -> low 4 from s[4..7] high nibble, high 2 from s[8..11] bits 6-7
        out[k + 12] = ((s[k + 4] >> 4) & 0x0F) | (((s[8 + k] >> 6) & 0x03) << 4)
    return out


def _pack_scale_q3k(sc6: np.ndarray) -> np.ndarray:
    """Inverse of _unpack_scale_q3k. sc6 is a length-16 uint8 array in [0, 63];
    returns a 12-byte uint8 array of packed scales. Bijective: pack(unpack(x))==x."""
    sc = sc6.astype(np.uint8)
    assert sc.shape == (16,)
    out = np.zeros(12, dtype=np.uint8)
    for k in range(4):
        out[k] = (sc[k] & 0x0F) | ((sc[k + 8] & 0x0F) << 4)
        out[k + 4] = (sc[k + 4] & 0x0F) | ((sc[k + 12] & 0x0F) << 4)
        out[k + 8] = (
            ((sc[k] >> 4) & 0x03)
            | (((sc[k + 4] >> 4) & 0x03) << 2)
            | (((sc[k + 8] >> 4) & 0x03) << 4)
            | (((sc[k + 12] >> 4) & 0x03) << 6)
        )
    return out


def _quantize_q3_k_row(row: np.ndarray) -> np.ndarray:
    """Quantize fp32 row to Q3_K wire bytes. Symmetric codec: 16 sub-blocks of
    16 weights, signed 6-bit-biased per-sub-block scales (encoded via the
    bit-shuffled 12-byte `scales` field), single fp16 super-block envelope `d`.
    q3 in [-4, 3] split as 2-bit qs payload + 1-bit hmask (hmask SET when q3 >= 0).
    Simplified non-optimal quantizer (positive sub-scales only) -- produces valid
    Q3_K bytes that round-trip through the reference dequantizer."""
    assert row.ndim == 1
    assert row.size % QK_K == 0, "Q3_K requires K % 256 == 0"
    n_super = row.size // QK_K
    out = np.zeros(n_super * Q3_K_BLOCK_BYTES, dtype=np.uint8)
    for sb_idx in range(n_super):
        super_block = (
            row[sb_idx * QK_K : (sb_idx + 1) * QK_K]
            .astype(np.float32)
            .reshape(16, 16)  # 16 sub-blocks of 16 weights each
        )
        scale_local = np.zeros(16, dtype=np.float32)
        L_signed = np.zeros((16, 16), dtype=np.int32)  # in [-4, 3]
        for j in range(16):
            block = super_block[j]
            amax = float(np.max(np.abs(block)))
            if amax == 0.0:
                scale_local[j] = 0.0
                L_signed[j, :] = 0
            else:
                scale_local[j] = amax / 4.0  # q in [-4, 3]
                L_signed[j, :] = np.clip(
                    np.round(block / scale_local[j]), -4, 3
                ).astype(np.int32)

        max_sc = float(scale_local.max())
        if max_sc <= 0.0:
            d = np.float32(0.0)
            sc6 = np.full(16, 32, dtype=np.uint8)  # encodes 0
        else:
            d = np.float32(max_sc / 31.0)
            sc_unsigned = np.clip(np.round(scale_local / d), 0, 31).astype(np.int32)
            sc6 = (sc_unsigned + 32).astype(np.uint8)  # bias by +32

        scales12 = _pack_scale_q3k(sc6)

        # Pack qs[64] (2-bit payload) and hmask[32] (1-bit high mask).
        # Linear weight w_idx = j*16 + i in [0, 256):
        #   qs_byte_idx = (w_idx // 128) * 32 + (w_idx & 31)
        #   qs_shift    = ((w_idx // 32) & 3) * 2
        #   hmask_byte  = w_idx & 31
        #   hmask_bit   = (w_idx >> 5) & 7
        # h_bit = 1 iff q3_signed >= 0; q2 = q3_signed & 3 (covers both ranges).
        L_flat = L_signed.reshape(QK_K)
        q2_arr = (L_flat & 3).astype(np.uint8)
        h_bit_arr = (L_flat >= 0).astype(np.uint8)

        qs = np.zeros(64, dtype=np.uint8)
        hmask = np.zeros(32, dtype=np.uint8)
        for w_idx in range(QK_K):
            outer_half = w_idx // 128
            shift_idx = (w_idx // 32) & 3
            qs_byte_idx = outer_half * 32 + (w_idx & 31)
            qs[qs_byte_idx] |= int(q2_arr[w_idx]) << (shift_idx * 2)
            hmask_byte = w_idx & 31
            hmask_bit = (w_idx >> 5) & 7
            hmask[hmask_byte] |= int(h_bit_arr[w_idx]) << hmask_bit

        d_fp16 = np.float16(d)
        base = sb_idx * Q3_K_BLOCK_BYTES
        out[base + Q3_K_HMASK_OFFSET : base + Q3_K_HMASK_OFFSET + 32] = hmask
        out[base + Q3_K_QS_OFFSET : base + Q3_K_QS_OFFSET + 64] = qs
        out[base + Q3_K_SCALES_OFFSET : base + Q3_K_SCALES_OFFSET + 12] = scales12
        out[base + Q3_K_D_OFFSET : base + Q3_K_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
    return out


def _quantize_q3_k_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % QK_K == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q3_K_BLOCK_BYTES // QK_K
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q3_k_row(W[i])
    return out


def _dequantize_q3_k_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q3_K dequantization. w[i] = d * (scales[is] - 32) * (q2 - h),
    where q2 in [0, 3] from qs and h in {0, 4} (h=0 if hmask bit SET, else 4)."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q3_K_BLOCK_BYTES // QK_K
    n_blocks_per_row = in_dim // QK_K
    blocks = W_q.reshape(out_dim * n_blocks_per_row, Q3_K_BLOCK_BYTES)
    n_blocks = blocks.shape[0]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for b in range(n_blocks):
        hmask = blocks[b, Q3_K_HMASK_OFFSET : Q3_K_HMASK_OFFSET + 32]
        qs_full = blocks[b, Q3_K_QS_OFFSET : Q3_K_QS_OFFSET + 64]
        scales12 = blocks[b, Q3_K_SCALES_OFFSET : Q3_K_SCALES_OFFSET + 12]
        d = float(
            np.frombuffer(
                blocks[b, Q3_K_D_OFFSET : Q3_K_D_OFFSET + 2].tobytes(),
                dtype=np.float16,
            )[0]
        )
        sc16 = _unpack_scale_q3k(scales12).astype(np.int32) - 32  # signed [-32, 31]

        out_idx = 0
        for outer_half in range(2):
            qs_chunk = qs_full[outer_half * 32 : (outer_half + 1) * 32]
            for shift_idx in range(4):
                shift = shift_idx * 2
                # m = 1 << (outer_half * 4 + shift_idx) -- same as 1 << hmask_bit
                m = 1 << (outer_half * 4 + shift_idx)
                is_left = outer_half * 8 + shift_idx * 2
                dl_left = d * float(sc16[is_left])
                for l in range(16):
                    q2 = (int(qs_chunk[l]) >> shift) & 3
                    h = 0 if (int(hmask[l]) & m) else 4
                    out[b, out_idx] = dl_left * (q2 - h)
                    out_idx += 1
                is_right = is_left + 1
                dl_right = d * float(sc16[is_right])
                for l in range(16):
                    q2 = (int(qs_chunk[l + 16]) >> shift) & 3
                    h = 0 if (int(hmask[l + 16]) & m) else 4
                    out[b, out_idx] = dl_right * (q2 - h)
                    out_idx += 1

    return out.reshape(out_dim, in_dim)


def _kquant_matmul_q3_k(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q3_K (gs=256, bits=3)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=256,
        bits=3,
        mode="kquant",
        kquant_type="q3_k",
    )


Q2_K_BLOCK_BYTES = 84  # scales[16] + qs[64] + fp16 d + fp16 dmin
Q2_K_SCALES_OFFSET = 0
Q2_K_QS_OFFSET = 16
Q2_K_D_OFFSET = 80
Q2_K_DMIN_OFFSET = 82


def _quantize_q2_k_row(row: np.ndarray) -> np.ndarray:
    """Quantize fp32 row to Q2_K wire bytes. Asymmetric codec: 16 sub-blocks
    of 16 weights, 4-bit (scale, min) nibble pair per sub-block in scales[16],
    fp16 (d, dmin) super-block envelope. q2 in [0, 3] in qs (2 bits/weight,
    4 weights/byte). Simplified non-optimal quantizer -- produces valid Q2_K
    bytes that round-trip through the reference dequantizer."""
    assert row.ndim == 1
    assert row.size % QK_K == 0, "Q2_K requires K % 256 == 0"
    n_super = row.size // QK_K
    out = np.zeros(n_super * Q2_K_BLOCK_BYTES, dtype=np.uint8)
    for sb_idx in range(n_super):
        super_block = (
            row[sb_idx * QK_K : (sb_idx + 1) * QK_K]
            .astype(np.float32)
            .reshape(16, 16)  # 16 sub-blocks of 16 weights each
        )
        sc_local = np.zeros(16, dtype=np.float32)
        mn_local = np.zeros(16, dtype=np.float32)
        q2_arr = np.zeros((16, 16), dtype=np.uint8)
        for j in range(16):
            block = super_block[j]
            mn_j = float(block.min())
            mx_j = float(block.max())
            if mx_j == mn_j:
                sc_local[j] = 0.0
                mn_local[j] = -mn_j
                q2_arr[j, :] = 0
            else:
                sc_local[j] = (mx_j - mn_j) / 3.0  # q in [0, 3]
                mn_local[j] = -mn_j
                q2_arr[j, :] = np.clip(
                    np.round((block - mn_j) / sc_local[j]), 0, 3
                ).astype(np.uint8)

        # Quantize sub-scales and sub-mins to 4-bit unsigned. mn_local can be
        # negative if all values in block are positive; the encoded min is
        # non-negative so we clamp >= 0 (dequant reconstructs values in
        # [0, d*sc*3] = [0, max_j], slight rounding loss for entirely positive
        # blocks but acceptable for the test reference).
        max_sc = float(sc_local.max())
        mn_clamped = np.maximum(mn_local, 0.0)
        max_mn = float(mn_clamped.max())
        if max_sc <= 0.0:
            d = np.float32(0.0)
            sc4 = np.zeros(16, dtype=np.uint8)
        else:
            d = np.float32(max_sc / 15.0)
            sc4 = np.clip(np.round(sc_local / d), 0, 15).astype(np.uint8)
        if max_mn <= 0.0:
            dmin = np.float32(0.0)
            mn4 = np.zeros(16, dtype=np.uint8)
        else:
            dmin = np.float32(max_mn / 15.0)
            mn4 = np.clip(np.round(mn_clamped / dmin), 0, 15).astype(np.uint8)

        scales = (sc4 | (mn4 << 4)).astype(np.uint8)  # length 16

        # Pack qs[64]: same access pattern as Q3_K (no hmask).
        L_flat = q2_arr.reshape(QK_K)
        qs = np.zeros(64, dtype=np.uint8)
        for w_idx in range(QK_K):
            outer_half = w_idx // 128
            shift_idx = (w_idx // 32) & 3
            qs_byte_idx = outer_half * 32 + (w_idx & 31)
            qs[qs_byte_idx] |= int(L_flat[w_idx]) << (shift_idx * 2)

        d_fp16 = np.float16(d)
        dmin_fp16 = np.float16(dmin)
        base = sb_idx * Q2_K_BLOCK_BYTES
        out[base + Q2_K_SCALES_OFFSET : base + Q2_K_SCALES_OFFSET + 16] = scales
        out[base + Q2_K_QS_OFFSET : base + Q2_K_QS_OFFSET + 64] = qs
        out[base + Q2_K_D_OFFSET : base + Q2_K_D_OFFSET + 2] = np.frombuffer(
            d_fp16.tobytes(), dtype=np.uint8
        )
        out[base + Q2_K_DMIN_OFFSET : base + Q2_K_DMIN_OFFSET + 2] = np.frombuffer(
            dmin_fp16.tobytes(), dtype=np.uint8
        )
    return out


def _quantize_q2_k_matrix(W: np.ndarray) -> np.ndarray:
    assert W.ndim == 2
    assert W.shape[1] % QK_K == 0
    out_dim, in_dim = W.shape
    bytes_per_row = in_dim * Q2_K_BLOCK_BYTES // QK_K
    out = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)
    for i in range(out_dim):
        out[i] = _quantize_q2_k_row(W[i])
    return out


def _dequantize_q2_k_matrix(W_q: np.ndarray, in_dim: int) -> np.ndarray:
    """Reference Q2_K dequantization.
    w[i] = d * (sc & 0xF) * q2 - dmin * (sc >> 4)."""
    assert W_q.dtype == np.uint8
    out_dim = W_q.shape[0]
    bytes_per_row = W_q.shape[1]
    assert bytes_per_row == in_dim * Q2_K_BLOCK_BYTES // QK_K
    n_blocks_per_row = in_dim // QK_K
    blocks = W_q.reshape(out_dim * n_blocks_per_row, Q2_K_BLOCK_BYTES)
    n_blocks = blocks.shape[0]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for b in range(n_blocks):
        scales = blocks[b, Q2_K_SCALES_OFFSET : Q2_K_SCALES_OFFSET + 16]
        qs_full = blocks[b, Q2_K_QS_OFFSET : Q2_K_QS_OFFSET + 64]
        d = float(
            np.frombuffer(
                blocks[b, Q2_K_D_OFFSET : Q2_K_D_OFFSET + 2].tobytes(),
                dtype=np.float16,
            )[0]
        )
        dmin = float(
            np.frombuffer(
                blocks[b, Q2_K_DMIN_OFFSET : Q2_K_DMIN_OFFSET + 2].tobytes(),
                dtype=np.float16,
            )[0]
        )

        out_idx = 0
        is_idx = 0
        for outer_half in range(2):
            qs_chunk = qs_full[outer_half * 32 : (outer_half + 1) * 32]
            for shift_idx in range(4):
                shift = shift_idx * 2
                sc_byte_left = int(scales[is_idx])
                is_idx += 1
                dl_left = d * float(sc_byte_left & 0x0F)
                ml_left = dmin * float(sc_byte_left >> 4)
                for l in range(16):
                    q2 = (int(qs_chunk[l]) >> shift) & 3
                    out[b, out_idx] = dl_left * q2 - ml_left
                    out_idx += 1
                sc_byte_right = int(scales[is_idx])
                is_idx += 1
                dl_right = d * float(sc_byte_right & 0x0F)
                ml_right = dmin * float(sc_byte_right >> 4)
                for l in range(16):
                    q2 = (int(qs_chunk[l + 16]) >> shift) & 3
                    out[b, out_idx] = dl_right * q2 - ml_right
                    out_idx += 1

    return out.reshape(out_dim, in_dim)


def _kquant_matmul_q2_k(x: mx.array, w_packed_np: np.ndarray) -> mx.array:
    """Wrap mx.quantized_matmul with mode='kquant' and Q2_K (gs=256, bits=2)."""
    return mx.quantized_matmul(
        x,
        mx.array(w_packed_np),
        scales=_scales_placeholder(),
        biases=None,
        transpose=True,
        group_size=256,
        bits=2,
        mode="kquant",
        kquant_type="q2_k",
    )


class TestKQuantQ4_K(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q4_K (mode='kquant', gs=256, bits=4).

    Q4_K has hierarchical scales: a per-super-block (d, dmin) envelope and
    8 sub-blocks of 32 weights each, each with a 6-bit (sub-scale, sub-min)
    pair packed into the 12-byte `scales` field. Validates the dispatch
    chain and the kq_q4_k_qmv / kq_q4_k_qmv_fast Metal kernels.
    """

    quantize_matrix = staticmethod(_quantize_q4_k_matrix)
    dequantize_matrix = staticmethod(_dequantize_q4_k_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q4_k)
    group_size = 256
    bits = 4
    block_bytes = Q4_K_BLOCK_BYTES
    kquant_type = "q4_k"
    general_out_dim = 13
    general_in_dim = 256
    qmm_n_shapes = ((8, 256, 1024), (17, 256, 1024))

    def test_random_matmul_large(self):
        # 4096 x 2048 fp16 -- closer to a real generation-time linear layer.
        self._check_random_matmul(4096, 2048, mx.float16)


class TestKQuantQ5_K(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q5_K (mode='kquant', gs=256, bits=5).

    Q5_K is Q4_K plus a high-bit array `qh[32]` between the scales and the
    qs payload. Each weight is 5 bits (low 4 from qs nibble, high 1 from qh).
    The qh layout is transposed: bit `sb` of qh[l] is the high bit of weight
    `sb*32 + l`. Validates the dispatch chain and the kq_q5_k_qmv /
    kq_q5_k_qmv_fast Metal kernels.
    """

    quantize_matrix = staticmethod(_quantize_q5_k_matrix)
    dequantize_matrix = staticmethod(_dequantize_q5_k_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q5_k)
    group_size = 256
    bits = 5
    block_bytes = Q5_K_BLOCK_BYTES
    kquant_type = "q5_k"
    general_out_dim = 13
    general_in_dim = 256
    qmm_n_shapes = ((8, 256, 1024), (17, 256, 1024))

    def test_random_matmul_large(self):
        self._check_random_matmul(4096, 2048, mx.float16)


class TestKQuantQ6_K(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q6_K (mode='kquant', gs=256, bits=6).

    Q6_K has 16 sub-blocks of 16 weights each, signed int8 per-sub-block
    scales, and a single fp16 super-block envelope `d`. Dequant is symmetric
    (no dmin): `w = d * scale[j] * (q6 - 32)`. Wire format reverses Q4_K/Q5_K
    field order: payload (`ql`/`qh`) comes first, envelope (`d`) is at the
    end of the 210-byte block. Validates the dispatch chain and the
    kq_q6_k_qmv / kq_q6_k_qmv_fast Metal kernels.
    """

    quantize_matrix = staticmethod(_quantize_q6_k_matrix)
    dequantize_matrix = staticmethod(_dequantize_q6_k_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q6_k)
    group_size = 256
    bits = 6
    block_bytes = Q6_K_BLOCK_BYTES
    kquant_type = "q6_k"
    general_out_dim = 13
    general_in_dim = 256
    qmm_n_shapes = ((8, 256, 1024), (17, 256, 1024))

    def test_random_matmul_large(self):
        self._check_random_matmul(4096, 2048, mx.float16)


class TestKQuantQ3_K(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q3_K (mode='kquant', gs=256, bits=3).

    Q3_K is symmetric (no dmin) with 16 sub-blocks of 16 weights, signed
    6-bit-biased per-sub-block scales packed via a 4-word bit-shuffle into
    a 12-byte field, a 2-bit qs payload (64 bytes), and a 1-bit hmask
    (32 bytes) selecting between q3 ranges [0, 3] (hmask SET) and [-4, -1]
    (hmask CLEAR). Validates the dispatch chain and the kq_q3_k_qmv /
    kq_q3_k_qmv_fast Metal kernels.
    """

    quantize_matrix = staticmethod(_quantize_q3_k_matrix)
    dequantize_matrix = staticmethod(_dequantize_q3_k_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q3_k)
    group_size = 256
    bits = 3
    block_bytes = Q3_K_BLOCK_BYTES
    kquant_type = "q3_k"
    general_out_dim = 13
    general_in_dim = 256
    qmm_t_shapes = ((4, 64, 1024), (17, 48, 1024), (64, 64, 2048), (8, 256, 1024))
    qmm_n_shapes = ((8, 256, 1024), (17, 256, 1024))

    def test_random_matmul_large(self):
        self._check_random_matmul(4096, 2048, mx.float16)

    def test_q3_k_scale_unpack_fixture(self):
        """Bit-exact validation of the 4-word scale unpack vs. a hand-computed
        expected output. De-risks the trickiest piece in this codec."""
        scales12 = np.array(
            [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x55, 0xAA, 0x33, 0xCC],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                0x12,
                0x24,
                0x36,
                0x08,
                0x1A,
                0x2C,
                0x0E,
                0x30,
                0x11,
                0x23,
                0x35,
                0x07,
                0x19,
                0x2B,
                0x0D,
                0x3F,
            ],
            dtype=np.uint8,
        )
        actual = _unpack_scale_q3k(scales12)
        np.testing.assert_array_equal(actual, expected)
        # Bijective round-trip: pack(unpack(x)) == x for any 12-byte input.
        np.testing.assert_array_equal(_pack_scale_q3k(actual), scales12)

    def test_q3_k_hmask_inversion(self):
        """Hand-crafted block: hmask all-set vs all-clear with a fixed qs/scale
        pattern. Catches Metal kernel `(hm & m) ? 0 : 4` inversion bugs."""
        # d=1.0 fp16, all 16 scales = 33 (encodes effective scale = 33-32 = 1),
        # qs = 0x55 (each byte holds 4 weights with q2=1 at every shift),
        # hmask = 0xFF (h=0, q3=q2-0=1) vs 0x00 (h=4, q3=q2-4=-3).
        sc6 = np.full(16, 33, dtype=np.uint8)
        scales12 = _pack_scale_q3k(sc6)
        for hmask_val, expected_w in ((0xFF, 1.0), (0x00, -3.0)):
            block = np.zeros(Q3_K_BLOCK_BYTES, dtype=np.uint8)
            block[Q3_K_HMASK_OFFSET : Q3_K_HMASK_OFFSET + 32] = hmask_val
            block[Q3_K_QS_OFFSET : Q3_K_QS_OFFSET + 64] = 0x55
            block[Q3_K_SCALES_OFFSET : Q3_K_SCALES_OFFSET + 12] = scales12
            d_fp16 = np.float16(1.0)
            block[Q3_K_D_OFFSET : Q3_K_D_OFFSET + 2] = np.frombuffer(
                d_fp16.tobytes(), dtype=np.uint8
            )
            W_q = block.reshape(1, Q3_K_BLOCK_BYTES)
            # Reference round-trip.
            W_ref = _dequantize_q3_k_matrix(W_q, 256)
            np.testing.assert_allclose(W_ref, expected_w, atol=1e-4)
            # Metal kernel via qmv with x = ones (sum should be 256 * expected).
            x_np = np.ones(256, dtype=np.float32)
            y = _kquant_matmul_q3_k(mx.array(x_np), W_q)
            mx.eval(y)
            y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
            np.testing.assert_allclose(
                y_np,
                np.array([256.0 * expected_w]),
                rtol=1e-3,
                err_msg=f"hmask=0x{hmask_val:02X}",
            )


class TestKQuantQ2_K(_KQuantCodecTestMixin, mlx_tests.MLXTestCase):
    """End-to-end tests for K-quant Q2_K (mode='kquant', gs=256, bits=2).

    Q2_K is asymmetric with 16 sub-blocks of 16 weights, 4-bit (scale, min)
    nibble pairs per sub-block, fp16 (d, dmin) super-block envelope, and a
    2-bit qs payload (64 bytes). Trivial nibble-split scales -- no
    bit-shuffle. Dequant: w = d * (sc & 0xF) * q2 - dmin * (sc >> 4).
    Validates the dispatch chain and the kq_q2_k_qmv / kq_q2_k_qmv_fast
    Metal kernels.
    """

    quantize_matrix = staticmethod(_quantize_q2_k_matrix)
    dequantize_matrix = staticmethod(_dequantize_q2_k_matrix)
    matmul_fn = staticmethod(_kquant_matmul_q2_k)
    group_size = 256
    bits = 2
    block_bytes = Q2_K_BLOCK_BYTES
    kquant_type = "q2_k"
    general_out_dim = 13
    general_in_dim = 256
    qmm_t_shapes = ((4, 64, 1024), (17, 48, 1024), (64, 64, 2048), (8, 256, 1024))
    qmm_n_shapes = ((8, 256, 1024), (17, 256, 1024))

    def test_random_matmul_large(self):
        self._check_random_matmul(4096, 2048, mx.float16)

    def test_q2_k_zero_min(self):
        """Hand-crafted block: dmin=0 collapses asymmetric -> symmetric. Catches
        Metal kernel mishandling of the (sc >> 4) min term separate from sc."""
        # d=1, dmin=0, scales[i] = 0x01 (sc=1, mn=0), qs all 0x55 -> q2=1.
        # Expected per-weight: 1 * 1 * 1 - 0 * 0 = 1.0.
        block = np.zeros(Q2_K_BLOCK_BYTES, dtype=np.uint8)
        block[Q2_K_SCALES_OFFSET : Q2_K_SCALES_OFFSET + 16] = 0x01
        block[Q2_K_QS_OFFSET : Q2_K_QS_OFFSET + 64] = 0x55
        block[Q2_K_D_OFFSET : Q2_K_D_OFFSET + 2] = np.frombuffer(
            np.float16(1.0).tobytes(), dtype=np.uint8
        )
        block[Q2_K_DMIN_OFFSET : Q2_K_DMIN_OFFSET + 2] = np.frombuffer(
            np.float16(0.0).tobytes(), dtype=np.uint8
        )
        W_q = block.reshape(1, Q2_K_BLOCK_BYTES)
        W_ref = _dequantize_q2_k_matrix(W_q, 256)
        np.testing.assert_allclose(W_ref, 1.0, atol=1e-4)
        x_np = np.ones(256, dtype=np.float32)
        y = _kquant_matmul_q2_k(mx.array(x_np), W_q)
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        np.testing.assert_allclose(y_np, np.array([256.0]), rtol=1e-3)


class TestKQuantNax(mlx_tests.MLXTestCase):
    """End-to-end correctness checks for the kquant NAX path.

    Each codec's existing per-class tests already exercise a mix of NAX-
    eligible and ALU-eligible shapes (large K -> NAX, K=96 -> ALU); these
    tests target NAX-specific concerns: cross-codec correctness and the
    `gather_qmm_rhs_nax` route (rhs-only gather).
    """

    # (codec, group_size, bits, quantize_matrix, dequantize_matrix)
    NAX_CODECS = [
        ("q8_0", 32, 8, _quantize_q8_0_matrix, _dequantize_q8_0_matrix),
        ("q5_1", 32, 5, _quantize_q5_1_matrix, _dequantize_q5_1_matrix),
        ("q4_k", 256, 4, _quantize_q4_k_matrix, _dequantize_q4_k_matrix),
        ("q5_k", 256, 5, _quantize_q5_k_matrix, _dequantize_q5_k_matrix),
        ("q6_k", 256, 6, _quantize_q6_k_matrix, _dequantize_q6_k_matrix),
        ("q3_k", 256, 3, _quantize_q3_k_matrix, _dequantize_q3_k_matrix),
        ("q2_k", 256, 2, _quantize_q2_k_matrix, _dequantize_q2_k_matrix),
    ]

    def test_nax_matmul_correctness(self):
        """For each NAX-supported codec run a NAX-eligible quantized_matmul
        (K % 64 == 0, transpose=true, fp16) and compare against a numpy
        reference computed from the dequantized weights. Tolerances follow
        _MATMUL_REL_TOL[fp16] (5e-3) -- NAX MMA rounds slightly differently
        from ALU simdgroup MMA but stays well within bound.
        """
        rng = np.random.default_rng(11)
        # K=512 is divisible by both 64 (NAX) and 256 (super-block codecs).
        K, N, M = 512, 64, 64
        for codec, gs, bits, qfn, dqfn in self.NAX_CODECS:
            W = rng.standard_normal((N, K)).astype(np.float32) * 0.25
            W_q = qfn(W)
            W_ref = dqfn(W_q, K)
            x_np = rng.standard_normal((M, K)).astype(np.float32)
            ref = x_np @ W_ref.T
            for dtype in (mx.float16, mx.bfloat16):
                with self.subTest(codec=codec, dtype=dtype):
                    x = mx.array(x_np).astype(dtype)
                    y = mx.quantized_matmul(
                        x,
                        mx.array(W_q),
                        scales=_scales_placeholder(),
                        biases=None,
                        transpose=True,
                        group_size=gs,
                        bits=bits,
                        mode="kquant",
                        kquant_type=codec,
                    )
                    mx.eval(y)
                    y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                    denom = max(1e-8, float(np.max(np.abs(ref))))
                    rel = float(np.max(np.abs(y_np - ref))) / denom
                    self.assertLess(
                        rel,
                        _MATMUL_REL_TOL[dtype],
                        msg=f"{codec} {dtype}: rel={rel:.3e}",
                    )

    def test_nax_gather_qmm(self):
        """Gather variants. `gather_qmm` with both lhs+rhs indices exercises
        `gather_qmm_nax` (T variant). With only rhs_indices via mx.gather_qmm
        + sorted_indices=True + M=1 the GatherQMM dispatch routes through
        `gather_qmm_rhs_nax`."""
        rng = np.random.default_rng(13)
        # E experts, B routed positions, K NAX-aligned, N tile-aligned.
        E, B, K, N = 4, 32, 256, 64
        codec, gs, bits, qfn, dqfn = (
            "q4_k",
            256,
            4,
            _quantize_q4_k_matrix,
            _dequantize_q4_k_matrix,
        )
        Ws_q, Ws_ref = [], []
        for _ in range(E):
            W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
            wq = qfn(W)
            Ws_q.append(wq)
            Ws_ref.append(dqfn(wq, K))
        w_stacked = mx.array(np.stack(Ws_q, axis=0))

        # -- A. gather_qmm with both indices (M=32 -> gather_qmm_nax) --
        rhs_idx = rng.integers(0, E, size=B).astype(np.uint32)
        lhs_idx = np.arange(B, dtype=np.uint32)
        M = 32
        X = rng.standard_normal((B, M, K)).astype(np.float32)
        ref = np.stack([X[b] @ Ws_ref[rhs_idx[b]].T for b in range(B)])
        x = mx.array(X).astype(mx.float16)
        y = mx.gather_qmm(
            x,
            w_stacked,
            scales=mx.zeros((E, 1, 1), dtype=mx.uint8),
            biases=None,
            lhs_indices=mx.array(lhs_idx),
            rhs_indices=mx.array(rhs_idx),
            transpose=True,
            group_size=gs,
            bits=bits,
            mode="kquant",
            kquant_type=codec,
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        self.assertLess(
            rel, _MATMUL_REL_TOL[mx.float16] * 5, msg=f"gather_qmm rel={rel:.3e}"
        )

        # -- B. gather_qmm with only rhs (M=1, sorted) -> gather_qmm_rhs_nax --
        # The eval_gpu fast path requires B>=16, B/E>=4, right_sorted=True.
        rhs_sorted = np.sort(rng.integers(0, E, size=B).astype(np.uint32))
        M = 1
        X = rng.standard_normal((B, M, K)).astype(np.float32)
        ref_rhs = np.stack([X[b] @ Ws_ref[rhs_sorted[b]].T for b in range(B)])
        x = mx.array(X).astype(mx.float16)
        y = mx.gather_qmm(
            x,
            w_stacked,
            scales=mx.zeros((E, 1, 1), dtype=mx.uint8),
            biases=None,
            lhs_indices=None,
            rhs_indices=mx.array(rhs_sorted),
            transpose=True,
            group_size=gs,
            bits=bits,
            mode="kquant",
            kquant_type=codec,
            sorted_indices=True,
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
        denom = max(1e-8, float(np.max(np.abs(ref_rhs))))
        rel = float(np.max(np.abs(y_np - ref_rhs))) / denom
        self.assertLess(
            rel, _MATMUL_REL_TOL[mx.float16] * 5, msg=f"gather_qmm_rhs rel={rel:.3e}"
        )


def _make_kquant_linear(
    K, N, codec, group_size, bits, *, weight_packed_np, bias_np=None
):
    """Construct an nn.QuantizedLinear with pre-built kquant wire-format weights.

    Constructs in affine mode and rewrites the layer state to kquant, since
    __init__ generates random weights but tests need specific byte patterns.
    """
    ql = nn.QuantizedLinear(K, N, bias=(bias_np is not None), mode="affine")
    ql.mode = "kquant"
    ql.kquant_type = codec
    ql.group_size = group_size
    ql.bits = bits
    ql.weight = mx.array(weight_packed_np)
    ql.scales = _scales_placeholder()
    ql.biases = None
    if bias_np is not None:
        ql.bias = mx.array(bias_np)
    return ql


class TestKQuantQuantizedLinear(mlx_tests.MLXTestCase):
    """End-to-end tests for nn.QuantizedLinear under mode="kquant".

    The mx.* op level is already covered by per-codec test classes above;
    these tests guard the layer wrapper itself so a future change to
    QuantizedLinear's storage or dispatch can't silently break kquant mode.
    """

    _CODEC_MATRIX = (
        # (codec, group_size, bits, quantize_fn, dequantize_fn)
        ("q8_0", 32, 8, _quantize_q8_0_matrix, _dequantize_q8_0_matrix),
        ("q4_k", 256, 4, _quantize_q4_k_matrix, _dequantize_q4_k_matrix),
        ("q6_k", 256, 6, _quantize_q6_k_matrix, _dequantize_q6_k_matrix),
    )

    def test_quantized_linear_forward(self):
        K, N, M = 1024, 64, 4
        rng = np.random.default_rng(0)
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        for codec, gs, bits, quantize_fn, dequantize_fn in self._CODEC_MATRIX:
            W_q = quantize_fn(W)
            W_ref = dequantize_fn(W_q, K)
            ref = x_np @ W_ref.T  # (M, N)

            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(codec=codec, dtype=dtype):
                    ql = _make_kquant_linear(
                        K, N, codec, gs, bits, weight_packed_np=W_q
                    )

                    x = mx.array(x_np)
                    if dtype != mx.float32:
                        x = x.astype(dtype)
                    y = ql(x)
                    mx.eval(y)

                    y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                    denom = max(1e-8, float(np.max(np.abs(ref))))
                    rel = float(np.max(np.abs(y_np - ref))) / denom
                    out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
                    tol = _MATMUL_REL_TOL[out_dtype]
                    self.assertLess(
                        rel,
                        tol,
                        msg=f"codec={codec} dtype={dtype}: "
                        f"rel={rel:.3e} tol={tol:.0e}",
                    )

    def test_quantized_linear_with_bias(self):
        K, N, M = 1024, 64, 4
        rng = np.random.default_rng(1)
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        x_np = rng.standard_normal((M, K)).astype(np.float32)
        bias_np = rng.standard_normal((N,)).astype(np.float32) * 0.5

        W_q = _quantize_q8_0_matrix(W)
        W_ref = _dequantize_q8_0_matrix(W_q, K)
        ref = x_np @ W_ref.T + bias_np

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                ql = _make_kquant_linear(
                    K, N, "q8_0", 32, 8, weight_packed_np=W_q, bias_np=bias_np
                )
                self.assertIn("bias", ql)

                x = mx.array(x_np)
                if dtype != mx.float32:
                    x = x.astype(dtype)
                y = ql(x)
                mx.eval(y)

                y_np = np.asarray(y.astype(mx.float32)).astype(np.float32)
                denom = max(1e-8, float(np.max(np.abs(ref))))
                rel = float(np.max(np.abs(y_np - ref))) / denom
                out_dtype = mx.bfloat16 if dtype == mx.float32 else dtype
                tol = _MATMUL_REL_TOL[out_dtype]
                self.assertLess(
                    rel, tol, msg=f"dtype={dtype} bias: rel={rel:.3e} tol={tol:.0e}"
                )

    def test_quantized_linear_save_load_roundtrip(self):
        K, N, M = 1024, 64, 4
        rng = np.random.default_rng(2)
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
        x_np = rng.standard_normal((M, K)).astype(np.float32)
        W_q = _quantize_q8_0_matrix(W)

        ql = _make_kquant_linear(K, N, "q8_0", 32, 8, weight_packed_np=W_q)
        x = mx.array(x_np).astype(mx.float16)
        y_orig = ql(x)
        mx.eval(y_orig)

        tdir = tempfile.TemporaryDirectory()
        try:
            path = os.path.join(tdir.name, "ql.safetensors")
            ql.save_weights(path)

            ql2 = _make_kquant_linear(
                K, N, "q8_0", 32, 8, weight_packed_np=np.zeros_like(W_q)
            )
            ql2.load_weights(path)

            y_reload = ql2(x)
            mx.eval(y_reload)
            self.assertTrue(mx.array_equal(y_orig, y_reload).item())
        finally:
            tdir.cleanup()

    def test_quantized_linear_repr(self):
        K, N = 1024, 64
        gs, bits = 256, 4
        # Q4_K: 144 bytes per 256-weight super-block.
        bytes_per_row = K * 144 // 256
        ql = _make_kquant_linear(
            K,
            N,
            "q4_k",
            gs,
            bits,
            weight_packed_np=np.zeros((N, bytes_per_row), dtype=np.uint8),
        )

        r = repr(ql)
        for needle in (
            "mode=kquant",
            "kquant_type=q4_k",
            f"input_dims={K}",
            f"output_dims={N}",
            f"group_size={gs}",
            f"bits={bits}",
        ):
            self.assertIn(needle, r, msg=f"missing {needle!r} in repr: {r}")


class TestKQuantVJP(mlx_tests.MLXTestCase):
    """Gradient tests for kquant quantized_matmul and gather_qmm."""

    def test_qmm_vjp_x(self):
        """VJP wrt x: d/dx (x @ dequant(w).T) cotangent should match."""
        rng = np.random.default_rng(42)
        M, K, N = 4, 256, 32
        W_fp = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        W_q_np = _quantize_q8_0_matrix(W_fp)
        w_q = mx.array(W_q_np)
        x = mx.random.normal((M, K))
        c = mx.ones((M, N))

        def fn(x_):
            return mx.quantized_matmul(
                x_,
                w_q,
                scales=_scales_placeholder(),
                transpose=True,
                group_size=32,
                bits=8,
                mode="kquant",
                kquant_type="q8_0",
            )

        _, vjp_out = mx.vjp(fn, primals=(x,), cotangents=(c,))

        expected = mx.quantized_matmul(
            c,
            w_q,
            scales=_scales_placeholder(),
            transpose=False,
            group_size=32,
            bits=8,
            mode="kquant",
            kquant_type="q8_0",
        )
        self.assertTrue(mx.allclose(vjp_out[0], expected, atol=1e-4))

    def test_qmm_jvp_x(self):
        """JVP wrt x: tangent through quantized_matmul for both transposes."""
        rng = np.random.default_rng(42)
        M, K, N = 4, 256, 32
        W_fp = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        W_q_np = _quantize_q8_0_matrix(W_fp)
        w_q = mx.array(W_q_np)
        x = mx.random.normal((M, K))
        x_tan = mx.ones((M, K))

        for transpose in (True, False):
            with self.subTest(transpose=transpose):
                W_shape = (N, K) if transpose else (K, N)
                W_local = rng.standard_normal(W_shape).astype(np.float32) * 0.1
                W_q_local = _quantize_q8_0_matrix(W_local)
                w_q_local = mx.array(W_q_local)

                def fn(x_):
                    return mx.quantized_matmul(
                        x_,
                        w_q_local,
                        scales=_scales_placeholder(),
                        transpose=transpose,
                        group_size=32,
                        bits=8,
                        mode="kquant",
                        kquant_type="q8_0",
                    )

                _, jvp_out = mx.jvp(fn, primals=(x,), tangents=(x_tan,))
                expected = mx.quantized_matmul(
                    x_tan,
                    w_q_local,
                    scales=_scales_placeholder(),
                    transpose=transpose,
                    group_size=32,
                    bits=8,
                    mode="kquant",
                    kquant_type="q8_0",
                )
                self.assertTrue(mx.allclose(jvp_out[0], expected, atol=1e-4))

    def test_gather_qmm_vjp_x(self):
        """VJP wrt x through gather_qmm."""
        rng = np.random.default_rng(7)
        M, K, N = 4, 256, 32
        B = 2
        W_fp = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        W_q_np = _quantize_q8_0_matrix(W_fp)
        w_q = mx.array(W_q_np)
        w_q = mx.broadcast_to(w_q[None], (B, N, W_q_np.shape[1]))
        x = mx.random.normal((B, M, K))
        rhs_indices = mx.array([[0], [1]])

        def fn(x_):
            return mx.gather_qmm(
                x_,
                w_q,
                scales=_scales_placeholder(),
                rhs_indices=rhs_indices,
                transpose=True,
                group_size=32,
                bits=8,
                mode="kquant",
                kquant_type="q8_0",
            )

        out = fn(x)
        c = mx.ones_like(out)
        _, vjp_out = mx.vjp(fn, primals=(x,), cotangents=(c,))
        mx.eval(vjp_out[0])
        self.assertEqual(vjp_out[0].shape, x.shape)


class TestKQuantQuantizedEmbedding(mlx_tests.MLXTestCase):
    """Tests for nn.QuantizedEmbedding under mode='kquant'."""

    def test_embedding_forward(self):
        """Construct a kquant QuantizedEmbedding and verify lookup."""
        num_embeddings, dims = 16, 256
        rng = np.random.default_rng(99)
        W = rng.standard_normal((num_embeddings, dims)).astype(np.float32) * 0.3
        W_q = _quantize_q8_0_matrix(W)
        W_ref = _dequantize_q8_0_matrix(W_q, dims)

        qe = nn.QuantizedEmbedding(num_embeddings, dims, mode="affine")
        qe.mode = "kquant"
        qe.kquant_type = "q8_0"
        qe.group_size = 32
        qe.bits = 8
        qe.weight = mx.array(W_q)
        qe.scales = _scales_placeholder()
        qe.biases = None

        indices = mx.array([0, 3, 7, 15])
        y = qe(indices)
        mx.eval(y)

        y_np = np.asarray(y.astype(mx.float32))
        ref = W_ref[np.array([0, 3, 7, 15])]
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        self.assertLess(
            rel, _MATMUL_REL_TOL[mx.bfloat16], msg=f"embedding rel={rel:.3e}"
        )


class TestKQuantEdgeCases(mlx_tests.MLXTestCase):
    """Edge case tests for kquant dispatch paths."""

    def test_codec_geometry_consistency(self):
        """Python _KQUANT_CODEC_GEOMETRY must match C++ kquant_codec_by_name."""
        from mlx.nn.layers.quantized import _KQUANT_CODEC_GEOMETRY

        for codec, (wpb, bpb) in _KQUANT_CODEC_GEOMETRY.items():
            with self.subTest(codec=codec):
                K = wpb * 4
                packed = mx.zeros((1, (K // wpb) * bpb), dtype=mx.uint8)
                out = mx.dequantize(
                    packed,
                    mx.zeros((1,), dtype=mx.uint8),
                    None,
                    mode="kquant",
                    kquant_type=codec,
                )
                self.assertEqual(out.shape[-1], K)

    def test_qmv_quad_nyi_raises(self):
        """K=64 or K=128 with M<vector_limit should raise NYI.

        Convert to a correctness test once qmv_quad is implemented.
        """
        rng = np.random.default_rng(50)
        for K in (64, 128):
            with self.subTest(K=K):
                N = 32
                W = rng.standard_normal((N, K)).astype(np.float32) * 0.3
                W_q = _quantize_q8_0_matrix(W)
                x = mx.random.normal((1, K))
                with self.assertRaises(RuntimeError):
                    y = mx.quantized_matmul(
                        x,
                        mx.array(W_q),
                        scales=_scales_placeholder(),
                        transpose=True,
                        group_size=32,
                        bits=8,
                        mode="kquant",
                        kquant_type="q8_0",
                    )
                    mx.eval(y)

    def test_splitk_path(self):
        """Exercise the split-k qmm_t path (small M, large K, non-batched)."""
        rng = np.random.default_rng(77)
        K, N, M = 4096, 64, 8
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        W_q = _quantize_q8_0_matrix(W)
        W_ref = _dequantize_q8_0_matrix(W_q, K)
        x_np = rng.standard_normal((M, K)).astype(np.float32)
        ref = x_np @ W_ref.T

        y = mx.quantized_matmul(
            mx.array(x_np),
            mx.array(W_q),
            scales=_scales_placeholder(),
            transpose=True,
            group_size=32,
            bits=8,
            mode="kquant",
            kquant_type="q8_0",
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32))
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        self.assertLess(rel, _MATMUL_REL_TOL[mx.bfloat16], msg=f"split-k rel={rel:.3e}")

    def test_strided_input(self):
        """quantized_matmul with non-contiguous (strided) x."""
        rng = np.random.default_rng(11)
        K, N = 512, 64
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        W_q = _quantize_q8_0_matrix(W)
        W_ref = _dequantize_q8_0_matrix(W_q, K)

        x_full = mx.array(rng.standard_normal((4, K * 2)).astype(np.float32))
        x_strided = x_full[:, ::2]  # non-contiguous slice

        ref = np.asarray(x_strided) @ W_ref.T
        y = mx.quantized_matmul(
            x_strided,
            mx.array(W_q),
            scales=_scales_placeholder(),
            transpose=True,
            group_size=32,
            bits=8,
            mode="kquant",
            kquant_type="q8_0",
        )
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32))
        denom = max(1e-8, float(np.max(np.abs(ref))))
        rel = float(np.max(np.abs(y_np - ref))) / denom
        self.assertLess(rel, _MATMUL_REL_TOL[mx.bfloat16], msg=f"strided rel={rel:.3e}")


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
