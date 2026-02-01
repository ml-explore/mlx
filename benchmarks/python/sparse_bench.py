# Copyright © 2025 Apple Inc.

import argparse

import mlx.core as mx
import numpy as np
from time_utils import time_fn

N = 1024
SPARSITY = 0.05


def to_csr(dense_np):
    rows, cols = np.nonzero(dense_np)
    values = dense_np[rows, cols]
    n_rows = dense_np.shape[0]
    row_ptr = np.zeros(n_rows + 1, dtype=np.int32)
    for r in rows:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr).astype(np.int32)
    return row_ptr, cols.astype(np.int32), values


def time_sparse_matmul(dtype, dtype_name):
    np.random.seed(42)

    A_np = np.random.randn(N, N).astype(np.float32)
    mask = np.random.rand(N, N) < SPARSITY
    A_np = A_np * mask

    row_ptr_np, col_indices_np, values_np = to_csr(A_np)
    B_np = np.random.randn(N, N).astype(np.float32)

    row_ptr = mx.array(row_ptr_np)
    col_indices = mx.array(col_indices_np)
    values = mx.array(values_np).astype(dtype)
    B = mx.array(B_np).astype(dtype)
    A = mx.array(A_np).astype(dtype)
    mx.eval(row_ptr, col_indices, values, B, A)

    print(f"  {dtype_name}:")
    time_fn(
        mx.sparse_matmul_csr,
        row_ptr,
        col_indices,
        values,
        B,
        n_rows=N,
        n_cols=N,
        msg=f"sparse ({dtype_name})",
    )
    time_fn(mx.matmul, A, B, msg=f"dense  ({dtype_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sparse matmul CSR benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    print(f"Sparse matmul CSR benchmark ({N}x{N}, {SPARSITY*100:.0f}% nonzero)")

    for dtype, name in [
        (mx.float16, "float16"),
        (mx.bfloat16, "bfloat16"),
        (mx.float32, "float32"),
    ]:
        time_sparse_matmul(dtype, name)
