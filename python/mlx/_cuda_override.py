# Copyright © 2023 Apple Inc.

import mlx.core as mx


def quantized_matmul_new(
    x: mx.array,
    w: mx.array,
    /,
    scales: mx.array,
    biases: mx.array | None = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream=None,
) -> mx.array:
    """
    Perform matrix multiplication with quantized matrix w.

    The quantization uses one floating point scale and bias per group_size elements.
    Each element in w takes `bits` bits and is packed in an unsigned 32 bit integer.

    Args:
        x: Input array of shape (..., in_features)
        w: Quantized matrix packed in uint32, shape (in_features // group_size * group_size // elements_per_int, out_features)
        scales: Scales per group, shape (out_features, in_features // group_size)
        biases: Optional biases per group, same shape as scales
        transpose: If True, compute x @ w.T, else x @ w
        group_size: Number of elements sharing one scale/bias
        bits: Bits per quantized element (typically 4 or 8)
        mode: Quantization mode ('affine' for scale+bias, 'symmetric' for scale only)

    Returns:
        Result of matrix multiplication with shape (..., out_features)
    """
    # Dequantize weights using MLX's dequantization function
    w_dequant = mx.dequantize(
        w, scales, biases, bits=bits, group_size=group_size, mode=mode, stream=stream
    )

    x_dequant = x.astype(dtype=w_dequant.dtype, stream=stream)
    if transpose:
        result = mx.matmul(x_dequant, mx.transpose(w_dequant))
    else:
        result = mx.matmul(x_dequant, w_dequant)

    result_quant = result.astype(dtype=x.dtype, stream=stream)
    return result_quant


def gather_qmm_new(
    x: mx.array,
    w: mx.array,
    /,
    scales: mx.array,
    biases: mx.array | None = None,
    lhs_indices: mx.array | None = None,
    rhs_indices: mx.array | None = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    sorted_indices: bool = False,
    stream=None,
) -> mx.array:
    """
    Perform quantized matrix multiplication with matrix-level gather.
    """

    # Apply gather operations to select specific batch elements
    if lhs_indices is not None:
        # Reshape x to flatten batch dimensions
        # x shape: (..., seq_len, hidden_dim) -> (batch_flat, seq_len, hidden_dim)
        x_flat = x.reshape(-1, *x.shape[-2:])
        x_gathered = x_flat[lhs_indices]  # Use indexing instead of take
    else:
        x_gathered = x

    if rhs_indices is not None:
        # w, scales, biases have shape (num_experts, ...)
        # Gather specific experts
        w_gathered = w[rhs_indices]
        scales_gathered = scales[rhs_indices]
        if biases is not None:
            biases_gathered = biases[rhs_indices]
        else:
            biases_gathered = None
    else:
        w_gathered = w
        scales_gathered = scales
        biases_gathered = biases

    # Dequantize the gathered weights
    w_dequant = mx.dequantize(
        w_gathered,
        scales_gathered,
        biases_gathered,
        bits=bits,
        group_size=group_size,
        mode=mode,
        stream=stream,
    )

    # Convert x to matching dtype
    x_dequant = x_gathered.astype(dtype=w_dequant.dtype, stream=stream)

    # Perform batched matrix multiplication
    # x_dequant: (num_tokens, seq_len, hidden_in)
    # w_dequant: (num_tokens, hidden_in, hidden_out) when transpose=True
    if transpose:
        # Transpose only the last two dimensions, keep all batch dimensions
        # w_dequant might be (..., in_features, out_features)
        # We want (..., out_features, in_features)
        ndim = w_dequant.ndim
        axes = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
        w_t = mx.transpose(w_dequant, axes=axes, stream=stream)
        result = mx.matmul(x_dequant, w_t, stream=stream)
    else:
        result = mx.matmul(x_dequant, w_dequant, stream=stream)

    # Convert back to original dtype
    result_quant = result.astype(dtype=x.dtype, stream=stream)
    return result_quant


if mx.cuda.is_available():
    for mod in sys.modules.values():
        if hasattr(mod, "quantized_matmul") and mod.quantized_matmul is mx.quantized_matmul:
            print("setting attr quantized_matmul")
            mod.quantized_matmul = quantized_matmul_new
        if hasattr(mod, "gather_qmm") and mod.gather_qmm is mx.gather_qmm:
            print("setting attr gather_qmm")
            mod.gather_qmm = gather_qmm_new