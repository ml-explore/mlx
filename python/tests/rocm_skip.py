# Tests to skip for ROCm backend
# Based on functionality comparison with CUDA backend

rocm_skip = {
    # Same as CUDA - Block masked matmul NYI
    "TestBlas.test_block_masked_matmul",
    # Same as CUDA - Gather matmul NYI (ROCm throws for M > 1 and N > 1)
    "TestBlas.test_gather_matmul",
    "TestBlas.test_gather_matmul_grad",
    "TestBlas.test_gather_mm_sorted_vjp",
    # Same as CUDA - Segmented matmul NYI
    "TestBlas.test_segmented_mm",
    # ROCm-specific: Complex GEMM not supported in naive fallback
    "TestBlas.test_complex_gemm",
    "TestBlas.test_complex_gemv",
    # ROCm-specific: addmm tolerance too tight for naive GEMM
    "TestBlas.test_addmm",
    "TestBlas.test_addmm_grad",
    # ROCm-specific: empty matmul has issues on unsupported architectures
    "TestBlas.test_empty_matmul",
    # ROCm-specific: batched matrix-vector has precision issues on gfx1011
    "TestBlas.test_matrix_vector_batched",
    # Same as CUDA - Hadamard NYI
    "TestOps.test_hadamard",
    "TestOps.test_hadamard_grad_vmap",
    # Same as CUDA - FFTs NYI
    "TestFFT.test_fft",
    "TestFFT.test_fft_big_powers_of_two",
    "TestFFT.test_fft_contiguity",
    "TestFFT.test_fft_exhaustive",
    "TestFFT.test_fft_grads",
    "TestFFT.test_fft_into_ifft",
    "TestFFT.test_fft_large_numbers",
    "TestFFT.test_fft_shared_mem",
    "TestFFT.test_fftn",
    # Same as CUDA - Lapack ops NYI
    "TestLinalg.test_cholesky",
    "TestLinalg.test_cholesky_inv",
    "TestLinalg.test_eig",
    "TestLinalg.test_eigh",
    "TestLinalg.test_inverse",
    "TestVmap.test_vmap_inverse",
    "TestLinalg.test_lu",
    "TestLinalg.test_lu_factor",
    "TestLinalg.test_pseudo_inverse",
    "TestLinalg.test_qr_factorization",
    "TestInit.test_orthogonal",
    "TestLinalg.test_svd_decomposition",
    "TestVmap.test_vmap_svd",
    "TestLinalg.test_tri_inverse",
    # Same as CUDA - Masked scatter NYI
    "TestOps.test_masked_scatter",
    "TestVmap.test_vmap_masked_scatter",
    "TestArray.test_setitem_with_boolean_mask",
    # Quantization - ROCm has different support than CUDA
    "TestQuantized.test_gather_matmul_grad",
    "TestQuantized.test_gather_qmm",
    "TestQuantized.test_gather_qmm_sorted",
    "TestQuantized.test_gather_qmm_grad",
    "TestQuantized.test_non_multiples",
    "TestQuantized.test_qmm",
    "TestQuantized.test_qmm_jvp",
    "TestQuantized.test_qmm_shapes",
    "TestQuantized.test_qmm_vjp",
    "TestQuantized.test_qmv",
    "TestQuantized.test_fp_qmv",
    "TestQuantized.test_fp_qvm",
    "TestQuantized.test_qvm",
    "TestQuantized.test_qvm_splitk",
    "TestQuantized.test_small_matrix",
    "TestQuantized.test_throw",
    "TestQuantized.test_vjp_scales_biases",
    "TestExportImport.test_export_quantized_model",
    "TestLayers.test_quantized_embedding",
    # ROCm-specific: Complex power has numerical issues
    "TestOps.test_complex_power",
    # ROCm-specific: Complex ops (arctan) has numerical issues
    "TestOps.test_complex_ops",
    # ROCm-specific: Scan operations don't support complex types
    "TestOps.test_logcumsumexp",
    "TestOps.test_scans",
    # ROCm-specific: logsumexp has numerical issues with complex types
    "TestOps.test_logsumexp",
    # ROCm-specific: sort has issues with multi-block sort
    "TestOps.test_sort",
    # ROCm-specific: Complex reduce operations not supported
    "TestReduce.test_nan_propagation_complex64",
    # ROCm-specific: vmap matmul fails on unsupported architectures
    "TestVmap.test_vmap_matmul",
    # ROCm-specific: group_norm has numerical precision issues
    "TestLayers.test_group_norm",
    # ROCm-specific: Custom kernel tests use Metal-specific APIs
    # hip_kernel is available but tests are written for metal_kernel
    "TestFast.test_custom_kernel_args",
    "TestFast.test_custom_kernel_attributes",
    "TestFast.test_custom_kernel_basic",
    "TestFast.test_custom_kernel_helper",
    "TestFast.test_custom_kernel_strides",
    # ROCm-specific: SDPA backward pass falls back to CPU
    # These tests may be slow but should still pass
}
