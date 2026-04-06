cuda_skip = {
    # Lapack ops NYI
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
    # Quantization NYI
    "TestQuantized.test_gather_matmul_grad",
    "TestQuantized.test_gather_qmm",
    "TestQuantized.test_gather_qmm_sorted",
    "TestQuantized.test_gather_qmm_grad",
}
