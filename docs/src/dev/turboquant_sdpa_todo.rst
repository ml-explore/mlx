TurboQuant SDPA TODO
====================

Issue #3404 proposes native ``mx.fast.scaled_dot_product_attention`` support for TurboQuant-style quantized KV caches. This repo does not currently have a first-class TurboQuant cache state, a Python proxy object for that state, or a C++ SDPA API that accepts the proposed multi-buffer key and value state. The existing quantized kernels cover affine and floating-point quantized matmul paths, but they are not enough to represent the issue's ``TurboQuantProdState`` and ``TurboQuantMSEState`` inputs without inventing an API.

Minimum viable native support should be added only after the public state format is fixed. A conservative implementation plan is:

1. Define a C++ primitive and Python binding for a new inference-only API, likely ``mx.fast.scaled_dot_product_attention_tq``. Avoid overloading the existing SDPA function until there is an actual MLX array or object type that can represent quantized KV state without ambiguity.
2. Mirror the issue's required buffers explicitly: key norms, key MSE indices, key residual norms, key QJL signs, value norms, value indices, key codebook, and value codebook. Validate shapes, bit widths, and head dimensions before dispatch.
3. Implement decode first in ``sdpa_vector.h`` for ``qL <= 8``. This is the smallest native path with the clearest extension point and avoids full-cache dequantization.
4. Extend the steel full-attention kernels only after decode has correctness tests against a dequantized reference. Prefill needs tiled inline dequantization in both score and value phases, plus GQA, causal masking, and chunked long-context behavior.
5. Add tests that construct a tiny quantized state, dequantize it with a reference implementation, and compare native TQ SDPA output against ordinary SDPA. Mark larger performance and long-context cases as skipped until the native kernel exists.

This branch intentionally does not add a placeholder Python-only Metal kernel or a fake quantized input detector. Those would not satisfy the issue's native-support goal and would make the public API harder to stabilize.
