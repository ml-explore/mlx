# CPU I8MM affine-QMM evidence prototype

This directory contains a local, evidence-only exploration following the
closed [mlx#4197](https://github.com/ml-explore/mlx/issues/4197) discussion. It
may inform a future Apple I8MM follow-up to Highway
[PR #3019](https://github.com/ml-explore/mlx/pull/3019). It is not compiled into
MLX, does not change a production dispatch, and is not a claim that dynamic-int8
activations preserve model quality.

## What it measures

The prototype uses MLX's transposed affine 8-bit geometry:

```text
x:       float32 [M, K]
w:       packed uint32 / uint8 bytes [N, K], 8-bit affine values
scales:  float32 [N, K / 64]
biases:  float32 [N, K / 64]
```

For each `(row, group-64)` it quantizes the activation once to symmetric int8,
retains the exact floating-point group sum for the affine-bias correction, and
reuses that identical prepared row and packed-weight layout in two controls:

- a non-I8MM U8×I8 NEON dot-product path (`vusdotq_s32`); and
- the 2×2 U8×I8 I8MM path (`vusmmlaq_s32`).

It reports these components separately:

1. activation quantization only;
2. dot-product and 2×2 I8MM compute using the same preparation;
3. each approximate method end to end, including one preparation per row; and
4. an intentionally repeated-preparation control.

The fourth measurement represents the current **threaded M>1 branch** in
PR #3019: `QmmTInt8HighwayTyped` passes `preq` only for the threaded `M == 1`
branch. Its threaded M>1 branch calls
`QmmTInt8RowFromActivation(..., nullptr)` inside every output-column chunk,
which repeats activation preparation. The single-threaded M>1 branch does not
have that per-chunk repetition. The prototype never credits fixing this
scheduling/data-lifetime issue to I8MM.

The native instruction is `vusmmlaq_s32` (U8 weights times I8 activations).
Highway has a U8*I8 dot primitive and a signed `PerBlock2x2MatMul`, but not the
mixed-type 2x2 wrapper required to express this exact tile directly. A signed
2x2 extraction would need either a mixed Highway primitive or a measured,
semantics-preserving recentering design:

```text
dot(u8_w, i8_x) = dot(i8(w - 128), i8_x) + 128 * sum(i8_x)
```

Recentring inside the hot loop is not assumed free. The native prototype exists
to measure the math and the reuse contract before either extraction is proposed.

## Build and light correctness test

This is Apple arm64/M5-only. Its CMake target is opt-in and defaults to off.
Build it through CMake: that target hashes the frozen C++ source and embeds the
SHA256 in the candidate. Direct compilation is intentionally unsupported,
because it cannot provide that diagnostic candidate-source identity.

```bash
cmake -S . -B /tmp/mlx-i8mm-build \
  -DMLX_BUILD_BENCHMARKS=ON -DMLX_BUILD_I8MM_EVIDENCE=ON
cmake --build /tmp/mlx-i8mm-build --target quantized_i8mm_evidence
/tmp/mlx-i8mm-build/benchmarks/cpp/quantized_i8mm_evidence --self-test
/tmp/mlx-i8mm-build/benchmarks/cpp/quantized_i8mm_evidence --identity
```

`--identity` prints the CMake-embedded source SHA256. This binds the candidate
diagnostically to the canonical C++ source supplied to the Python runner; it
does not prove any MLX binary or source provenance and cannot enable
throughput.

The self-test is deliberately small. It proves both the dot-product control and
the 2×2 I8MM integer mappings exactly against scalar integer dots. It then
checks both final fp32 affine results against the same scalar prequantized
oracle and checks the method-to-method metric decomposition. It covers M/N
tails, activation reuse across chunks, all-zero activations, symmetric
saturation, NaN/Inf rejection, and invalid K-tail rejection. It does not
establish model-level quality or production performance.

## Diagnostic receipt

The runner records the following MLX provenance as record-only, unbound data:

- the loaded MLX extension path and SHA256;
- every discoverable loaded `libmlx` path and SHA256;
- source HEAD and HEAD-tree identity;
- hashes of the tracked binary diff, porcelain status, untracked-file
  manifest, and combined source state.

Current MLX build plumbing cannot independently attest that those loaded files
were built from that source state. The receipt therefore makes no binding
between them. No proof-file flag or other opt-in exists: throughput is
permanently ineligible, and all timing values are diagnostic only.

```bash
cd /Users/pjb/git/worktrees/mlx-i8mm
python benchmarks/python/quantized_i8mm_evidence.py \
  --candidate /tmp/mlx-i8mm-build/benchmarks/cpp/quantized_i8mm_evidence \
  --M 1,2,8,64 --N 4096 --K 4096 \
  --warmups 3 --outer-trials 9 \
  --mlx-source-checkout /Users/pjb/git/worktrees/mlx-i8mm \
  --canonical-candidate-source benchmarks/cpp/quantized_i8mm_evidence.cpp \
  --output /tmp/mlx-i8mm-4197-receipt.json
```

Python independently hashes the supplied canonical source and rejects a
candidate whose `--identity` SHA256 differs. It also records the actual
candidate binary SHA256. This is a diagnostic candidate-source binding only,
not MLX provenance.

For each shape, Python first starts one `--correctness` candidate subprocess.
It writes the dot and I8MM outputs and computes numerical metrics. Python then
uses `--timing-only` candidate subprocesses for the alternating MLX-first and
candidate-first outer trials. Timing-only mode writes no outputs and does not
compute the scalar reference or correctness metrics before timing. The dot and
I8MM methods remain internally alternated and warmed. At least one warmup and
five outer trials are required.

The receipt reports a same-preparation implementation elapsed-time ratio:
`same_preparation_elapsed_time_ratio_dot_control_over_i8mm_2x2`.
It compares these two complete prototype implementations under the same
prepared activations. It does not isolate an instruction-only effect and must
not be described as an I8MM instruction gain. The receipt also reports max
absolute error, floor-stabilized maximum relative error, normalized maximum
absolute error, RMSE, and cosine similarity for both approximate methods
against MLX and against each other. Its `throughput` object is always `null`.

## Exact Highway extraction point

If the evidence supports a production experiment, keep these changes
independent:

1. Refactor the threaded M>1 scheduler in
   `mlx/backend/cpu/quantized_highway.cpp` so every row is quantized once before
   output-column work starts, then pass row-owned `QmmTInt8Prequantized` state
   into every chunk. Benchmark that reuse fix separately.
2. Only then evaluate a `qmm_t_int8_cols` two-row/two-column I8MM path. It must
   retain MLX affine scale and exact raw-sum bias correction, and it must use a
   Highway mixed U8*I8 2x2 primitive or separately benchmark packed/recentered
   signed weights. Do not compare it with a version that repeats preparation.
3. Integrate the ARM Highway target in CMake and add Highway runtime target
   dispatch/fallback coverage. A change confined to `quantized_highway.cpp`
   cannot safely introduce the Apple I8MM target into a portable MLX build.

The first item is a scheduling/data-lifetime correction. The second is the
instruction/tiling experiment. The third is required production plumbing. They
need separate correctness and performance evidence.
