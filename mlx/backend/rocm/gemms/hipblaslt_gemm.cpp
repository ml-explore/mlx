// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace mlx::core::rocm {

namespace {

// Maximum workspace size for hipBLASLt algorithms (32 MB).
// hipBLASLt may request scratch memory for certain algorithm choices.
constexpr size_t kMaxWorkspaceBytes = 32u * 1024u * 1024u;

// Per-device hipBLASLt handle cache. Lazily initialised, thread-safe.
struct HipblasltState {
  hipblasLtHandle_t handle{nullptr};
  bool initialized{false};
  bool available{false};
  std::mutex mutex;

  // Persistent workspace allocation (grown as needed, never shrunk).
  void* workspace{nullptr};
  size_t workspace_size{0};
};

// One state per device (indexed by HIP device ordinal).
// 16 devices should be more than enough for any system.
static constexpr int kMaxDevices = 16;
static HipblasltState g_state[kMaxDevices];

HipblasltState& get_state(int device_id) {
  if (device_id < 0 || device_id >= kMaxDevices) {
    throw std::runtime_error(
        "hipBLASLt: device id out of range: " + std::to_string(device_id));
  }
  return g_state[device_id];
}

// Initialise the hipBLASLt handle for the given device.
// Must be called with state.mutex held.
void init_handle(HipblasltState& state, int device_id) {
  if (state.initialized) {
    return;
  }
  state.initialized = true;

  hipblasStatus_t status = hipblasLtCreate(&state.handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    state.available = false;
    state.handle = nullptr;
    std::cerr << "Warning: hipBLASLt initialization failed (status "
              << static_cast<int>(status) << ")." << std::endl;
    return;
  }
  state.available = true;

  // Pre-allocate the matmul workspace to the maximum size NOW so that
  // ensure_workspace() never calls hipMalloc during a HIP-graph capture (a
  // device alloc on the capturing stream invalidates the graph). Any algorithm
  // the heuristic returns fits within kMaxWorkspaceBytes, so a single up-front
  // allocation makes hipblasLtMatmul capture-safe.
  int prev_dev = 0;
  (void)hipGetDevice(&prev_dev);
  (void)hipSetDevice(device_id);
  if (hipMalloc(&state.workspace, kMaxWorkspaceBytes) == hipSuccess) {
    state.workspace_size = kMaxWorkspaceBytes;
  } else {
    state.workspace = nullptr;
    state.workspace_size = 0;
  }
  (void)hipSetDevice(prev_dev);
}

hipblasLtHandle_t get_handle(int device_id) {
  auto& state = get_state(device_id);
  if (!state.initialized) {
    std::lock_guard<std::mutex> lock(state.mutex);
    init_handle(state, device_id);
  }
  if (!state.available) {
    throw std::runtime_error("hipBLASLt is not available on this device.");
  }
  return state.handle;
}

// Ensure the per-device workspace is at least `required` bytes.
// Returns the workspace pointer and the actual allocated size.
// Must be called from within a launch_kernel callback (i.e., on the
// stream-submission thread for this device), so no extra locking is needed
// beyond the device serialisation that CommandEncoder already provides.
std::pair<void*, size_t> ensure_workspace(int device_id, size_t required) {
  auto& state = get_state(device_id);
  if (required <= state.workspace_size && state.workspace != nullptr) {
    return {state.workspace, state.workspace_size};
  }
  // Free old allocation (hipFree is a no-op on nullptr).
  if (state.workspace) {
    (void)hipFree(state.workspace);
    state.workspace = nullptr;
    state.workspace_size = 0;
  }
  if (required == 0) {
    return {nullptr, 0};
  }
  hipError_t err = hipMalloc(&state.workspace, required);
  if (err != hipSuccess) {
    state.workspace = nullptr;
    state.workspace_size = 0;
    return {nullptr, 0};
  }
  state.workspace_size = required;
  return {state.workspace, state.workspace_size};
}

hipDataType to_hipblaslt_dtype(Dtype dtype) {
  switch (dtype) {
    case float32:
      return HIP_R_32F;
    case float16:
      return HIP_R_16F;
    case bfloat16:
      return HIP_R_16BF;
    default:
      throw std::runtime_error("Unsupported dtype for hipBLASLt GEMM");
  }
}

hipblasOperation_t to_hipblas_op(bool transpose) {
  return transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}

// Per-device GEMM capability table, discovered at load time by asking
// hipBLASLt's heuristic which input types yield kernels on this GPU. This is a
// runtime probe rather than a hardcoded arch list, so it tracks whatever the
// installed Tensile library actually supports.
struct GemmCaps {
  bool probed{false};
  bool bf16{false};
  bool fp8_e4m3{false};
  bool fp8_e5m2{false};
  bool int8{false};
};
static GemmCaps g_caps[kMaxDevices];
static std::mutex g_caps_mutex;

// Does this (input, output, compute) combination have any hipBLASLt algorithm
// on the given handle? AlgoGetHeuristic only inspects descriptors, so no device
// memory is touched. Uses a representative GEMM shape.
bool probe_gemm_combo(
    hipblasLtHandle_t handle,
    hipDataType in_type,
    hipDataType out_type,
    hipblasComputeType_t compute_type) {
  hipblasLtMatmulDesc_t desc = nullptr;
  if (hipblasLtMatmulDescCreate(&desc, compute_type, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  int32_t op_t = HIPBLAS_OP_T, op_n = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(
      desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
  hipblasLtMatmulDescSetAttribute(
      desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
  const int M = 2048, N = 512, K = 2048;
  hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
  hipblasLtMatrixLayoutCreate(&la, in_type, K, M, K);
  hipblasLtMatrixLayoutCreate(&lb, in_type, K, N, K);
  hipblasLtMatrixLayoutCreate(&lc, out_type, M, N, M);
  hipblasLtMatrixLayoutCreate(&ld, out_type, M, N, M);
  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasLtMatmulPreferenceCreate(&pref);
  uint64_t ws = kMaxWorkspaceBytes;
  hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  hipblasLtMatmulHeuristicResult_t res[4];
  int count = 0;
  hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
      handle, desc, la, lb, lc, ld, pref, 4, res, &count);
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (ld)
    hipblasLtMatrixLayoutDestroy(ld);
  if (lc)
    hipblasLtMatrixLayoutDestroy(lc);
  if (lb)
    hipblasLtMatrixLayoutDestroy(lb);
  if (la)
    hipblasLtMatrixLayoutDestroy(la);
  if (desc)
    hipblasLtMatmulDescDestroy(desc);
  return st == HIPBLAS_STATUS_SUCCESS && count > 0;
}

const GemmCaps& gemm_caps(int device_id) {
  std::lock_guard<std::mutex> lock(g_caps_mutex);
  GemmCaps& caps = g_caps[device_id];
  if (caps.probed) {
    return caps;
  }
  caps.probed = true;
  hipblasLtHandle_t handle = nullptr;
  try {
    handle = get_handle(device_id);
  } catch (...) {
    return caps;
  }
  caps.bf16 = probe_gemm_combo(handle, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
  caps.fp8_e4m3 =
      probe_gemm_combo(handle, HIP_R_8F_E4M3, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
  caps.fp8_e5m2 =
      probe_gemm_combo(handle, HIP_R_8F_E5M2, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
  caps.int8 = probe_gemm_combo(handle, HIP_R_8I, HIP_R_32I, HIPBLAS_COMPUTE_32I);

  hipDeviceProp_t props;
  const char* arch =
      (hipGetDeviceProperties(&props, device_id) == hipSuccess)
      ? props.gcnArchName
      : "?";
  fprintf(
      stderr,
      "[hipBLASLt caps] device %d (%s): bf16=%d fp8_e4m3=%d fp8_e5m2=%d int8=%d\n",
      device_id,
      arch,
      caps.bf16,
      caps.fp8_e4m3,
      caps.fp8_e5m2,
      caps.int8);
  return caps;
}

// Input precision chosen for a GEMM on a given device. The hardware/library
// capability table decides which is reachable; accuracy ranks them e4m3 > bf16
// for our (already-quantized) weights.
enum class GemmPrecision { Bf16, Fp8E4M3, Fp8E5M2, Int8 };

// Highest-throughput input precision this device can run for half-precision
// GEMMs while preserving accuracy: fp8 e4m3 where the library has kernels
// (RDNA4), otherwise bf16 (RDNA3.5 and anything without fp8 Tensile kernels).
GemmPrecision preferred_gemm_precision(int device_id) {
  const GemmCaps& caps = gemm_caps(device_id);
  if (caps.fp8_e4m3) {
    return GemmPrecision::Fp8E4M3;
  }
  return GemmPrecision::Bf16;
}

// RAII wrappers for hipBLASLt descriptors to avoid leaks on error paths.
struct MatmulDescGuard {
  hipblasLtMatmulDesc_t desc{nullptr};
  ~MatmulDescGuard() {
    if (desc)
      hipblasLtMatmulDescDestroy(desc);
  }
};
struct MatrixLayoutGuard {
  hipblasLtMatrixLayout_t layout{nullptr};
  ~MatrixLayoutGuard() {
    if (layout)
      hipblasLtMatrixLayoutDestroy(layout);
  }
};
struct PreferenceGuard {
  hipblasLtMatmulPreference_t pref{nullptr};
  ~PreferenceGuard() {
    if (pref)
      hipblasLtMatmulPreferenceDestroy(pref);
  }
};

// Core implementation: set up descriptors, find the best algorithm, and
// execute the matmul on the given stream.
void hipblaslt_gemm_impl(
    hipblasLtHandle_t handle,
    int device_id,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int M,
    int N,
    int K,
    const float* alpha,
    const void* a_ptr,
    int lda,
    int64_t stride_a,
    const void* b_ptr,
    int ldb,
    int64_t stride_b,
    const float* beta,
    void* c_ptr,
    int ldc,
    int64_t stride_c,
    int batch_count,
    hipDataType data_type,
    hipStream_t stream) {
  hipblasStatus_t status;

  // Discover this device's GEMM capability table on first use (prints once).
  GemmPrecision precision = preferred_gemm_precision(device_id);
  (void)precision;

  hipDataType scale_type = HIP_R_32F;
  int32_t trans_a_val = static_cast<int32_t>(op_a);
  int32_t trans_b_val = static_cast<int32_t>(op_b);

  // --- Matrix layouts (column-major, as expected by BLAS) ---
  // A is (op_a == N) ? M x K : K x M  in column-major
  // B is (op_b == N) ? K x N : N x K  in column-major
  // C is M x N in column-major
  uint64_t a_rows = (op_a == HIPBLAS_OP_N) ? M : K;
  uint64_t a_cols = (op_a == HIPBLAS_OP_N) ? K : M;
  uint64_t b_rows = (op_b == HIPBLAS_OP_N) ? K : N;
  uint64_t b_cols = (op_b == HIPBLAS_OP_N) ? N : K;

  MatrixLayoutGuard layout_a, layout_b, layout_c, layout_d;

  status = hipblasLtMatrixLayoutCreate(
      &layout_a.layout, data_type, a_rows, a_cols, lda);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(A) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  status = hipblasLtMatrixLayoutCreate(
      &layout_b.layout, data_type, b_rows, b_cols, ldb);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(B) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  status = hipblasLtMatrixLayoutCreate(&layout_c.layout, data_type, M, N, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(C) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  // D has the same layout as C (in-place: D == C).
  status = hipblasLtMatrixLayoutCreate(&layout_d.layout, data_type, M, N, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(D) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  // Set batch attributes when doing strided batched GEMM.
  if (batch_count > 1) {
    int32_t bc = batch_count;
    hipblasLtMatrixLayoutSetAttribute(
        layout_a.layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_a.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_a,
        sizeof(stride_a));

    hipblasLtMatrixLayoutSetAttribute(
        layout_b.layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_b.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_b,
        sizeof(stride_b));

    hipblasLtMatrixLayoutSetAttribute(
        layout_c.layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_c.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_c,
        sizeof(stride_c));

    hipblasLtMatrixLayoutSetAttribute(
        layout_d.layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_d.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_c,
        sizeof(stride_c));
  }

  // --- Algorithm selection via heuristic ---
  PreferenceGuard pref_guard;
  status = hipblasLtMatmulPreferenceCreate(&pref_guard.pref);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmulPreferenceCreate failed: " +
        std::to_string(static_cast<int>(status)));
  }

  uint64_t max_ws = kMaxWorkspaceBytes;
  hipblasLtMatmulPreferenceSetAttribute(
      pref_guard.pref,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_ws,
      sizeof(max_ws));

  // Request multiple algorithms for better occupancy/performance
  static constexpr int kMaxAlgos = 8;
  hipblasLtMatmulHeuristicResult_t heuristics[kMaxAlgos];
  int returned_algo_count = 0;

  MatmulDescGuard matmul_guard;
  status = hipblasLtMatmulDescCreate(
      &matmul_guard.desc, HIPBLAS_COMPUTE_32F, scale_type);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmulDescCreate failed: " +
        std::to_string(static_cast<int>(status)));
  }
  hipblasLtMatmulDescSetAttribute(
      matmul_guard.desc,
      HIPBLASLT_MATMUL_DESC_TRANSA,
      &trans_a_val,
      sizeof(trans_a_val));
  hipblasLtMatmulDescSetAttribute(
      matmul_guard.desc,
      HIPBLASLT_MATMUL_DESC_TRANSB,
      &trans_b_val,
      sizeof(trans_b_val));

  // Per-(shape,dtype,transpose,device) algorithm cache. The chosen heuristic
  // result is reusable across calls with identical problem geometry, so warm
  // calls skip AlgoGetHeuristic — the dominant per-call cost for the many small
  // GEMMs in a forward pass.
  struct AlgoKey {
    int M, N, K, batch, dt, ta, tb, dev;
    bool operator==(const AlgoKey& o) const {
      return M == o.M && N == o.N && K == o.K && batch == o.batch &&
          dt == o.dt && ta == o.ta && tb == o.tb && dev == o.dev;
    }
  };
  struct AlgoKeyHash {
    size_t operator()(const AlgoKey& k) const {
      size_t h = 1469598103934665603ULL;
      for (int v : {k.M, k.N, k.K, k.batch, k.dt, k.ta, k.tb, k.dev}) {
        h = (h ^ static_cast<size_t>(v)) * 1099511628211ULL;
      }
      return h;
    }
  };
  static std::mutex algo_mutex;
  static std::unordered_map<AlgoKey, hipblasLtMatmulHeuristicResult_t, AlgoKeyHash>
      algo_cache;

  AlgoKey key{
      M,
      N,
      K,
      batch_count,
      static_cast<int>(data_type),
      trans_a_val,
      trans_b_val,
      device_id};
  hipblasLtMatmulHeuristicResult_t heuristic;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(algo_mutex);
    auto cached = algo_cache.find(key);
    if (cached != algo_cache.end()) {
      heuristic = cached->second;
      cache_hit = true;
    }
  }

  if (!cache_hit) {
    status = hipblasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_guard.desc,
        layout_a.layout,
        layout_b.layout,
        layout_c.layout,
        layout_d.layout,
        pref_guard.pref,
        kMaxAlgos,
        heuristics,
        &returned_algo_count);

    if (status != HIPBLAS_STATUS_SUCCESS || returned_algo_count == 0) {
      throw std::runtime_error(
          "hipblasLtMatmulAlgoGetHeuristic failed (status=" +
          std::to_string(static_cast<int>(status)) +
          ", returned=" + std::to_string(returned_algo_count) + ")");
    }

    int best_algo_idx = 0;

    // Auto-tuning: benchmark all algorithms to find the fastest for each shape.
    // Disabled by default — for quantized models the GEMM path is rarely used
    // and the tuning overhead causes warm prompt regression.
    // Enable with MLX_ROCM_HIPBLASLT_TUNE=1 for non-quantized models.
    static bool do_tune = std::getenv("MLX_ROCM_HIPBLASLT_TUNE") != nullptr;

    if (do_tune && returned_algo_count > 1) {
    double best_time = 1e30;
    for (int algo_idx = 0; algo_idx < returned_algo_count; algo_idx++) {
      size_t ws_need = heuristics[algo_idx].workspaceSize;
      void* ws_p = nullptr;
      size_t ws_s = 0;
      if (ws_need > 0) {
        auto [p, s] = ensure_workspace(device_id, ws_need);
        ws_p = p;
        ws_s = s;
        if (!ws_p)
          continue;
      }

      // Warm-up
      (void)hipblasLtMatmul(
          handle,
          matmul_guard.desc,
          alpha,
          a_ptr,
          layout_a.layout,
          b_ptr,
          layout_b.layout,
          beta,
          c_ptr,
          layout_c.layout,
          c_ptr,
          layout_d.layout,
          &heuristics[algo_idx].algo,
          ws_p,
          ws_s,
          stream);
      (void)hipStreamSynchronize(stream);

      // Timed run
      hipEvent_t start_ev, stop_ev;
      (void)hipEventCreate(&start_ev);
      (void)hipEventCreate(&stop_ev);
      (void)hipEventRecord(start_ev, stream);

      static constexpr int kBenchIters = 3;
      for (int r = 0; r < kBenchIters; r++) {
        (void)hipblasLtMatmul(
            handle,
            matmul_guard.desc,
            alpha,
            a_ptr,
            layout_a.layout,
            b_ptr,
            layout_b.layout,
            beta,
            c_ptr,
            layout_c.layout,
            c_ptr,
            layout_d.layout,
            &heuristics[algo_idx].algo,
            ws_p,
            ws_s,
            stream);
      }

      (void)hipEventRecord(stop_ev, stream);
      (void)hipStreamSynchronize(stream);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, start_ev, stop_ev);
      (void)hipEventDestroy(start_ev);
      (void)hipEventDestroy(stop_ev);

      double avg = ms / kBenchIters;
      if (avg < best_time) {
        best_time = avg;
        best_algo_idx = algo_idx;
      }
    }
    }

    heuristic = heuristics[best_algo_idx];
    {
      std::lock_guard<std::mutex> lock(algo_mutex);
      algo_cache[key] = heuristic;
    }
  }

  // --- Workspace allocation ---
  size_t ws_needed = heuristic.workspaceSize;
  void* ws_ptr = nullptr;
  size_t ws_actual = 0;
  if (ws_needed > 0) {
    auto [p, s] = ensure_workspace(device_id, ws_needed);
    ws_ptr = p;
    ws_actual = s;
    if (ws_ptr == nullptr && ws_needed > 0) {
      throw std::runtime_error(
          "hipBLASLt: failed to allocate workspace of " +
          std::to_string(ws_needed) + " bytes");
    }
  }

  // --- Execute the matmul ---
  status = hipblasLtMatmul(
      handle,
      matmul_guard.desc,
      alpha,
      a_ptr,
      layout_a.layout,
      b_ptr,
      layout_b.layout,
      beta,
      c_ptr,
      layout_c.layout,
      c_ptr, // D == C (in-place)
      layout_d.layout,
      &heuristic.algo,
      ws_ptr,
      ws_actual,
      stream);

  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmul failed: " + std::to_string(static_cast<int>(status)));
  }
}

} // namespace

bool is_hipblaslt_available() {
  // Diagnostic: force the rocBLAS path everywhere to test whether rocBLAS bf16
  // GEMM is numerically correct for this model.
  static const bool g_force_rocblas = std::getenv("MLX_NO_HIPBLASLT") != nullptr;
  if (g_force_rocblas)
    return false;
  // When automatic HIP-graph batching is on, the GEMM is graph-split and run
  // immediately, but hipBLASLt's lazy hipblasLtCreate / AlgoGetHeuristic /
  // workspace hipMalloc are non-capturable and abort the process if the stream
  // is mid-graph. rocBLAS is graph-safe here, so force it whenever graphs are
  // enabled. (rocBLAS == hipBLASLt speed at decode, so this costs nothing.)
  if (use_hip_graphs())
    return false;
  // hipBLASLt's lazy init is non-capturable; force rocBLAS during any capture.
  if (stream_capturing())
    return false;
  int device_id = 0;
  (void)hipGetDevice(&device_id);
  auto& state = get_state(device_id);
  if (!state.initialized) {
    std::lock_guard<std::mutex> lock(state.mutex);
    init_handle(state, device_id);
  }
  return state.available;
}

void hipblaslt_gemm(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype) {
  int device_id = encoder.device().hip_device();
  hipblasLtHandle_t handle = get_handle(device_id);
  hipDataType hip_dtype = to_hipblaslt_dtype(dtype);

  // hipBLASLt uses column-major layout. MLX stores row-major, so we swap A
  // and B and compute C^T = B^T * A^T, just like the rocBLAS path.
  hipblasOperation_t op_a = to_hipblas_op(transpose_b);
  hipblasOperation_t op_b = to_hipblas_op(transpose_a);

  // Per-call GEMM tracing, gated behind an env flag.
  static const bool kGemmDebug = std::getenv("MLX_ROCM_GEMM_DEBUG") != nullptr;
  if (kGemmDebug) {
    fprintf(
        stderr,
        "[hipBLASLt] M=%d N=%d K=%d ta=%d tb=%d lda=%d ldb=%d ldc=%d\n",
        M, N, K, (int)transpose_a, (int)transpose_b, lda, ldb, ldc);
  }

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel([=, &encoder](hipStream_t stream) {
    hipblaslt_gemm_impl(
        handle,
        device_id,
        op_a,
        op_b,
        N, // swap M/N for col-major trick
        M,
        K,
        &alpha,
        b_ptr, // swap A/B
        ldb,
        0, // stride_a (unused for non-batched)
        a_ptr,
        lda,
        0, // stride_b (unused for non-batched)
        &beta,
        c_ptr,
        ldc,
        0, // stride_c (unused for non-batched)
        1, // batch_count
        hip_dtype,
        stream);
  });
}

void hipblaslt_gemm_batched(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype) {
  int device_id = encoder.device().hip_device();
  hipblasLtHandle_t handle = get_handle(device_id);
  hipDataType hip_dtype = to_hipblaslt_dtype(dtype);

  // Same column-major swap as above.
  hipblasOperation_t op_a = to_hipblas_op(transpose_b);
  hipblasOperation_t op_b = to_hipblas_op(transpose_a);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel([=, &encoder](hipStream_t stream) {
    hipblaslt_gemm_impl(
        handle,
        device_id,
        op_a,
        op_b,
        N,
        M,
        K,
        &alpha,
        b_ptr,
        ldb,
        stride_b, // swapped: was b, now is "A" in col-major
        a_ptr,
        lda,
        stride_a, // swapped: was a, now is "B" in col-major
        &beta,
        c_ptr,
        ldc,
        stride_c,
        batch_count,
        hip_dtype,
        stream);
  });
}

void hipblaslt_gemm_raw(
    hipStream_t stream,
    int op_a,
    int op_b,
    int M,
    int N,
    int K,
    const float* alpha,
    const void* a_ptr,
    int lda,
    const void* b_ptr,
    int ldb,
    const float* beta,
    void* c_ptr,
    int ldc,
    int data_type_hint,
    int /*compute_type_hint*/) {
  int device_id = 0;
  (void)hipGetDevice(&device_id);
  hipblasLtHandle_t handle = get_handle(device_id);

  // Map data_type_hint: 1=fp16, 2=bf16, 3=fp32
  hipDataType hip_dtype;
  switch (data_type_hint) {
    case 1:
      hip_dtype = HIP_R_16F;
      break;
    case 2:
      hip_dtype = HIP_R_16BF;
      break;
    default:
      hip_dtype = HIP_R_32F;
      break;
  }

  hipblaslt_gemm_impl(
      handle,
      device_id,
      static_cast<hipblasOperation_t>(op_a),
      static_cast<hipblasOperation_t>(op_b),
      M,
      N,
      K,
      alpha,
      a_ptr,
      lda,
      0,
      b_ptr,
      ldb,
      0,
      beta,
      c_ptr,
      ldc,
      0,
      1, // batch_count
      hip_dtype,
      stream);
}

bool device_has_fp8_gemm(int device_id) {
  return gemm_caps(device_id).fp8_e4m3;
}

void hipblaslt_gemm_fp8_raw(
    hipStream_t stream,
    int op_a,
    int op_b,
    int M,
    int N,
    int K,
    const void* a_ptr,
    int lda,
    const void* b_ptr,
    int ldb,
    void* c_ptr,
    int ldc,
    const float* a_scale,
    const float* b_scale) {
  int device_id = 0;
  (void)hipGetDevice(&device_id);
  hipblasLtHandle_t handle = get_handle(device_id);

  MatmulDescGuard desc_guard;
  if (hipblasLtMatmulDescCreate(
          &desc_guard.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("fp8 GEMM: descriptor create failed");
  }
  int32_t ta = op_a, tb = op_b;
  hipblasLtMatmulDescSetAttribute(
      desc_guard.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  hipblasLtMatmulDescSetAttribute(
      desc_guard.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  hipblasLtMatmulDescSetAttribute(
      desc_guard.desc,
      HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &a_scale,
      sizeof(a_scale));
  hipblasLtMatmulDescSetAttribute(
      desc_guard.desc,
      HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &b_scale,
      sizeof(b_scale));

  hipblasOperation_t oa = static_cast<hipblasOperation_t>(op_a);
  hipblasOperation_t ob = static_cast<hipblasOperation_t>(op_b);
  uint64_t a_rows = (oa == HIPBLAS_OP_N) ? M : K;
  uint64_t a_cols = (oa == HIPBLAS_OP_N) ? K : M;
  uint64_t b_rows = (ob == HIPBLAS_OP_N) ? K : N;
  uint64_t b_cols = (ob == HIPBLAS_OP_N) ? N : K;
  MatrixLayoutGuard la, lb, lc, ld;
  hipblasLtMatrixLayoutCreate(&la.layout, HIP_R_8F_E4M3, a_rows, a_cols, lda);
  hipblasLtMatrixLayoutCreate(&lb.layout, HIP_R_8F_E4M3, b_rows, b_cols, ldb);
  hipblasLtMatrixLayoutCreate(&lc.layout, HIP_R_16BF, M, N, ldc);
  hipblasLtMatrixLayoutCreate(&ld.layout, HIP_R_16BF, M, N, ldc);

  PreferenceGuard pref_guard;
  hipblasLtMatmulPreferenceCreate(&pref_guard.pref);
  uint64_t max_ws = kMaxWorkspaceBytes;
  hipblasLtMatmulPreferenceSetAttribute(
      pref_guard.pref,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_ws,
      sizeof(max_ws));

  // Best algorithm per (shape, device), tuned once. hipBLASLt's heuristic
  // top-pick is poor for fp8; timing all candidates on the first call and
  // caching the winner is worth the one-time cost (shapes repeat every layer).
  struct Key {
    int M, N, K, dev;
    bool operator==(const Key& o) const {
      return M == o.M && N == o.N && K == o.K && dev == o.dev;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      size_t h = 1469598103934665603ULL;
      for (int v : {k.M, k.N, k.K, k.dev}) {
        h = (h ^ static_cast<size_t>(v)) * 1099511628211ULL;
      }
      return h;
    }
  };
  static std::mutex mtx;
  static std::unordered_map<Key, hipblasLtMatmulHeuristicResult_t, KeyHash>
      algo_cache;

  Key key{M, N, K, device_id};
  hipblasLtMatmulHeuristicResult_t chosen;
  bool hit = false;
  {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = algo_cache.find(key);
    if (it != algo_cache.end()) {
      chosen = it->second;
      hit = true;
    }
  }

  float alpha = 1.0f, beta = 0.0f;
  if (!hit) {
    static constexpr int kNA = 16;
    hipblasLtMatmulHeuristicResult_t res[kNA];
    int cnt = 0;
    if (hipblasLtMatmulAlgoGetHeuristic(
            handle,
            desc_guard.desc,
            la.layout,
            lb.layout,
            lc.layout,
            ld.layout,
            pref_guard.pref,
            kNA,
            res,
            &cnt) != HIPBLAS_STATUS_SUCCESS ||
        cnt == 0) {
      throw std::runtime_error("fp8 GEMM: no algorithm for shape");
    }
    // Timing candidate algos requires synchronizing the stream, which is illegal
    // while it is in HIP-graph capture (the sync invalidates the capture and
    // every following matmul fails). Under capture, take the heuristic's top pick
    // — it records into the captured graph and runs on replay. Benchmark only
    // when not capturing.
    hipStreamCaptureStatus cap_st = hipStreamCaptureStatusNone;
    (void)hipStreamGetCaptureInfo(stream, &cap_st, nullptr);
    if (cap_st != hipStreamCaptureStatusNone) {
      chosen = res[0];
    } else {
    double best = 1e30;
    int best_idx = 0;
    for (int a = 0; a < cnt; ++a) {
      size_t need = res[a].workspaceSize;
      void* wp = nullptr;
      size_t ws = 0;
      if (need > 0) {
        auto [p, s] = ensure_workspace(device_id, need);
        wp = p;
        ws = s;
        if (!wp)
          continue;
      }
      hipblasStatus_t feas = hipblasLtMatmul(
              handle,
              desc_guard.desc,
              &alpha,
              a_ptr,
              la.layout,
              b_ptr,
              lb.layout,
              &beta,
              c_ptr,
              lc.layout,
              c_ptr,
              ld.layout,
              &res[a].algo,
              wp,
              ws,
              stream);
      if (feas != HIPBLAS_STATUS_SUCCESS) {
        continue;
      }
      (void)hipStreamSynchronize(stream);
      hipEvent_t e0, e1;
      (void)hipEventCreate(&e0);
      (void)hipEventCreate(&e1);
      (void)hipEventRecord(e0, stream);
      for (int r = 0; r < 3; ++r) {
        (void)hipblasLtMatmul(
            handle,
            desc_guard.desc,
            &alpha,
            a_ptr,
            la.layout,
            b_ptr,
            lb.layout,
            &beta,
            c_ptr,
            lc.layout,
            c_ptr,
            ld.layout,
            &res[a].algo,
            wp,
            ws,
            stream);
      }
      (void)hipEventRecord(e1, stream);
      (void)hipEventSynchronize(e1);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, e0, e1);
      (void)hipEventDestroy(e0);
      (void)hipEventDestroy(e1);
      if (ms < best) {
        best = ms;
        best_idx = a;
      }
    }
    chosen = res[best_idx];
    }
    std::lock_guard<std::mutex> lock(mtx);
    algo_cache[key] = chosen;
  }

  size_t need = chosen.workspaceSize;
  void* wp = nullptr;
  size_t ws = 0;
  if (need > 0) {
    auto [p, s] = ensure_workspace(device_id, need);
    wp = p;
    ws = s;
  }
  hipblasStatus_t fst = hipblasLtMatmul(
          handle,
          desc_guard.desc,
          &alpha,
          a_ptr,
          la.layout,
          b_ptr,
          lb.layout,
          &beta,
          c_ptr,
          lc.layout,
          c_ptr,
          ld.layout,
          &chosen.algo,
          wp,
          ws,
          stream);
  if (fst != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "fp8 GEMM: hipblasLtMatmul failed (status " + std::to_string((int)fst) +
        ")");
  }
}

} // namespace mlx::core::rocm
