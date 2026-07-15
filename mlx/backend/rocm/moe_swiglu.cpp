// Copyright © 2025 Apple Inc.
//
// Fused sorted-MoE SwiGLU as a compile-safe Primitive:
// one host D2H of expert ids + segment GEMMs for gate, up, silu_mul, and down.
// Replaces three separate gather_mm calls that each pipeline-drained the train step.
//
// VJP uses the gather_mm fallback (Custom base) so value_and_grad / compile work.
//
// NOTE: Primitive class lives in mlx::core (not rocm) to avoid Shape/Device name
// collisions with mlx::core::rocm::Shape (hip_array) and related aliases.

#include "mlx/backend/rocm/rocm.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/device.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>

namespace mlx::core {

namespace {

// Compile-safe primitive: eval_gpu only (no mid-transform eval).
class MoeSwigluSorted : public fast::Custom {
 public:
  explicit MoeSwigluSorted(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback)
      : Custom(stream, std::move(fallback)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MoeSwigluSorted has no CPU implementation");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  DEFINE_NAME(MoeSwigluSorted)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

void MoeSwigluSorted::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // inputs: x[T,D], w_gate[E,D,I], w_up[E,D,I], w_down[E,I,D], ids[T]
  // outputs[0]: y[T,D]
  assert(inputs.size() == 5);
  assert(outputs.size() == 1);

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  array& out = outputs[0];

  array x = inputs[0];
  array wg = inputs[1];
  array wu = inputs[2];
  array wd = inputs[3];
  array ids = inputs[4];

  const int T = x.shape(0);
  const int D = x.shape(1);
  const int E = wg.shape(0);
  const int I = wg.shape(2);

  if (T == 0) {
    out.set_data(rocm::malloc_async(0, encoder));
    return;
  }

  // Contiguous copies if needed (inputs already realized by the evaluator).
  if (!x.flags().row_contiguous) {
    x = contiguous_copy_gpu(x, s);
    encoder.add_temporary(x);
  }
  if (!ids.flags().row_contiguous) {
    ids = contiguous_copy_gpu(ids, s);
    encoder.add_temporary(ids);
  }
  if (!wg.flags().row_contiguous) {
    wg = contiguous_copy_gpu(wg, s);
    encoder.add_temporary(wg);
  }
  if (!wu.flags().row_contiguous) {
    wu = contiguous_copy_gpu(wu, s);
    encoder.add_temporary(wu);
  }
  if (!wd.flags().row_contiguous) {
    wd = contiguous_copy_gpu(wd, s);
    encoder.add_temporary(wd);
  }

  out.set_data(rocm::malloc_async(
      static_cast<size_t>(T) * D * size_of(bfloat16), encoder));

  Shape mid_shape{T, I};
  array gate(
      rocm::malloc_async(
          static_cast<size_t>(T) * I * size_of(bfloat16), encoder),
      mid_shape,
      bfloat16,
      allocator::free);
  array up(
      rocm::malloc_async(
          static_cast<size_t>(T) * I * size_of(bfloat16), encoder),
      mid_shape,
      bfloat16,
      allocator::free);
  array h(
      rocm::malloc_async(
          static_cast<size_t>(T) * I * size_of(bfloat16), encoder),
      mid_shape,
      bfloat16,
      allocator::free);
  encoder.add_temporary(gate);
  encoder.add_temporary(up);
  encoder.add_temporary(h);

  encoder.set_input_array(x);
  encoder.set_input_array(wg);
  encoder.set_input_array(wu);
  encoder.set_input_array(wd);
  encoder.set_input_array(ids);
  encoder.set_output_array(out);
  encoder.set_output_array(gate);
  encoder.set_output_array(up);
  encoder.set_output_array(h);

  // One D2H of expert ids — the only host sync for the whole SwiGLU.
  static thread_local uint32_t* pin = nullptr;
  static thread_local size_t pin_cap = 0;
  const size_t need = static_cast<size_t>(T);
  if (need > pin_cap) {
    if (pin)
      (void)hipHostFree(pin);
    pin_cap = need + need / 2 + 1024;
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&pin),
        pin_cap * sizeof(uint32_t),
        hipHostMallocDefault));
  }
  hipStream_t hs = static_cast<hipStream_t>(encoder.stream());
  CHECK_HIP_ERROR(hipMemcpyAsync(
      pin,
      gpu_ptr<const uint32_t>(ids),
      need * sizeof(uint32_t),
      hipMemcpyDeviceToHost,
      hs));
  CHECK_HIP_ERROR(hipStreamSynchronize(hs));

  const size_t esz = sizeof(uint16_t); // bf16
  const char* xB = static_cast<const char*>(gpu_ptr<void>(x));
  const char* wgB = static_cast<const char*>(gpu_ptr<void>(wg));
  const char* wuB = static_cast<const char*>(gpu_ptr<void>(wu));
  const char* wdB = static_cast<const char*>(gpu_ptr<void>(wd));
  char* gateB = static_cast<char*>(gpu_ptr<void>(gate));
  char* upB = static_cast<char*>(gpu_ptr<void>(up));
  char* hB = static_cast<char*>(gpu_ptr<void>(h));
  char* outB = static_cast<char*>(gpu_ptr<void>(out));

  // Weight layouts match lemonseed gather_mm after swapaxes:
  //   w_gate/w_up [E,D,I]: x[M,D] @ B[D,I] → [M,I]
  //   w_down [E,I,D]:      h[M,I] @ B[I,D] → [M,D]
  const int64_t wg_stride = static_cast<int64_t>(D) * I;
  const int64_t wu_stride = wg_stride;
  const int64_t wd_stride = static_cast<int64_t>(I) * D;

  // Exact-M: each expert uses its true token count (no pad).
  int start = 0;
  while (start < T) {
    uint32_t e = pin[static_cast<size_t>(start)];
    if (static_cast<int>(e) >= E) {
      throw std::runtime_error("MoeSwigluSorted: expert id out of range");
    }
    int end = start + 1;
    while (end < T && pin[static_cast<size_t>(end)] == e)
      ++end;
    int Mseg = end - start;
    if (Mseg <= 0) {
      start = end;
      continue;
    }

    rocm::hipblaslt_gemm_ptrs(
        encoder,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        Mseg,
        I,
        D,
        1.0f,
        xB + static_cast<size_t>(start) * D * esz,
        /*lda=*/D,
        wgB + static_cast<size_t>(e) * wg_stride * esz,
        /*ldb=*/I,
        0.0f,
        gateB + static_cast<size_t>(start) * I * esz,
        /*ldc=*/I,
        bfloat16);

    rocm::hipblaslt_gemm_ptrs(
        encoder,
        false,
        false,
        Mseg,
        I,
        D,
        1.0f,
        xB + static_cast<size_t>(start) * D * esz,
        D,
        wuB + static_cast<size_t>(e) * wu_stride * esz,
        I,
        0.0f,
        upB + static_cast<size_t>(start) * I * esz,
        I,
        bfloat16);

    start = end;
  }

  rocm::silu_mul_bf16(
      encoder,
      gpu_ptr<void>(gate),
      gpu_ptr<void>(up),
      gpu_ptr<void>(h),
      T * I);

  start = 0;
  while (start < T) {
    uint32_t e = pin[static_cast<size_t>(start)];
    int end = start + 1;
    while (end < T && pin[static_cast<size_t>(end)] == e)
      ++end;
    int Mseg = end - start;
    if (Mseg <= 0) {
      start = end;
      continue;
    }
    rocm::hipblaslt_gemm_ptrs(
        encoder,
        false,
        false,
        Mseg,
        D,
        I,
        1.0f,
        hB + static_cast<size_t>(start) * I * esz,
        /*lda=*/I,
        wdB + static_cast<size_t>(e) * wd_stride * esz,
        /*ldb=*/D,
        0.0f,
        outB + static_cast<size_t>(start) * D * esz,
        /*ldc=*/D,
        bfloat16);
    start = end;
  }
  // No final stream sync — evaluator owns completion / fences.
}

// Differentiable fallback for VJP / jvp / vmap via Custom base.
std::vector<array> moe_swiglu_fallback(
    const std::vector<array>& inputs,
    Stream s) {
  const array& x = inputs[0]; // [T, D]
  const array& wg = inputs[1]; // [E, D, I]
  const array& wu = inputs[2];
  const array& wd = inputs[3]; // [E, I, D]
  const array& ids = inputs[4]; // [T]

  const int T = x.shape(0);
  if (T == 0) {
    return {array(Shape{0, x.shape(1)}, x.dtype(), nullptr, {})};
  }

  auto xg = reshape(x, Shape{T, 1, 1, x.shape(1)}, s);
  // Indices are discrete routing — never differentiable (GatherMM rejects
  // index VJP). stop_gradient keeps value_and_grad on x/weights working.
  auto lhs = reshape(
      stop_gradient(arange(T, uint32, s), s), Shape{T, 1, 1}, s);
  auto rhs = reshape(stop_gradient(ids, s), Shape{T, 1, 1}, s);

  auto gate = gather_mm(xg, wg, lhs, rhs, /*sorted_indices=*/true, s);
  auto up = gather_mm(xg, wu, lhs, rhs, /*sorted_indices=*/true, s);
  // silu(g)*u = g * sigmoid(g) * u
  auto h = multiply(multiply(gate, sigmoid(gate, s), s), up, s);
  auto down = gather_mm(h, wd, lhs, rhs, /*sorted_indices=*/true, s);
  return {reshape(down, Shape{T, x.shape(1)}, s)};
}

// Fused bwd Primitive: zero-sync device exact-M (default) or host-RLE hipBLASLt.
// inputs: x, wg, wu, wd, ids, dy
// outputs: dx, dwg, dwu, dwd
class MoeSwigluSortedVJP : public Primitive {
 public:
  explicit MoeSwigluSortedVJP(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MoeSwigluSortedVJP has no CPU implementation");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  DEFINE_NAME(MoeSwigluSortedVJP)
  DEFINE_DEFAULT_IS_EQUIVALENT()

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    // dx, dwg, dwu, dwd
    return {inputs[0].shape(),
            inputs[1].shape(),
            inputs[2].shape(),
            inputs[3].shape()};
  }
};

void MoeSwigluSortedVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 6);
  assert(outputs.size() == 4);

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  array x = inputs[0];
  array wg = inputs[1];
  array wu = inputs[2];
  array wd = inputs[3];
  array ids = inputs[4];
  array dy = inputs[5];

  const int T = x.shape(0);
  const int D = x.shape(1);
  const int E = wg.shape(0);
  const int I = wg.shape(2);

  auto contig = [&](array a) {
    if (!a.flags().row_contiguous) {
      a = contiguous_copy_gpu(a, s);
      encoder.add_temporary(a);
    }
    return a;
  };
  x = contig(x);
  wg = contig(wg);
  wu = contig(wu);
  wd = contig(wd);
  ids = contig(ids);
  dy = contig(dy);

  array& dx = outputs[0];
  array& dwg = outputs[1];
  array& dwu = outputs[2];
  array& dwd = outputs[3];

  auto alloc = [&](array& a, size_t nbytes) {
    a.set_data(rocm::malloc_async(nbytes, encoder));
  };
  const size_t esz = sizeof(uint16_t);
  alloc(dx, static_cast<size_t>(T) * D * esz);
  alloc(dwg, static_cast<size_t>(E) * D * I * esz);
  alloc(dwu, static_cast<size_t>(E) * D * I * esz);
  alloc(dwd, static_cast<size_t>(E) * I * D * esz);

  if (T == 0) {
    // Zero weight grads for empty batch.
    encoder.launch_kernel([ptr = gpu_ptr<void>(dwg),
                           n = dwg.nbytes()](hipStream_t st) {
      (void)hipMemsetAsync(ptr, 0, n, st);
    });
    encoder.launch_kernel([ptr = gpu_ptr<void>(dwu),
                           n = dwu.nbytes()](hipStream_t st) {
      (void)hipMemsetAsync(ptr, 0, n, st);
    });
    encoder.launch_kernel([ptr = gpu_ptr<void>(dwd),
                           n = dwd.nbytes()](hipStream_t st) {
      (void)hipMemsetAsync(ptr, 0, n, st);
    });
    return;
  }

  for (const array& a : {x, wg, wu, wd, ids, dy}) {
    encoder.set_input_array(a);
  }
  for (array* o : {&dx, &dwg, &dwu, &dwd}) {
    encoder.set_output_array(*o);
  }

  // Default: zero-sync device-segmented exact-M (token-tiled expert GEMM +
  // device segments for dW). No D2H / StreamSynchronize — pipeline-friendly.
  // Kill-switch MLX_ROCM_MOE_VJP_HIPBLAS=1 → host-RLE + hipBLASLt (MFMA) path.
  static const bool use_hipblas_vjp = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_VJP_HIPBLAS");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();

  const int64_t wg_stride = static_cast<int64_t>(D) * I;
  const int64_t wd_stride = static_cast<int64_t>(I) * D;

  auto mk = [&](Shape sh) {
    array a(sh, bfloat16, nullptr, {});
    a.set_data(rocm::malloc_async(a.nbytes(), encoder));
    encoder.add_temporary(a);
    return a;
  };
  array gate = mk(Shape{T, I});
  array up = mk(Shape{T, I});
  array h = mk(Shape{T, I});
  array dh = mk(Shape{T, I});
  array dg = mk(Shape{T, I});
  array du = mk(Shape{T, I});

  if (!use_hipblas_vjp) {
    // ---- Zero-sync device exact-M path ----
    rocm::moe_sorted_expert_gemm(
        encoder, x, wg, ids, gate, T, /*N=*/I, /*K=*/D, E,
        /*b_transposed=*/false, /*ldb=*/I, wg_stride);
    rocm::moe_sorted_expert_gemm(
        encoder, x, wu, ids, up, T, I, D, E, false, I, wg_stride);
    rocm::moe_sorted_expert_gemm(
        encoder, dy, wd, ids, dh, T, /*N=*/I, /*K=*/D, E,
        /*b_transposed=*/true, /*ldb=*/D, wd_stride);

    rocm::swiglu_bwd_elem_bf16(
        encoder,
        gpu_ptr<void>(gate),
        gpu_ptr<void>(up),
        gpu_ptr<void>(dh),
        gpu_ptr<void>(h),
        gpu_ptr<void>(dg),
        gpu_ptr<void>(du),
        T * I);

    // dx = dg @ Wg.T + du @ Wu.T  (device kernel has no beta)
    {
      void* op = gpu_ptr<void>(dx);
      encoder.launch_kernel([op, n = dx.nbytes()](hipStream_t st) {
        (void)hipMemsetAsync(op, 0, n, st);
      });
    }
    rocm::moe_sorted_expert_gemm(
        encoder, dg, wg, ids, dx, T, /*N=*/D, /*K=*/I, E, true, I, wg_stride);
    array dx_u = mk(Shape{T, D});
    rocm::moe_sorted_expert_gemm(
        encoder, du, wu, ids, dx_u, T, D, I, E, true, I, wg_stride);
    rocm::bf16_add_inplace(
        encoder, gpu_ptr<void>(dx_u), gpu_ptr<void>(dx), T * D);

    array segments(Shape{E, 2}, uint32, nullptr, {});
    segments.set_data(rocm::malloc_async(segments.nbytes(), encoder));
    encoder.add_temporary(segments);
    rocm::moe_sorted_segments(encoder, ids, segments, T, E);

    rocm::segmented_mm_device(
        encoder, x, dg, segments, dwg, /*M=*/D, /*N=*/I, E,
        /*a_transposed=*/true, /*lda=*/D, /*a_k_stride=*/D,
        /*b_transposed=*/false, /*ldb=*/I, /*b_k_stride=*/I,
        /*out_stride=*/wg_stride);
    rocm::segmented_mm_device(
        encoder, x, du, segments, dwu, D, I, E, true, D, D, false, I, I,
        wg_stride);
    rocm::segmented_mm_device(
        encoder, h, dy, segments, dwd, /*M=*/I, /*N=*/D, E, true, /*lda=*/I,
        /*a_k_stride=*/I, false, /*ldb=*/D, /*b_k_stride=*/D,
        /*out_stride=*/wd_stride);
    return;
  }

  // ---- Host-RLE + hipBLASLt exact-M (opt-in MFMA path) ----
  static thread_local uint32_t* pin = nullptr;
  static thread_local size_t pin_cap = 0;
  const size_t need = static_cast<size_t>(T);
  if (need > pin_cap) {
    if (pin)
      (void)hipHostFree(pin);
    pin_cap = need + need / 2 + 1024;
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&pin),
        pin_cap * sizeof(uint32_t),
        hipHostMallocDefault));
  }
  hipStream_t hs = static_cast<hipStream_t>(encoder.stream());
  CHECK_HIP_ERROR(hipMemcpyAsync(
      pin,
      gpu_ptr<const uint32_t>(ids),
      need * sizeof(uint32_t),
      hipMemcpyDeviceToHost,
      hs));
  CHECK_HIP_ERROR(hipStreamSynchronize(hs));

  const char* xB = static_cast<const char*>(gpu_ptr<void>(x));
  const char* dyB = static_cast<const char*>(gpu_ptr<void>(dy));
  const char* wgB = static_cast<const char*>(gpu_ptr<void>(wg));
  const char* wuB = static_cast<const char*>(gpu_ptr<void>(wu));
  const char* wdB = static_cast<const char*>(gpu_ptr<void>(wd));
  char* dwgB = static_cast<char*>(gpu_ptr<void>(dwg));
  char* dwuB = static_cast<char*>(gpu_ptr<void>(dwu));
  char* dwdB = static_cast<char*>(gpu_ptr<void>(dwd));
  char* dxB = static_cast<char*>(gpu_ptr<void>(dx));
  char* gateB = static_cast<char*>(gpu_ptr<void>(gate));
  char* upB = static_cast<char*>(gpu_ptr<void>(up));
  char* hB = static_cast<char*>(gpu_ptr<void>(h));
  char* dhB = static_cast<char*>(gpu_ptr<void>(dh));
  char* dgB = static_cast<char*>(gpu_ptr<void>(dg));
  char* duB = static_cast<char*>(gpu_ptr<void>(du));

  encoder.launch_kernel([ptr = gpu_ptr<void>(dwg), n = dwg.nbytes()](hipStream_t st) {
    (void)hipMemsetAsync(ptr, 0, n, st);
  });
  encoder.launch_kernel([ptr = gpu_ptr<void>(dwu), n = dwu.nbytes()](hipStream_t st) {
    (void)hipMemsetAsync(ptr, 0, n, st);
  });
  encoder.launch_kernel([ptr = gpu_ptr<void>(dwd), n = dwd.nbytes()](hipStream_t st) {
    (void)hipMemsetAsync(ptr, 0, n, st);
  });

  auto gemm = [&](bool ta, bool tb, int M, int N, int K, const char* a, int lda,
                  const char* b, int ldb, float beta, char* c, int ldc) {
    if (rocm::is_hipblaslt_available()) {
      rocm::hipblaslt_gemm_ptrs(
          encoder, ta, tb, M, N, K, 1.0f, a, lda, b, ldb, beta, c, ldc,
          bfloat16);
    } else {
      rocm::rocblas_gemm_ptrs(
          encoder, ta, tb, M, N, K, 1.0f, a, lda, b, ldb, beta, c, ldc,
          bfloat16);
    }
  };

  int start = 0;
  while (start < T) {
    uint32_t e = pin[static_cast<size_t>(start)];
    if (static_cast<int>(e) >= E) {
      throw std::runtime_error("MoeSwigluSortedVJP: expert id out of range");
    }
    int end = start + 1;
    while (end < T && pin[static_cast<size_t>(end)] == e)
      ++end;
    int Mseg = end - start;
    if (Mseg <= 0) {
      start = end;
      continue;
    }
    const size_t xoff = static_cast<size_t>(start) * D * esz;
    const size_t ioff = static_cast<size_t>(start) * I * esz;
    const size_t wgoff = static_cast<size_t>(e) * wg_stride * esz;
    const size_t wdoff = static_cast<size_t>(e) * wd_stride * esz;

    gemm(false, false, Mseg, I, D, xB + xoff, D, wgB + wgoff, I, 0.0f,
         gateB + ioff, I);
    gemm(false, false, Mseg, I, D, xB + xoff, D, wuB + wgoff, I, 0.0f,
         upB + ioff, I);
    gemm(false, true, Mseg, I, D, dyB + xoff, D, wdB + wdoff, D, 0.0f,
         dhB + ioff, I);
    start = end;
  }

  rocm::swiglu_bwd_elem_bf16(
      encoder, gpu_ptr<void>(gate), gpu_ptr<void>(up), gpu_ptr<void>(dh),
      gpu_ptr<void>(h), gpu_ptr<void>(dg), gpu_ptr<void>(du), T * I);

  start = 0;
  while (start < T) {
    uint32_t e = pin[static_cast<size_t>(start)];
    int end = start + 1;
    while (end < T && pin[static_cast<size_t>(end)] == e)
      ++end;
    int Mseg = end - start;
    if (Mseg <= 0) {
      start = end;
      continue;
    }
    const size_t xoff = static_cast<size_t>(start) * D * esz;
    const size_t ioff = static_cast<size_t>(start) * I * esz;
    const size_t wgoff = static_cast<size_t>(e) * wg_stride * esz;
    const size_t wdoff = static_cast<size_t>(e) * wd_stride * esz;

    gemm(false, true, Mseg, D, I, dgB + ioff, I, wgB + wgoff, I, 0.0f,
         dxB + xoff, D);
    gemm(false, true, Mseg, D, I, duB + ioff, I, wuB + wgoff, I, 1.0f,
         dxB + xoff, D);

    gemm(true, false, D, I, Mseg, xB + xoff, D, dgB + ioff, I, 0.0f,
         dwgB + wgoff, I);
    gemm(true, false, D, I, Mseg, xB + xoff, D, duB + ioff, I, 0.0f,
         dwuB + wgoff, I);
    gemm(true, false, I, D, Mseg, hB + ioff, I, dyB + xoff, D, 0.0f,
         dwdB + wdoff, D);
    start = end;
  }
}

} // namespace

namespace rocm {

array moe_swiglu_sorted(
    const array& x_in,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids_in,
    StreamOrDevice s_in) {
  // x: [T, D]
  // w_gate/w_up: [E, D, I]
  // w_down: [E, I, D]
  // ids: [T] uint32 sorted by expert
  // out: [T, D]
  if (!is_available()) {
    throw std::runtime_error("moe_swiglu_sorted requires ROCm");
  }
  if (x_in.ndim() != 2 || expert_ids_in.ndim() != 1 || w_gate.ndim() != 3 ||
      w_up.ndim() != 3 || w_down.ndim() != 3) {
    throw std::invalid_argument(
        "moe_swiglu_sorted: expected x[T,D], w_gate/up[E,D,I], w_down[E,I,D], ids[T]");
  }
  if (x_in.dtype() != bfloat16 || w_gate.dtype() != bfloat16 ||
      w_up.dtype() != bfloat16 || w_down.dtype() != bfloat16) {
    throw std::invalid_argument("moe_swiglu_sorted: bf16 only");
  }
  if (expert_ids_in.dtype() != uint32) {
    throw std::invalid_argument("moe_swiglu_sorted: expert_ids must be uint32");
  }

  const int T = x_in.shape(0);
  const int D = x_in.shape(1);
  const int E = w_gate.shape(0);
  const int I = w_gate.shape(2);
  if (w_gate.shape(1) != D || w_up.shape(0) != E || w_up.shape(1) != D ||
      w_up.shape(2) != I || w_down.shape(0) != E || w_down.shape(1) != I ||
      w_down.shape(2) != D || expert_ids_in.shape(0) != T) {
    throw std::invalid_argument("moe_swiglu_sorted: shape mismatch");
  }

  // Force GPU stream (default monostate → gpu, not host default).
  // Fully-qualify Device/Shape: rocm:: namespace aliases collide with core.
  Stream s = to_stream(s_in, ::mlx::core::Device::gpu);
  if (s.device.type != ::mlx::core::Device::DeviceType::gpu) {
    throw std::runtime_error("moe_swiglu_sorted: GPU stream required");
  }

  if (T == 0) {
    return array(::mlx::core::Shape{0, D}, bfloat16, nullptr, {});
  }

  auto fallback = [s](const std::vector<array>& inputs) {
    return moe_swiglu_fallback(inputs, s);
  };

  // Lazy array with Primitive — safe under mx.compile / value_and_grad.
  return array(
      ::mlx::core::Shape{T, D},
      bfloat16,
      std::make_shared<MoeSwigluSorted>(s, std::move(fallback)),
      {x_in, w_gate, w_up, w_down, expert_ids_in});
}

std::vector<array> moe_swiglu_sorted_vjp(
    const array& x_in,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids_in,
    const array& dy_in,
    StreamOrDevice s_in) {
  if (!is_available()) {
    throw std::runtime_error("moe_swiglu_sorted_vjp requires ROCm");
  }
  if (x_in.ndim() != 2 || expert_ids_in.ndim() != 1 || dy_in.ndim() != 2 ||
      w_gate.ndim() != 3 || w_up.ndim() != 3 || w_down.ndim() != 3) {
    throw std::invalid_argument(
        "moe_swiglu_sorted_vjp: expected x/dy[T,D], w_gate/up[E,D,I], "
        "w_down[E,I,D], ids[T]");
  }
  if (x_in.dtype() != bfloat16 || w_gate.dtype() != bfloat16 ||
      w_up.dtype() != bfloat16 || w_down.dtype() != bfloat16 ||
      dy_in.dtype() != bfloat16) {
    throw std::invalid_argument("moe_swiglu_sorted_vjp: bf16 only");
  }
  if (expert_ids_in.dtype() != uint32) {
    throw std::invalid_argument(
        "moe_swiglu_sorted_vjp: expert_ids must be uint32");
  }

  const int T = x_in.shape(0);
  const int D = x_in.shape(1);
  const int E = w_gate.shape(0);
  const int I = w_gate.shape(2);
  if (w_gate.shape(1) != D || w_up.shape(0) != E || w_up.shape(1) != D ||
      w_up.shape(2) != I || w_down.shape(0) != E || w_down.shape(1) != I ||
      w_down.shape(2) != D || expert_ids_in.shape(0) != T ||
      dy_in.shape(0) != T || dy_in.shape(1) != D) {
    throw std::invalid_argument("moe_swiglu_sorted_vjp: shape mismatch");
  }

  Stream s = to_stream(s_in, ::mlx::core::Device::gpu);
  if (s.device.type != ::mlx::core::Device::DeviceType::gpu) {
    throw std::runtime_error("moe_swiglu_sorted_vjp: GPU stream required");
  }

  return array::make_arrays(
      {::mlx::core::Shape{T, D},
       ::mlx::core::Shape{E, D, I},
       ::mlx::core::Shape{E, D, I},
       ::mlx::core::Shape{E, I, D}},
      {bfloat16, bfloat16, bfloat16, bfloat16},
      std::make_shared<MoeSwigluSortedVJP>(s),
      {x_in, w_gate, w_up, w_down, expert_ids_in, dy_in});
}

} // namespace rocm

} // namespace mlx::core
