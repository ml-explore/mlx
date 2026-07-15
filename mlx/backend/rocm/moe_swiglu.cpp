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

// Fused bwd Primitive: one D2H + recompute + all grads (exact-M GEMMs).
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

  // Zero weight grads once (experts not hit stay 0).
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

  // Device pack + tiny counts D2H (E ints only) for exact-M GEMMs.
  // Avoids full-ids sync AND pad-to-T FLOP waste (was ~E× slower than pure).
  int M_fixed = (T + 31) & ~31;
  if (M_fixed < 32)
    M_fixed = 32;

  const int64_t wg_stride = static_cast<int64_t>(D) * I;
  const int64_t wd_stride = static_cast<int64_t>(I) * D;
  const int64_t stride_x = static_cast<int64_t>(M_fixed) * D;
  const int64_t stride_i = static_cast<int64_t>(M_fixed) * I;

  auto mk = [&](Shape sh, Dtype dt = bfloat16) {
    array a(sh, dt, nullptr, {});
    a.set_data(rocm::malloc_async(a.nbytes(), encoder));
    encoder.add_temporary(a);
    return a;
  };
  array px = mk(Shape{E, M_fixed, D});
  array pdy = mk(Shape{E, M_fixed, D});
  array pg = mk(Shape{E, M_fixed, I});
  array pu = mk(Shape{E, M_fixed, I});
  array ph = mk(Shape{E, M_fixed, I});
  array pdh = mk(Shape{E, M_fixed, I});
  array pdg = mk(Shape{E, M_fixed, I});
  array pdu = mk(Shape{E, M_fixed, I});
  array pdx = mk(Shape{E, M_fixed, D});
  array slot = mk(Shape{E, M_fixed}, int32);
  array cnt = mk(Shape{E}, int32);

  // Joint pack x|dy so slots match.
  array xdy = mk(Shape{T, 2 * D});
  rocm::bf16_concat_rows(
      encoder, gpu_ptr<void>(x), gpu_ptr<void>(dy), gpu_ptr<void>(xdy), T, D);
  array pxdy = mk(Shape{E, M_fixed, 2 * D});
  rocm::moe_pack_tokens(
      encoder, xdy, ids, pxdy, slot, cnt, T, 2 * D, E, M_fixed);
  rocm::bf16_split_rows(
      encoder,
      gpu_ptr<void>(pxdy),
      gpu_ptr<void>(px),
      gpu_ptr<void>(pdy),
      E * M_fixed,
      D);

  // Zero-sync path: batched/padded GEMMs (pad rows are zero → correct dW).
  // Prefer this over tiny counts D2H: host sync per layer kills pipeline peaks.
  if (rocm::is_hipblaslt_available()) {
    rocm::hipblaslt_gemm_batched(
        encoder, false, false, M_fixed, I, D, 1.0f, px, D, stride_x, wg, I,
        wg_stride, 0.0f, pg, I, stride_i, E, bfloat16);
    rocm::hipblaslt_gemm_batched(
        encoder, false, false, M_fixed, I, D, 1.0f, px, D, stride_x, wu, I,
        wg_stride, 0.0f, pu, I, stride_i, E, bfloat16);
  } else {
    throw std::runtime_error("MoeSwigluSortedVJP: hipBLASLt required");
  }

  // TN via rocBLAS (E expert launches, no host wait)
  {
    char* pdyB = static_cast<char*>(gpu_ptr<void>(pdy));
    char* pdhB = static_cast<char*>(gpu_ptr<void>(pdh));
    char* wdB = static_cast<char*>(gpu_ptr<void>(wd));
    for (int e = 0; e < E; ++e) {
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, I, D, 1.0f,
          pdyB + e * stride_x * esz, D, wdB + e * wd_stride * esz, D, 0.0f,
          pdhB + e * stride_i * esz, I, bfloat16);
    }
  }

  rocm::swiglu_bwd_elem_bf16(
      encoder, gpu_ptr<void>(pg), gpu_ptr<void>(pu), gpu_ptr<void>(pdh),
      gpu_ptr<void>(ph), gpu_ptr<void>(pdg), gpu_ptr<void>(pdu),
      E * M_fixed * I);

  {
    char* pdgB = static_cast<char*>(gpu_ptr<void>(pdg));
    char* pduB = static_cast<char*>(gpu_ptr<void>(pdu));
    char* pdxB = static_cast<char*>(gpu_ptr<void>(pdx));
    char* pxB = static_cast<char*>(gpu_ptr<void>(px));
    char* phB = static_cast<char*>(gpu_ptr<void>(ph));
    char* pdyB = static_cast<char*>(gpu_ptr<void>(pdy));
    char* wgB = static_cast<char*>(gpu_ptr<void>(wg));
    char* wuB = static_cast<char*>(gpu_ptr<void>(wu));
    char* dwgB = static_cast<char*>(gpu_ptr<void>(dwg));
    char* dwuB = static_cast<char*>(gpu_ptr<void>(dwu));
    char* dwdB = static_cast<char*>(gpu_ptr<void>(dwd));
    for (int e = 0; e < E; ++e) {
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, D, I, 1.0f,
          pdgB + e * stride_i * esz, I, wgB + e * wg_stride * esz, I, 0.0f,
          pdxB + e * stride_x * esz, D, bfloat16);
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, D, I, 1.0f,
          pduB + e * stride_i * esz, I, wuB + e * wg_stride * esz, I, 1.0f,
          pdxB + e * stride_x * esz, D, bfloat16);
      rocm::rocblas_gemm_ptrs(
          encoder, true, false, D, I, M_fixed, 1.0f,
          pxB + e * stride_x * esz, D, pdgB + e * stride_i * esz, I, 0.0f,
          dwgB + e * wg_stride * esz, I, bfloat16);
      rocm::rocblas_gemm_ptrs(
          encoder, true, false, D, I, M_fixed, 1.0f,
          pxB + e * stride_x * esz, D, pduB + e * stride_i * esz, I, 0.0f,
          dwuB + e * wg_stride * esz, I, bfloat16);
      rocm::rocblas_gemm_ptrs(
          encoder, true, false, I, D, M_fixed, 1.0f,
          phB + e * stride_i * esz, I, pdyB + e * stride_x * esz, D, 0.0f,
          dwdB + e * wd_stride * esz, D, bfloat16);
    }
  }

  {
    void* op = gpu_ptr<void>(dx);
    encoder.launch_kernel([op, n = dx.nbytes()](hipStream_t st) {
      (void)hipMemsetAsync(op, 0, n, st);
    });
  }
  rocm::moe_unpack_tokens(encoder, pdx, slot, dx, E, M_fixed, D);
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
