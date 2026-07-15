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

  Shape mid{T, I};
  array gate(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  array up(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  array h(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  array dh(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  array dg(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  array du(
      rocm::malloc_async(static_cast<size_t>(T) * I * esz, encoder),
      mid,
      bfloat16,
      allocator::free);
  for (array* t : {&gate, &up, &h, &dh, &dg, &du}) {
    encoder.add_temporary(*t);
  }

  for (const array& a : {x, wg, wu, wd, ids, dy}) {
    encoder.set_input_array(a);
  }
  for (array* o : {&dx, &dwg, &dwu, &dwd, &gate, &up, &h, &dh, &dg, &du}) {
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

  // One D2H of expert ids.
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
  const char* wgB = static_cast<const char*>(gpu_ptr<void>(wg));
  const char* wuB = static_cast<const char*>(gpu_ptr<void>(wu));
  const char* wdB = static_cast<const char*>(gpu_ptr<void>(wd));
  const char* dyB = static_cast<const char*>(gpu_ptr<void>(dy));
  char* gateB = static_cast<char*>(gpu_ptr<void>(gate));
  char* upB = static_cast<char*>(gpu_ptr<void>(up));
  char* hB = static_cast<char*>(gpu_ptr<void>(h));
  char* dhB = static_cast<char*>(gpu_ptr<void>(dh));
  char* dgB = static_cast<char*>(gpu_ptr<void>(dg));
  char* duB = static_cast<char*>(gpu_ptr<void>(du));
  char* dxB = static_cast<char*>(gpu_ptr<void>(dx));
  char* dwgB = static_cast<char*>(gpu_ptr<void>(dwg));
  char* dwuB = static_cast<char*>(gpu_ptr<void>(dwu));
  char* dwdB = static_cast<char*>(gpu_ptr<void>(dwd));

  const int64_t wg_stride = static_cast<int64_t>(D) * I;
  const int64_t wd_stride = static_cast<int64_t>(I) * D;

  // Collect segments from pin (already sorted).
  struct Seg {
    int e, start, M;
  };
  std::vector<Seg> segs;
  segs.reserve(static_cast<size_t>(E));
  int max_m = 1;
  for (int i = 0; i < T;) {
    uint32_t e = pin[static_cast<size_t>(i)];
    if (static_cast<int>(e) >= E) {
      throw std::runtime_error("MoeSwigluSortedVJP: expert id out of range");
    }
    int j = i + 1;
    while (j < T && pin[static_cast<size_t>(j)] == e)
      ++j;
    int Mseg = j - i;
    if (Mseg > 0) {
      segs.push_back({static_cast<int>(e), i, Mseg});
      max_m = std::max(max_m, Mseg);
    }
    i = j;
  }
  // Tile-align for hipBLASLt batched MFMA (same as forward pack path).
  int M_fixed = (max_m + 31) & ~31;
  if (M_fixed < 32)
    M_fixed = 32;

  // Packed buffers [E, M_fixed, *] — zero pad keeps unused slots out of dW.
  auto mk = [&](Shape sh) {
    array a(sh, bfloat16, nullptr, {});
    a.set_data(rocm::malloc_async(a.nbytes(), encoder));
    encoder.add_temporary(a);
    encoder.launch_kernel(
        [p = gpu_ptr<void>(a), n = a.nbytes()](hipStream_t st) {
          (void)hipMemsetAsync(p, 0, n, st);
        });
    return a;
  };
  array px = mk(Shape{E, M_fixed, D}); // packed x
  array pdy = mk(Shape{E, M_fixed, D}); // packed dy
  array pg = mk(Shape{E, M_fixed, I});
  array pu = mk(Shape{E, M_fixed, I});
  array ph = mk(Shape{E, M_fixed, I});
  array pdh = mk(Shape{E, M_fixed, I});
  array pdg = mk(Shape{E, M_fixed, I});
  array pdu = mk(Shape{E, M_fixed, I});
  array pdx = mk(Shape{E, M_fixed, D});

  // Host-RLE pack: memcpy each expert's exact run into its pad slot.
  for (const auto& sseg : segs) {
    const size_t src = static_cast<size_t>(sseg.start) * D * esz;
    const size_t dst =
        (static_cast<size_t>(sseg.e) * M_fixed) * D * esz;
    const size_t nbytes = static_cast<size_t>(sseg.M) * D * esz;
    char* pxB = static_cast<char*>(gpu_ptr<void>(px));
    char* pdyB = static_cast<char*>(gpu_ptr<void>(pdy));
    encoder.launch_kernel([=](hipStream_t st) {
      (void)hipMemcpyAsync(pxB + dst, xB + src, nbytes, hipMemcpyDeviceToDevice, st);
      (void)hipMemcpyAsync(
          pdyB + dst, dyB + src, nbytes, hipMemcpyDeviceToDevice, st);
    });
  }

  const int64_t stride_x = static_cast<int64_t>(M_fixed) * D;
  const int64_t stride_i = static_cast<int64_t>(M_fixed) * I;
  const int64_t stride_wg = wg_stride;
  const int64_t stride_wd = wd_stride;

  // Prefer batched hipBLASLt for NN; rocBLAS ptrs for TN/NT (train-safe).
  auto gemm_batched_nn = [&](array& a,
                             array& b,
                             array& c,
                             int M,
                             int N,
                             int K,
                             int64_t sa,
                             int lda,
                             int64_t sb,
                             int ldb,
                             int64_t sc,
                             int ldc,
                             bool tb) {
    if (!tb && rocm::is_hipblaslt_available()) {
      rocm::hipblaslt_gemm_batched(
          encoder, false, false, M, N, K, 1.0f, a, lda, sa, b, ldb, sb, 0.0f,
          c, ldc, sc, E, bfloat16);
    } else {
      // Fall back: E expert GEMMs (rocBLAS if tb).
      char* aB = static_cast<char*>(gpu_ptr<void>(a));
      char* bB = static_cast<char*>(gpu_ptr<void>(b));
      char* cB = static_cast<char*>(gpu_ptr<void>(c));
      for (int e = 0; e < E; ++e) {
        if (tb) {
          rocm::rocblas_gemm_ptrs(
              encoder, false, true, M, N, K, 1.0f,
              aB + e * sa * esz, lda, bB + e * sb * esz, ldb, 0.0f,
              cB + e * sc * esz, ldc, bfloat16);
        } else {
          rocm::hipblaslt_gemm_ptrs(
              encoder, false, false, M, N, K, 1.0f,
              aB + e * sa * esz, lda, bB + e * sb * esz, ldb, 0.0f,
              cB + e * sc * esz, ldc, bfloat16);
        }
      }
    }
  };

  // gate = x @ Wg  [E,M,D]@[E,D,I]
  gemm_batched_nn(
      px, wg, pg, M_fixed, I, D, stride_x, D, stride_wg, I, stride_i, I,
      /*tb=*/false);
  gemm_batched_nn(
      px, wu, pu, M_fixed, I, D, stride_x, D, stride_wg, I, stride_i, I,
      false);

  // dh = dy @ Wd.T  Wd [E,I,D]
  {
    char* pdyB = static_cast<char*>(gpu_ptr<void>(pdy));
    char* pdhB = static_cast<char*>(gpu_ptr<void>(pdh));
    char* wdBp = static_cast<char*>(gpu_ptr<void>(wd));
    for (int e = 0; e < E; ++e) {
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, I, D, 1.0f,
          pdyB + e * stride_x * esz, D, wdBp + e * stride_wd * esz, D, 0.0f,
          pdhB + e * stride_i * esz, I, bfloat16);
    }
  }

  rocm::swiglu_bwd_elem_bf16(
      encoder,
      gpu_ptr<void>(pg),
      gpu_ptr<void>(pu),
      gpu_ptr<void>(pdh),
      gpu_ptr<void>(ph),
      gpu_ptr<void>(pdg),
      gpu_ptr<void>(pdu),
      E * M_fixed * I);

  // dx = dg @ Wg.T + du @ Wu.T
  {
    char* pdgB = static_cast<char*>(gpu_ptr<void>(pdg));
    char* pduB = static_cast<char*>(gpu_ptr<void>(pdu));
    char* pdxB = static_cast<char*>(gpu_ptr<void>(pdx));
    char* wgBp = static_cast<char*>(gpu_ptr<void>(wg));
    char* wuBp = static_cast<char*>(gpu_ptr<void>(wu));
    for (int e = 0; e < E; ++e) {
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, D, I, 1.0f,
          pdgB + e * stride_i * esz, I, wgBp + e * stride_wg * esz, I, 0.0f,
          pdxB + e * stride_x * esz, D, bfloat16);
      rocm::rocblas_gemm_ptrs(
          encoder, false, true, M_fixed, D, I, 1.0f,
          pduB + e * stride_i * esz, I, wuBp + e * stride_wg * esz, I, 1.0f,
          pdxB + e * stride_x * esz, D, bfloat16);
    }
  }

  // dW: zero already; per-expert NT with exact M from segs (no pad pollution)
  for (const auto& sseg : segs) {
    const size_t xoff = static_cast<size_t>(sseg.start) * D * esz;
    const size_t doff = static_cast<size_t>(sseg.start) * D * esz;
    // dg/du/h live in packed buffers at expert slot
    const size_t poff =
        static_cast<size_t>(sseg.e) * static_cast<size_t>(M_fixed) * I * esz;
    const size_t wgoff = static_cast<size_t>(sseg.e) * wg_stride * esz;
    const size_t wdoff = static_cast<size_t>(sseg.e) * wd_stride * esz;
    char* pdgB = static_cast<char*>(gpu_ptr<void>(pdg));
    char* pduB = static_cast<char*>(gpu_ptr<void>(pdu));
    char* phB = static_cast<char*>(gpu_ptr<void>(ph));
    // dWg = x.T @ dg  — use exact Mseg from original x/dy (not pad)
    rocm::rocblas_gemm_ptrs(
        encoder, true, false, D, I, sseg.M, 1.0f, xB + xoff, D, pdgB + poff, I,
        0.0f, dwgB + wgoff, I, bfloat16);
    rocm::rocblas_gemm_ptrs(
        encoder, true, false, D, I, sseg.M, 1.0f, xB + xoff, D, pduB + poff, I,
        0.0f, dwuB + wgoff, I, bfloat16);
    rocm::rocblas_gemm_ptrs(
        encoder, true, false, I, D, sseg.M, 1.0f, phB + poff, I, dyB + doff, D,
        0.0f, dwdB + wdoff, D, bfloat16);
  }

  // Unpack pdx → dx using RLE (exact rows)
  char* pdxB = static_cast<char*>(gpu_ptr<void>(pdx));
  for (const auto& sseg : segs) {
    const size_t src =
        (static_cast<size_t>(sseg.e) * M_fixed) * D * esz;
    const size_t dst = static_cast<size_t>(sseg.start) * D * esz;
    const size_t nbytes = static_cast<size_t>(sseg.M) * D * esz;
    encoder.launch_kernel([=](hipStream_t st) {
      (void)hipMemcpyAsync(
          dxB + dst, pdxB + src, nbytes, hipMemcpyDeviceToDevice, st);
    });
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
