// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <cassert>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

bool Cholesky::use_fallback(Dtype dtype, int n, size_t num_matrices, Stream s) {
  if (s.device == Device::cpu || dtype != float32) {
    return true;
  }
  // Minimum batch size at which the Metal kernels beat the CPU, measured on
  // an M3 Pro at both ends of each size range. Below the threshold, and for
  // sizes where the GPU was not measured to win, the CPU implementation is
  // used.
  size_t min_batch;
  if (n <= 12) {
    return true;
  } else if (n <= 31) {
    min_batch = 1000;
  } else if (n <= 64) {
    min_batch = 500;
  } else if (n <= 128) {
    min_batch = 250;
  } else if (n <= 192) {
    min_batch = 128;
  } else if (n <= 256) {
    min_batch = 100;
  } else if (n <= 512) {
    min_batch = 80;
  } else if (n <= 1024) {
    min_batch = 48;
  } else if (n <= 2048) {
    min_batch = 16;
  } else {
    return true;
  }
  return num_matrices < min_batch;
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  if (inputs[0].dtype() != float32) {
    throw std::runtime_error(
        "[Cholesky::eval_gpu] Metal Cholesky only supports float32. "
        "Use stream=Device::cpu for other types.");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  // The factorization runs in place, so copy the input into the output.
  const array& in = inputs[0];
  copy_gpu(
      in,
      out,
      in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      s);

  int N = out.shape(-1);
  size_t num_matrices = (N == 0) ? 0 : out.size() / (size_t(N) * N);
  if (num_matrices == 0 || N == 0) {
    return;
  }

  int upper = upper_ ? 1 : 0;
  size_t tg_bytes = size_t(N) * N * sizeof(float);
  size_t max_tg_mem = d.mtl_device()->maxThreadgroupMemoryLength();
  constexpr int kSimdSize = 32;

  if (N > 128) {
    // Batched blocked right-looking factorization (see kernels/cholesky.h):
    // per 32-column panel, one batched panel dispatch (POTF2 + TRSM) and one
    // batched simdgroup-matrix SYRK dispatch over the trailing lower-triangle
    // tiles, separated by buffer barriers, then one fixup pass to zero the
    // strict upper triangle (or transpose in place when upper). Separate
    // grids keep the SYRK at high occupancy across the whole batch.
    constexpr int NB = 32;
    std::string tname = type_to_name(out);
    auto panel_kernel = get_cholesky_kernel(d, "cholesky_panel_" + tname, out);
    auto syrk32_kernel =
        get_cholesky_kernel(d, "cholesky_syrk32_" + tname, out);
    auto syrk64_kernel =
        get_cholesky_kernel(d, "cholesky_syrk64_" + tname, out);
    auto fixup_kernel = get_cholesky_kernel(d, "cholesky_fixup_" + tname, out);
    size_t panel_tg = std::min<size_t>(
        256, panel_kernel->maxTotalThreadsPerThreadgroup() / 32 * 32);

    auto dispatch_panel = [&](int p) {
      compute_encoder.set_compute_pipeline_state(panel_kernel);
      compute_encoder.set_output_array(out, 0);
      compute_encoder.set_bytes(N, 1);
      compute_encoder.set_bytes(p, 2);
      compute_encoder.dispatch_threadgroups(
          MTL::Size(num_matrices, 1, 1), MTL::Size(panel_tg, 1, 1));
      compute_encoder.barrier();
    };
    auto dispatch_syrk =
        [&](MTL::ComputePipelineState* kernel, int p, int kd, bool col0) {
          int ht = N - p - kd;
          if (ht <= 0) {
            return;
          }
          int nt = (ht + NB - 1) / NB;
          int num_tiles = col0 ? nt : nt * (nt + 1) / 2;
          int c0 = col0 ? 1 : 0;
          compute_encoder.set_compute_pipeline_state(kernel);
          compute_encoder.set_output_array(out, 0);
          compute_encoder.set_bytes(N, 1);
          compute_encoder.set_bytes(p, 2);
          compute_encoder.set_bytes(c0, 3);
          compute_encoder.dispatch_threadgroups(
              MTL::Size(num_tiles, num_matrices, 1), MTL::Size(128, 1, 1));
          compute_encoder.barrier();
        };

    // Panels are processed in pairs: prime the second panel's block-column
    // with a rank-32 update, factor it, then apply a single rank-64 update
    // to the rest of the trailing matrix. The panels stay 32 columns wide
    // while the trailing read-modify-write traffic halves.
    int p = 0;
    while (p < N) {
      dispatch_panel(p);
      if (N - p > NB) {
        dispatch_syrk(syrk32_kernel, p, NB, true);
        dispatch_panel(p + NB);
        dispatch_syrk(syrk64_kernel, p, 2 * NB, false);
        p += 2 * NB;
      } else {
        p += NB;
      }
    }
    compute_encoder.set_compute_pipeline_state(fixup_kernel);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_bytes(N, 1);
    compute_encoder.set_bytes(upper, 2);
    compute_encoder.dispatch_threads(
        MTL::Size(N, num_matrices, 1), MTL::Size(std::min(N, 256), 1, 1));
    return;
  }

  // N <= 32: one matrix per simdgroup, no barriers. Up to 64 the whole matrix
  // is staged in threadgroup memory. Otherwise (up to 128) factor directly in
  // device memory, one matrix per threadgroup.
  std::string kname;
  if (N <= kSimdSize) {
    kname = "cholesky_simd_";
  } else if (N <= 2 * kSimdSize && tg_bytes <= max_tg_mem) {
    kname = "cholesky_shared_";
  } else {
    kname = "cholesky_device_";
  }
  kname += type_to_name(out);
  auto kernel = get_cholesky_kernel(d, kname, out);
  size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_output_array(out, 0);
  compute_encoder.set_bytes(N, 1);
  compute_encoder.set_bytes(upper, 2);

  if (N <= kSimdSize) {
    int simds_per_tg = 4;
    int nmat = static_cast<int>(num_matrices);
    compute_encoder.set_bytes(nmat, 3);
    size_t num_blocks =
        (num_matrices + simds_per_tg - 1) / size_t(simds_per_tg);
    MTL::Size grid_dims = MTL::Size(num_blocks, 1, 1);
    MTL::Size group_dims = MTL::Size(size_t(simds_per_tg) * kSimdSize, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    if (tg_bytes <= max_tg_mem) {
      compute_encoder.set_threadgroup_memory_length(tg_bytes, 0);
    }
    size_t tgsize = std::min<size_t>(N, max_threads);
    MTL::Size grid_dims = MTL::Size(tgsize * num_matrices, 1, 1);
    MTL::Size group_dims = MTL::Size(tgsize, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
