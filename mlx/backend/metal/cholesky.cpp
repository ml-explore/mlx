// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <cassert>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  if (inputs[0].dtype() != float32) {
    throw std::runtime_error(
        "[Cholesky::eval_gpu] Metal Cholesky only supports float32.");
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

  // Blocked right-looking factorization. Each strip of nb_outer columns is
  // factored 32 columns at a time (cholesky_potf2 + cholesky_trsm + a
  // strip-bounded cholesky_syrk), and the trailing update beyond the strip,
  // where nearly all the FLOPs are for a large matrix, runs on the steel
  // GEMM. A final fixup pass zeroes the strict upper triangle, or transposes
  // in place for the upper factor. The phases are separate dispatches
  // ordered by buffer barriers.
  constexpr int NB = 32;
  constexpr int nb_outer = 512;
  int upper = upper_ ? 1 : 0;
  std::string tname = type_to_name(out);
  auto potf2_kernel = get_cholesky_kernel(d, "cholesky_potf2_" + tname, out);
  auto trsm_kernel = get_cholesky_kernel(d, "cholesky_trsm_" + tname, out);
  auto syrk32_kernel = get_cholesky_kernel(d, "cholesky_syrk32_" + tname, out);
  auto syrk64_kernel = get_cholesky_kernel(d, "cholesky_syrk64_" + tname, out);
  auto fixup_kernel = get_cholesky_kernel(d, "cholesky_fixup_" + tname, out);

  // Scratch for the inverted diagonal blocks.
  array linv({static_cast<int>(num_matrices), NB, NB}, float32, nullptr, {});
  linv.set_data(allocator::malloc(linv.nbytes()));
  compute_encoder.add_temporary(linv);

  // Factor the 32 columns at p: POTF2 (+ TRTRI) then the TRSM below.
  auto factor32 = [&](int p) {
    compute_encoder.set_compute_pipeline_state(potf2_kernel);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_bytes(N, 1);
    compute_encoder.set_bytes(p, 2);
    compute_encoder.set_output_array(linv, 3);
    compute_encoder.dispatch_threadgroups(
        MTL::Size(num_matrices, 1, 1), MTL::Size(256, 1, 1));
    compute_encoder.barrier();
    int m = N - p - NB;
    if (m > 0) {
      compute_encoder.set_compute_pipeline_state(trsm_kernel);
      compute_encoder.set_output_array(out, 0);
      compute_encoder.set_bytes(N, 1);
      compute_encoder.set_bytes(p, 2);
      compute_encoder.set_input_array(linv, 3);
      int nrb = (m + 127) / 128;
      compute_encoder.dispatch_threadgroups(
          MTL::Size(nrb, num_matrices, 1), MTL::Size(256, 1, 1));
      compute_encoder.barrier();
    }
  };
  // Rank-kd update bounded to the next ncols columns of the strip.
  auto strip_syrk =
      [&](MTL::ComputePipelineState* kernel, int p, int kd, int ncols) {
        int ht = N - p - kd;
        int cb = std::min(ht, ncols);
        if (ht <= 0 || cb <= 0) {
          return;
        }
        int ntj = (cb + NB - 1) / NB;
        int nti = (ht + NB - 1) / NB;
        compute_encoder.set_compute_pipeline_state(kernel);
        compute_encoder.set_output_array(out, 0);
        compute_encoder.set_bytes(N, 1);
        compute_encoder.set_bytes(p, 2);
        compute_encoder.set_bytes(cb, 3);
        compute_encoder.dispatch_threadgroups(
            MTL::Size(size_t(nti) * ntj, num_matrices, 1),
            MTL::Size(128, 1, 1));
        compute_encoder.barrier();
      };

  auto strided_flags = out.flags();
  strided_flags.row_contiguous = false;
  strided_flags.col_contiguous = false;
  strided_flags.contiguous = false;

  for (int po = 0; po < N; po += nb_outer) {
    int W = std::min(nb_outer, N - po);
    int strip_end = po + W;

    // Factor the strip with paired 32-column panels: prime the second
    // panel's columns with a rank-32 update, factor it, then apply a single
    // rank-64 update to the rest of the strip.
    int pi = po;
    while (pi < strip_end) {
      factor32(pi);
      int rem = strip_end - pi - NB;
      if (rem <= 0) {
        pi += NB;
      } else if (rem <= NB) {
        strip_syrk(syrk32_kernel, pi, NB, rem);
        factor32(pi + NB);
        pi += 2 * NB;
      } else {
        strip_syrk(syrk32_kernel, pi, NB, NB);
        factor32(pi + NB);
        strip_syrk(syrk64_kernel, pi, 2 * NB, rem - NB);
        pi += 2 * NB;
      }
    }

    // Trailing update beyond the strip on the steel GEMM: C -= L21 L21ᵀ, in
    // place on strided submatrix views.
    int m = N - strip_end;
    if (m > 0) {
      Strides sub_strides{int64_t(N) * N, int64_t(N), 1};
      array l21({static_cast<int>(num_matrices), m, W}, float32, nullptr, {});
      l21.copy_shared_buffer(
          out,
          sub_strides,
          strided_flags,
          out.data_size(),
          int64_t(strip_end) * N + po);
      array csub({static_cast<int>(num_matrices), m, m}, float32, nullptr, {});
      csub.copy_shared_buffer(
          out,
          sub_strides,
          strided_flags,
          out.data_size(),
          int64_t(strip_end) * N + strip_end);
      std::vector<array> copies;
      steel_matmul_regular_axpby<true>(
          /* s = */ s,
          /* d = */ d,
          /* a = */ l21,
          /* b = */ l21,
          /* c = */ csub,
          /* out = */ csub,
          /* M = */ m,
          /* N = */ m,
          /* K = */ W,
          /* batch_size_out = */ static_cast<int>(num_matrices),
          /* lda = */ N,
          /* ldb = */ N,
          /* ldd = */ N,
          /* transpose_a = */ false,
          /* transpose_b = */ true,
          /* copies = */ copies,
          /* batch_shape = */ {1},
          /* batch_strides = */ {0},
          /* A_batch_stride = */ int64_t(N) * N,
          /* B_batch_stride = */ int64_t(N) * N,
          /* matrix_stride_out = */ int64_t(N) * N,
          /* C_batch_stride = */ int64_t(N) * N,
          /* alpha = */ -1.0f,
          /* beta = */ 1.0f);
      compute_encoder.barrier();
    }
  }

  compute_encoder.set_compute_pipeline_state(fixup_kernel);
  compute_encoder.set_output_array(out, 0);
  compute_encoder.set_bytes(N, 1);
  compute_encoder.set_bytes(upper, 2);
  compute_encoder.dispatch_threads(
      MTL::Size(N, num_matrices, 1), MTL::Size(std::min(N, 256), 1, 1));
}

} // namespace mlx::core
