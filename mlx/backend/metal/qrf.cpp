// Copyright © 2023-2024 Apple Inc.

#include <iostream>

#include "mlx/array.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/qrf.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {
size_t colmajor_idx(size_t row, size_t col, size_t nrows) {
  return col * nrows + row;
}

// Flip to true to run the scalar cpu code for debugging.
static constexpr bool RUN_ON_CPU_FOR_DEBUG = false;

} // namespace

void qrf_block(
    const Stream& stream,
    array& a,
    array& q,
    int m,
    int n,
    int R,
    int startc,
    array& Y,
    array& W,
    array& betas,
    array& Yp,
    array& WY,
    array& QWY,
    array& Wp) {
  float* a_data = a.data<float>();
  float* q_data = q.data<float>();
  float* w_data = W.data<float>();
  float* y_data = Y.data<float>();
  float* betas_data = betas.data<float>();
  float* yp_data = Yp.data<float>();
  float* wy_data = WY.data<float>();
  float* qwy_data = QWY.data<float>();
  float* wp_data = Wp.data<float>();

  auto& device = metal::device(stream.device);

  auto& compute_encoder = device.get_command_encoder(stream.index);

  for (int jj = 0; jj < R; jj++) {
    const int col = startc + jj;

    if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
      auto launch_kernel = [&](const std::string& kernel_name) {
        auto kernel = device.get_kernel(kernel_name);

        compute_encoder->setComputePipelineState(kernel);
        compute_encoder.set_output_array(a, 0);
        compute_encoder.set_output_array(Y, 1);
        compute_encoder.set_output_array(Yp, 2);
        compute_encoder.set_output_array(betas, 3);
        compute_encoder->setBytes(&jj, sizeof(jj), 4);
        compute_encoder->setBytes(&startc, sizeof(startc), 5);
        compute_encoder->setBytes(&m, sizeof(m), 6);

        const MTL::Size threads_per_grid(m, 1, 1);
        const MTL::Size threads_per_threadgroup(
            std::min(threads_per_grid.width, kernel->threadExecutionWidth()),
            1,
            1);
        compute_encoder->dispatchThreads(
            threads_per_grid, threads_per_threadgroup);
      };

      launch_kernel("qrf_reset");
      launch_kernel("qrf_compute_sq_norm_xc");
      launch_kernel("qrf_compute_v");
      launch_kernel("qrf_compute_beta");
    } else {
      // Accumulate norm(x) where x is the part of the current column on and
      // below the diagonal.
      float sq_norm_xc = 0;

      // NOTE: only the elements on and below the diagonal.
      for (int i = col; i < m; i++) {
        sq_norm_xc += std::pow(a_data[colmajor_idx(i, col, m)], 2);
      }

      float norm_xc = std::pow(sq_norm_xc, 0.5);

      // Build Householder vector v (stored in the jj-th column of Y).
      float* v = y_data + colmajor_idx(0, jj, m);
      float sq_norm_v = 0;
      for (int i = 0; i < m; i++) {
        if (i < col) {
          v[i] = 0;
        } else if (i == col) {
          const float old_val = a_data[colmajor_idx(i, i, m)];
          const float new_val =
              old_val >= 0 ? old_val + norm_xc : old_val - norm_xc;
          v[i] = new_val;
        } else {
          v[i] = a_data[colmajor_idx(i, col, m)];
        }
        sq_norm_v += std::pow(v[i], 2);
      }

      const float beta = 2.0f / sq_norm_v;
      betas_data[jj] = beta;
    }

    // Apply the reflection to the rest of the current block.
    // First, compute:
    //    c = v.T @ a[:, col:startc+r] (of shape startc+r-col, 1)
    if (!RUN_ON_CPU_FOR_DEBUG) {
      auto a_restblock_shape = a.shape();
      a_restblock_shape.back() = startc + R - col;
      assert(a_restblock_shape.back() > 0);
      array a_restblock(a_restblock_shape, a.dtype(), nullptr, {});
      const size_t a_rest_offset = m * col;
      a_restblock.copy_shared_buffer(
          a, a.strides(), a.flags(), a_restblock.size(), a_rest_offset);

      std::vector<int> v_shape{m, 1};
      array v_array(v_shape, a.dtype(), nullptr, {});
      const size_t v_offset = m * jj;
      v_array.copy_shared_buffer(
          Y, Y.strides(), Y.flags(), v_array.size(), v_offset);

      std::vector<array> copies;

      // We store data in colmajor order, but the matmul kernel assumes
      // row-major. To produce the right result in col-major order, we swap the
      // order of the inputs and invert the transpose_* flags, taking advantage
      // of the identity:
      //
      //    (A @ B)ᵀ = Bᵀ @ Aᵀ
      steel_matmul(
          /* const Stream& s = */ stream,
          /* metal::Device& d = */ device,
          /* const array& a = */ a_restblock,
          /* const array& b = */ v_array,
          /* array& out = */ Wp,
          /* int M = */ startc + R - col,
          /* int N = */ 1,
          /* int K = */ m,
          /* int batch_size_out = */ 1,
          /* int lda = */ m,
          /* int ldb = */ m,
          /* bool transpose_a = */ false,
          /* bool transpose_b = */ true,
          /* std::vector<array>& = */ copies,
          /* std::vector<int> batch_shape = */ {1},
          /* std::vector<size_t> A_batch_stride = */ {0},
          /* std::vector<size_t> B_batch_stride = */ {0});
    } else {
      // Use the first column of Wp as a temporary of size (startc + R - jj).
      // It's guaranteed to be big enough because Wp has shape r x n.
      float* va = wp_data;

      const float* v = y_data + colmajor_idx(0, jj, m);
      const float beta = betas_data[jj];

      for (int j = 0; j < startc + R - col; j++) {
        va[j] = 0;
      }

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < startc + R - col; j++) {
          va[j] += v[i] * a_data[colmajor_idx(i, col + j, m)];
        }
      }
    }

    // Then finish updating the rest of the current block with the following
    // outer product:
    //
    //    a[:, col:startc+r] -= beta * np.outer(v, c)
    if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
      auto kernel = device.get_kernel("qrf_reflect_current_block");

      compute_encoder->setComputePipelineState(kernel);
      compute_encoder.set_input_array(betas, 0);
      compute_encoder.set_input_array(Y, 1);
      compute_encoder.set_output_array(a, 2);
      compute_encoder.set_input_array(Wp, 3);
      compute_encoder->setBytes(&jj, sizeof(jj), 4);
      compute_encoder->setBytes(&startc, sizeof(startc), 5);
      compute_encoder->setBytes(&m, sizeof(m), 6);
      compute_encoder->setBytes(&R, sizeof(R), 7);

      const MTL::Size threads_per_grid(m, 1, 1);
      const MTL::Size threads_per_threadgroup(
          std::min(threads_per_grid.width, kernel->threadExecutionWidth()),
          1,
          1);
      compute_encoder->dispatchThreads(
          threads_per_grid, threads_per_threadgroup);
    } else {
      float* va = wp_data;

      const float* v = y_data + colmajor_idx(0, jj, m);
      const float beta = betas_data[jj];

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < startc + R - col; j++) {
          a_data[colmajor_idx(i, col + j, m)] -= beta * v[i] * va[j];
        }
      }
    }
  }

  // Precompute Yp = Y.T @ Y
  if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
    auto kernel = device.get_kernel("qrf_compute_yp");

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(Y, 0);
    compute_encoder.set_output_array(Yp, 1);
    compute_encoder->setBytes(&m, sizeof(m), 2);
    compute_encoder->setBytes(&R, sizeof(R), 3);

    // Specify the grid dimension and use the largest possible threads groups,
    // see
    // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
    const MTL::Size threads_per_grid(R, R, 1);
    const int threadgroup_width =
        std::min(threads_per_grid.width, kernel->threadExecutionWidth());
    const MTL::Size threads_per_threadgroup(
        std::min(threads_per_grid.width, kernel->threadExecutionWidth()),
        std::min(
            threads_per_grid.height,
            kernel->maxTotalThreadsPerThreadgroup() / threadgroup_width),
        1);
    compute_encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
  } else {
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < R; j++) {
        float res = 0;
        for (int k = 0; k < m; k++) {
          res += y_data[colmajor_idx(k, i, m)] * y_data[colmajor_idx(k, j, m)];
        }
        yp_data[colmajor_idx(i, j, R)] = res;
      }
    }
  }

  // Compute remaining columns of W one column at a time.
  if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
    auto kernel = device.get_kernel("qrf_compute_w_col");

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(betas, 0);
    compute_encoder.set_input_array(Y, 1);
    compute_encoder.set_input_array(Yp, 2);
    compute_encoder.set_output_array(W, 3);
    compute_encoder->setBytes(&m, sizeof(m), 4);
    compute_encoder->setBytes(&R, sizeof(R), 5);

    const MTL::Size threads_per_grid(m, 1, 1);
    const MTL::Size threads_per_threadgroup(
        std::min(threads_per_grid.width, kernel->threadExecutionWidth()), 1, 1);
    compute_encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
  } else {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < R; j++) {
        const auto loc = colmajor_idx(i, j, m);
        float z = 0;
        z = 0;
        z -= betas_data[j] * y_data[loc];
        for (int k = 0; k < j; k++) {
          z -= betas_data[j] * w_data[colmajor_idx(i, k, m)] *
              yp_data[colmajor_idx(k, j, R)];
        }
        w_data[loc] = z;
      }
    }
  }

  // Create a view of the part of A right of the current block, ie
  // a[:, startc+r:]).
  auto a_rest_shape = a.shape();
  a_rest_shape[a_rest_shape.size() - 1] = n - (startc + R);
  array a_rest(a_rest_shape, a.dtype(), nullptr, {});
  const size_t a_rest_offset = m * (startc + R);
  a_rest.copy_shared_buffer(
      a, a.strides(), a.flags(), a_rest.size(), a_rest_offset);

  // Apply reflection to the rest of A right of the current block (if any). We
  // only do this if this is not the last block.
  if (startc + R < n) {
    if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
      std::vector<array> copies;

      // See comment above about using a transpose trick with the matmul kernel.
      steel_matmul(
          /* const Stream& s = */ stream,
          /* metal::Device& d = */ device,
          /* const array& a = */ a_rest,
          /* const array& b = */ W,
          /* array& out = */ Wp,
          /* int M = */ n - (startc + R),
          /* int N = */ R,
          /* int K = */ m,
          /* int batch_size_out = */ 1,
          /* int lda = */ m,
          /* int ldb = */ m,
          /* bool transpose_a = */ false,
          /* bool transpose_b = */ true,
          /* std::vector<array>& = */ copies,
          /* std::vector<int> batch_shape = */ {1},
          /* std::vector<size_t> A_batch_stride = */ {0},
          /* std::vector<size_t> B_batch_stride = */ {0});
    } else {
      // wp = w_data.T @ a[:, startc+r:]
      for (int i = 0; i < R; i++) {
        for (int j = 0; startc + R + j < n; j++) {
          float res = 0;
          for (int k = 0; k < m; k++) {
            res += w_data[colmajor_idx(k, i, m)] *
                a_data[colmajor_idx(k, startc + R + j, m)];
          }
          wp_data[colmajor_idx(i, j, R)] = res;
        }
      }
    }

    if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
      std::vector<array> copies;

      int bm = 32, bn = 32, bk = 16;
      int wm = 2, wn = 2;

      bool transpose_a = false;
      bool transpose_b = false;

      // See comment above about using a transpose trick with the matmul kernel.
      const auto M = n - startc - R;
      const auto N = m;

      const auto K = R;
      const float alpha = 1;
      const float beta = 1;

      int tn = (N + bn - 1) / bn;
      int tm = (M + bm - 1) / bm;
      int swizzle_log = 0;

      // From AddMM::eval_gpu()
      std::ostringstream kname;
      kname << "steel_addmm_" << (transpose_a ? 't' : 'n')
            << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
            << type_to_name(a) << "_bm" << bm << "_bn" << bn << "_bk" << bk
            << "_wm" << wm << "_wn" << wn << "_MN_"
            << ((M % bm == 0 && N % bn == 0) ? "t" : "n") << "aligned"
            << "_K_" << ((K % bk == 0) ? "t" : "n") << "aligned"
            << ((alpha == 1. && beta == 1.) ? "_add" : "_axpby");

      auto kernel = device.get_kernel(kname.str());
      compute_encoder->setComputePipelineState(kernel);

      steel::GEMMParams gemm_params{
          /* const int M = */ M,
          /* const int N = */ N,
          /* const int K = */ K,
          /* const int lda = */ R,
          /* const int ldb = */ m,
          /* const int ldd = */ N,
          /* const int tiles_n = */ tn,
          /* const int tiles_m = */ tm,
          /* const int batch_stride_a = */ 0,
          /* const int batch_stride_b = */ 0,
          /* const int batch_stride_d = */ M * N,
          /* const int swizzle_log = */ swizzle_log,
          /* const int gemm_k_iterations_aligned = */ (K / bk),
          /* const int batch_ndim = */ 1};

      steel::GEMMAddMMParams params{
          /* const int ldc = */ m,
          /* const int fdc = */ 1,
          /* const int batch_stride_c = */ 0,
          /* const float alpha = */ alpha,
          /* const float beta = */ beta};

      const int batch_size_out = 1;
      const int tile = 1 << swizzle_log;
      tm = (tm + tile - 1) / tile;
      tn = tn * tile;

      MTL::Size group_dims = MTL::Size(32, wn, wm);
      MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

      std::vector<size_t> batch_shape{1};
      std::vector<size_t> batch_strides{0, 0, 0};

      compute_encoder.set_input_array(Wp, 0); // a
      compute_encoder.set_input_array(Y, 1); // b
      compute_encoder.set_input_array(a_rest, 2); // c
      compute_encoder.set_output_array(a_rest, 3); // out

      compute_encoder->setBytes(&gemm_params, sizeof(steel::GEMMParams), 4);
      compute_encoder->setBytes(&params, sizeof(steel::GEMMAddMMParams), 5);

      compute_encoder->setBytes(
          batch_shape.data(), sizeof(int) * batch_shape.size(), 6);
      compute_encoder->setBytes(
          batch_strides.data(), sizeof(size_t) * batch_strides.size(), 7);

      compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
    } else {
      // a[:, startc+r:] += y_data @ wp_data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j + startc + R < n; j++) {
          float res = 0;
          for (int k = 0; k < R; k++) {
            res +=
                y_data[colmajor_idx(i, k, m)] * wp_data[colmajor_idx(k, j, R)];
          }

          const int a_loc = colmajor_idx(i, j + startc + R, m);
          a_data[a_loc] += res;
        }
      }
    }
  }

  // wy_data = w_data @ y_data.T of shape m x m.
  if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
    std::vector<array> copies;

    // See comment above about using a transpose trick with the matmul kernel.
    steel_matmul(
        /* const Stream& s = */ stream,
        /* metal::Device& d = */ device,
        /* const array& a = */ Y,
        /* const array& b = */ W,
        /* array& out = */ WY,
        /* int M = */ m,
        /* int N = */ m,
        /* int K = */ R,
        /* int batch_size_out = */ 1,
        /* int lda = */ m,
        /* int ldb = */ m,
        /* bool transpose_a = */ true,
        /* bool transpose_b = */ false,
        /* std::vector<array>& = */ copies,
        /* std::vector<int> batch_shape = */ {1},
        /* std::vector<size_t> A_batch_stride = */ {0},
        /* std::vector<size_t> B_batch_stride = */ {0});
  } else {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float res = 0;
        for (int k = 0; k < R; k++) {
          res += w_data[colmajor_idx(i, k, m)] * y_data[colmajor_idx(j, k, m)];
        }
        wy_data[colmajor_idx(i, j, m)] = res;
      }
    }
  }

  // q += q @ wy_data, done in two steps.

  // First, qwy = q @ wy.
  if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
    std::vector<array> copies;

    // See comment above about using a transpose trick with the matmul kernel.
    steel_matmul(
        /* const Stream& s = */ stream,
        /* metal::Device& d = */ device,
        /* const array& a = */ WY,
        /* const array& b = */ q,
        /* array& out = */ QWY,
        /* int M = */ m,
        /* int N = */ m,
        /* int K = */ m,
        /* int batch_size_out = */ 1,
        /* int lda = */ m,
        /* int ldb = */ m,
        /* bool transpose_a = */ false,
        /* bool transpose_b = */ false,
        /* std::vector<array>& = */ copies,
        /* std::vector<int> batch_shape = */ {1},
        /* std::vector<size_t> A_batch_stride = */ {0},
        /* std::vector<size_t> B_batch_stride = */ {0});
  } else {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float res = 0;
        for (int k = 0; k < m; k++) {
          res += q_data[colmajor_idx(i, k, m)] * wy_data[colmajor_idx(k, j, m)];
        }
        qwy_data[colmajor_idx(i, j, m)] = res;
      }
    }
  }

  // Then, q += qwy
  if constexpr (!RUN_ON_CPU_FOR_DEBUG) {
    auto binary_add_kernel = device.get_kernel("vvaddfloat32");

    compute_encoder->setComputePipelineState(binary_add_kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(QWY, 1);
    compute_encoder.set_output_array(q, 2);

    const size_t nthreads = m * m;
    const MTL::Size grid_dims(nthreads, 1, 1);
    const NS::UInteger thread_group_size =
        std::min(nthreads, binary_add_kernel->maxTotalThreadsPerThreadgroup());
    const MTL::Size group_dims(thread_group_size, 1, 1);

    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        const auto loc = colmajor_idx(i, j, m);
        q_data[loc] += qwy_data[loc];
      }
    }
  }
}

void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 2);

  // The algorithm clobbers the input, so make a copy.
  auto& a = outputs[1];
  {
    auto flags = inputs[0].flags();
    flags.row_contiguous = false;
    // We want the data in col-major layout because of how the kernel reads
    // it. There's no relationship between the QR factorization of a matrix
    // and its transpose's, so there's no way to avoid this.
    std::vector<size_t> strides = inputs[0].strides();
    strides[a.ndim() - 2] = 1;
    strides[a.ndim() - 1] = a.shape(-2);
    a.set_data(
        allocator::malloc_or_wait(a.nbytes()), a.nbytes(), strides, flags);
    copy_inplace(inputs[0], a, CopyType::GeneralGeneral);
  }

  const auto m = a.shape(-2);
  const auto n = a.shape(-1);
  // Number of matrices for stacked inputs.
  const int batch_size = a.size() / (m * n);
  auto& q = outputs[0];
  auto& r = outputs[1];

  {
    auto flags = q.flags();
    flags.row_contiguous = false;
    std::vector<size_t> strides = q.strides();
    strides[q.ndim() - 2] = 1;
    strides[q.ndim() - 1] = q.shape(-2);
    q.set_data(
        allocator::malloc_or_wait(q.nbytes()), q.nbytes(), strides, flags);
  }

  // Fill Q with identity.
  for (int k = 0; k < batch_size; k++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        const auto batch_offset = m * n * k;
        const auto loc = batch_offset + colmajor_idx(i, j, m);
        q.data<float>()[loc] = i == j ? 1 : 0;
      }
    }
  }

  // Number of columns for blocked Householder rotations.
  constexpr int max_col_block_size = QRFParams::max_col_block_size;

  // Allocate W and Y temporaries of shape m x r.
  const std::vector<int> w_shape{m, max_col_block_size};
  const std::vector<size_t> w_strides{1, static_cast<unsigned long>(m)};
  array temp_y(w_shape, a.dtype(), nullptr, {});
  temp_y.set_data(
      allocator::malloc_or_wait(temp_y.nbytes()),
      temp_y.nbytes(),
      w_strides,
      a.flags());
  array temp_w(w_shape, a.dtype(), nullptr, {});
  temp_w.set_data(
      allocator::malloc_or_wait(temp_w.nbytes()),
      temp_w.nbytes(),
      w_strides,
      a.flags());

  // Allocate Wp temporary of shape r x n.
  const std::vector<int> wp_shape{max_col_block_size, n};
  array temp_wp(wp_shape, a.dtype(), nullptr, {});
  temp_wp.set_data(allocator::malloc_or_wait(temp_wp.nbytes()));

  // Allocate Yp temporary of shape r x r.
  const std::vector<int> yp_shape{max_col_block_size, max_col_block_size};
  array temp_yp(yp_shape, a.dtype(), nullptr, {});
  temp_yp.set_data(allocator::malloc_or_wait(temp_yp.nbytes()));

  // Allocate WY and QWY temporaries of shape m x m.
  const std::vector<int> wy_shape{m, m};
  array temp_wy(wy_shape, a.dtype(), nullptr, {});
  temp_wy.set_data(allocator::malloc_or_wait(temp_wy.nbytes()));
  array temp_qwy(wy_shape, a.dtype(), nullptr, {});
  temp_qwy.set_data(allocator::malloc_or_wait(temp_qwy.nbytes()));

  // Allocate betas temporary of shape r.
  const std::vector<int> betas_shape{max_col_block_size};
  array temp_betas(betas_shape, a.dtype(), nullptr, {});
  temp_betas.set_data(allocator::malloc_or_wait(temp_betas.nbytes()));

  // Loop over matrices in the input batch. We could avoid this outer loop by
  // lifting the kernels to a 3D grid and using batched matmuls, but for now
  // this is simpler to implement.
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // Create array views over a and q.
    array this_a({m, n}, a.dtype(), nullptr, {});
    const std::vector<size_t> this_a_strides{
        a.strides()[a.ndim() - 2], a.strides()[a.ndim() - 1]};
    this_a.copy_shared_buffer(
        a, this_a_strides, a.flags(), this_a.size(), batch_idx * this_a.size());

    array this_q({m, m}, a.dtype(), nullptr, {});
    const std::vector<size_t> this_q_strides{
        q.strides()[q.ndim() - 2], q.strides()[q.ndim() - 1]};
    this_q.copy_shared_buffer(
        q, this_q_strides, q.flags(), this_q.size(), batch_idx * this_q.size());

    // Loop over the blocks of the current matrix within the batch.
    for (int col = 0; col < n; col += max_col_block_size) {
      int col_block_size = std::min(max_col_block_size, n - col);
      qrf_block(
          q.primitive().stream(),
          this_a,
          this_q,
          m,
          n,
          col_block_size,
          col,
          temp_y,
          temp_w,
          temp_betas,
          temp_yp,
          temp_wy,
          temp_qwy,
          temp_wp);
    }
  }
}

} // namespace mlx::core
