// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    d.add_temporary(x_copy, s.index);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  auto stride_0 = x.strides()[x.ndim() - 2];
  auto stride_1 = x.strides()[x.ndim() - 1];
  if (stride_0 == x.shape(-1) && stride_1 == 1) {
    return x;
  } else {
    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    d.add_temporary(x_copy, s.index);
    return x_copy;
  }
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch = d.get_architecture();
  auto arch_size = arch.back();
  auto arch_gen = arch.substr(arch.size() - 3, 2);
  if (arch_gen == "13" || arch_gen == "14") {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int offset) {
  if (skip) {
    return 0;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset);
  compute_encoder.set_vector_bytes(x.shape(), offset + 1);
  compute_encoder.set_vector_bytes(x.strides(), offset + 2);
  compute_encoder.set_bytes(w_batch_ndims, offset + 3);
  compute_encoder.set_vector_bytes(w.shape(), offset + 4);
  compute_encoder.set_vector_bytes(w.strides(), offset + 5);
  compute_encoder.set_vector_bytes(scales.strides(), offset + 6);
  compute_encoder.set_vector_bytes(biases.strides(), offset + 7);

  return 8;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset);
  compute_encoder.set_vector_bytes(shape, offset + 1);
  compute_encoder.set_vector_bytes(strides[0], offset + 2);
  compute_encoder.set_vector_bytes(strides[1], offset + 3);

  return 4;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      "qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto template_def = get_template_definition(
      kname, "qmv_quad", type_string, group_size, bits, K, B > 1);

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(K, 5);
  compute_encoder.set_bytes(N, 6);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, 7);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      fast ? "qmv_fast_" : "qmv_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto template_def = get_template_definition(
      kname, fast ? "qmv_fast" : "qmv", type_string, group_size, bits, B > 1);

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(K, 5);
  compute_encoder.set_bytes(N, 6);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, 7);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;
  B *= split_k;

  int bn = 64;
  int bk = 32;
  MTL::Size group_dims = MTL::Size(bk, 2, 1);
  MTL::Size grid_dims = MTL::Size(M, N / bn, B);

  int x_batch_ndims = x.ndim() - 2;
  auto x_shape = x.shape();
  auto x_strides = x.strides();
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();
  auto b_strides = biases.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x.ndim() - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));
  b_strides.insert(b_strides.end() - 2, split_D * biases.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  d.add_temporary(intermediate, s.index);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);
  auto template_def = get_template_definition(
      kname, "qvm_split_k", type_string, group_size, bits, split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(intermediate, 4);
  compute_encoder.set_bytes(split_D, 5);
  compute_encoder.set_bytes(N, 6);

  compute_encoder.set_bytes(x_batch_ndims, 7);
  compute_encoder.set_vector_bytes(x_shape, 8);
  compute_encoder.set_vector_bytes(x_strides, 9);
  compute_encoder.set_bytes(w_batch_ndims, 10);
  compute_encoder.set_vector_bytes(w_shape, 11);
  compute_encoder.set_vector_bytes(w_strides, 12);
  compute_encoder.set_vector_bytes(s_strides, 13);
  compute_encoder.set_vector_bytes(b_strides, 14);
  compute_encoder.set_bytes(final_block_size, 15);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int bn = 64;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      "qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto template_def = get_template_definition(
      kname, "qvm", type_string, group_size, bits, B > 1);

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(K, 5);
  compute_encoder.set_bytes(N, 6);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, 7);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      transpose ? "qmm_t_" : "qmm_n_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  if (transpose) {
    template_def = get_template_definition(
        kname, "qmm_t", type_string, group_size, bits, aligned, batched);
  } else {
    template_def = get_template_definition(
        kname, "qmm_n", type_string, group_size, bits, batched);
  }

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(K, 5);
  compute_encoder.set_bytes(N, 6);
  compute_encoder.set_bytes(M, 7);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, 8);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      transpose ? "gather_qmm_t_" : "gather_qmm_n_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  std::string template_def;
  if (transpose) {
    template_def = get_template_definition(
        kname, "gather_qmm_t", type_string, group_size, bits, aligned);
  } else {
    template_def = get_template_definition(
        kname, "gather_qmm_n", type_string, group_size, bits);
  }

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_input_array(lhs_indices, 4);
  compute_encoder.set_input_array(rhs_indices, 5);
  compute_encoder.set_output_array(out, 6);
  compute_encoder.set_bytes(K, 7);
  compute_encoder.set_bytes(N, 8);
  compute_encoder.set_bytes(M, 9);
  int n =
      add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, 10);
  add_gather_strides_and_shapes(
      compute_encoder, lhs_indices, rhs_indices, 10 + n);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      fast ? "gather_qmv_fast_" : "gather_qmv_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto template_def = get_template_definition(
      kname,
      fast ? "gather_qmv_fast" : "gather_qmv",
      type_string,
      group_size,
      bits);

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_input_array(lhs_indices, 4);
  compute_encoder.set_input_array(rhs_indices, 5);
  compute_encoder.set_output_array(out, 6);
  compute_encoder.set_bytes(K, 7);
  compute_encoder.set_bytes(N, 8);
  int n =
      add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, 9);
  add_gather_strides_and_shapes(
      compute_encoder, lhs_indices, rhs_indices, 9 + n);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  int B = out.size() / M / N;

  int bn = 64;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname, "gather_qvm_", type_string, "_gs_", group_size, "_b_", bits);
  auto template_def = get_template_definition(
      kname, "gather_qvm", type_string, group_size, bits);

  auto kernel = get_quantized_kernel(d, kname, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_input_array(lhs_indices, 4);
  compute_encoder.set_input_array(rhs_indices, 5);
  compute_encoder.set_output_array(out, 6);
  compute_encoder.set_bytes(K, 7);
  compute_encoder.set_bytes(N, 8);
  int n =
      add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, 9);
  add_gather_strides_and_shapes(
      compute_encoder, lhs_indices, rhs_indices, 9 + n);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const array& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);
  array biases = ensure_row_contiguous(biases_, d, s);

  // TODO: Tune the block sizes
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      transpose ? "gather_qmm_rhs_nt_" : "gather_qmm_rhs_nn_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(scales, 2);
  compute_encoder.set_input_array(biases, 3);
  compute_encoder.set_input_array(indices, 4);
  compute_encoder.set_output_array(out, 5);
  compute_encoder.set_bytes(M, 6);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(K, 8);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  array biases = ensure_row_contiguous_matrix(inputs[3], d, s);

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // It is a matrix matrix product.
  if (M >= vector_limit) {
    qmm(x,
        w,
        scales,
        biases,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s);
    return;
  }

  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if (transpose_ && (K == 128 || K == 64) && is_power_of_2(bits_)) {
    qmv_quad(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    qmv(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    qvm(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  array biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  const array& lhs_indices = inputs[4];
  const array& rhs_indices = inputs[5];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // We are walking x in order and w is also in order so we can batch up the
  // matmuls and reuse reading x and w.
  //
  // TODO: Tune 16 and 8 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 8) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s);
}

void fast::AffineQuantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s.index);

  auto w = ensure_row_contiguous(w_pre, d, s);
  compute_encoder.set_input_array(w, 0);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    auto biases = ensure_row_contiguous(inputs[2], d, s);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_input_array(biases, 2);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    auto& biases = outputs[2];
    scales.set_data(allocator::malloc(scales.nbytes()));
    biases.set_data(allocator::malloc(biases.nbytes()));
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
    compute_encoder.set_output_array(biases, 3);
  }

  std::ostringstream kname;
  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto kernel_func = dequantize_ ? "affine_dequantize" : "affine_quantize";
  kname << kernel_func << "_" << type_string << "_gs_" << group_size_ << "_b_"
        << bits_;
  auto template_def = get_template_definition(
      kname.str(), kernel_func, type_string, group_size_, bits_);
  auto kernel = get_quantized_kernel(d, kname.str(), template_def);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = bits_ == 3 ? 8 : bits_ == 6 ? 4 : 8 / bits_;
  int per_thread = dequantize_ ? packs_per_int : group_size_ / simd_size;
  size_t nthreads =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
