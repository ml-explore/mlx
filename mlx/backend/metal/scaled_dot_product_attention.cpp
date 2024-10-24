// Copyright © 2024 Apple Inc.

#include <cassert>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {
void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float alpha,
    array& out) {
  std::ostringstream kname_self_attention;
  kname_self_attention << "steel_gemm_attention_";

  constexpr const int bm = 16;
  constexpr const int bn = 16;
  const int bk = q.shape(-1); // already forced to be 64 or 128

  if (bk != 64 && bk != 128) {
    throw std::runtime_error(
        "[ScaledDotProductAttention::eval_gpu]: hidden dim: expected either 64, 128");
  }

  constexpr const int wm = 2;
  constexpr const int wn = 2;

  std::string delimiter = "_";

  kname_self_attention << "bm_" + std::to_string(bm) + delimiter;
  kname_self_attention << "bn_" + std::to_string(bn) + delimiter;
  kname_self_attention << "bk_" + std::to_string(bk) + delimiter;

  for (const auto& arr : {k, v, out}) {
    if (arr.dtype() != q.dtype()) {
      throw std::runtime_error(
          "[ScaledDotProductAttention::eval_gpu]: expected matching dtypes for q,k,v,o");
    }
  }

  if (q.dtype() == float32) {
    kname_self_attention << "itype" + delimiter + "float";
  } else if (q.dtype() == float16) {
    kname_self_attention << "itype" + delimiter + "half";
  } else {
    throw std::runtime_error(
        "[ScaledDotProductAttention::eval_gpu]: unexpected dtype found for queries: expected either float32 or float16.");
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname_self_attention.str());
  compute_encoder->setComputePipelineState(kernel);

  uint hidden_dim = q.shape(-1);
  uint qseq = q.shape(-2);
  uint qheads = q.shape(-3);

  const uint64_t KV_sequence_length = k.shape(-2);
  const uint query_sequence_length = q.shape(-2);
  const uint n_q_heads = q.shape(1);
  const uint n_kv_heads = k.shape(1);

  const int M = q.shape(-2);
  const int N = M;
  const int K = q.shape(-1);
  const size_t batch_size_out = q.shape(0) * q.shape(1);

  const std::vector<int> batch_shape = {q.shape(0) * q.shape(1)};
  const int dk = q.shape(-1);
  const int ldq = dk;
  const int ldk = dk;
  const int ldv = dk;
  const int lds = bn;
  const int ldo = dk;

  int tn = 1;
  int tm = (M + bm - 1) / bm;

  const int batch_stride_q = dk * query_sequence_length;
  const int batch_stride_k = dk * query_sequence_length;
  const int batch_stride_v = dk * query_sequence_length;
  const int batch_stride_o = dk * query_sequence_length;
  const int swizzle_log = 0;
  const int gemm_n_iterations_aligned = (N + bn - 1) / bn;
  const int gemm_k_iterations_aligned = (K + bk - 1) / bk;
  const int gemm_sv_m_block_iterations = (M + bm - 1) / bm;
  const int batch_ndim = int(batch_shape.size());

  MLXFastAttentionParams params{
      (int)M,
      (int)N,
      (int)K,
      ldq,
      ldk,
      ldv,
      lds,
      ldo,
      tn,
      tm,
      batch_stride_q,
      batch_stride_k,
      batch_stride_v,
      batch_stride_o,
      swizzle_log,
      gemm_n_iterations_aligned,
      gemm_k_iterations_aligned,
      gemm_sv_m_block_iterations,
      batch_ndim,
      alpha};

  const std::vector<size_t> batch_strides = {
      (size_t)batch_stride_q,
      (size_t)batch_stride_k,
      (size_t)batch_stride_v,
      (size_t)batch_stride_o};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&params, sizeof(MLXFastAttentionParams), 4);
  compute_encoder->setBytes(
      batch_shape.data(), sizeof(int) * batch_shape.size(), 6);

  compute_encoder->setBytes(
      batch_strides.data(), sizeof(size_t) * batch_strides.size(), 7);

  MTL::Size grid_dims = MTL::Size(1, tm, batch_size_out);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int B = q.shape(0) * q.shape(1);
  size_t stride = k.strides()[1];
  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(1, B, 1);

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  compute_encoder->setComputePipelineState(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q.data_shared_ptr() == nullptr ? out : q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder->setBytes(&gqa_factor, sizeof(int), 4);
  compute_encoder->setBytes(&N, sizeof(int), 5);
  compute_encoder->setBytes(&stride, sizeof(size_t), 6);
  compute_encoder->setBytes(&scale, sizeof(float), 7);

  // Launch
  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
}

void quant_sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& k_scales,
    const array& k_biases,
    const array& v,
    const array& v_scales,
    const array& v_biases,
    array& out,
    float scale,
    int group_size,
    int bits) {
  // Set the kernel name
  std::string kname;
  kname.reserve(96);
  kname += "quant_sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(group_size);
  kname += "_";
  kname += std::to_string(bits);

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int B = q.shape(0) * q.shape(1);
  size_t stride = k.strides()[1];
  size_t group_stride = k_scales.strides()[1];
  MTL::Size group_dims(128, 1, 1);
  MTL::Size grid_dims(1, B, 1);

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  compute_encoder->setComputePipelineState(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q.data_shared_ptr() == nullptr ? out : q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_input_array(v, 4);
  compute_encoder.set_input_array(v_scales, 5);
  compute_encoder.set_input_array(v_biases, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder->setBytes(&gqa_factor, sizeof(int), 8);
  compute_encoder->setBytes(&N, sizeof(int), 9);
  compute_encoder->setBytes(&stride, sizeof(size_t), 10);
  compute_encoder->setBytes(&group_stride, sizeof(size_t), 11);
  compute_encoder->setBytes(&scale, sizeof(float), 12);

  // Launch
  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
}

} // namespace

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& o = out;

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  auto copy_unless = [&copies, &s](auto predicate, const array& arr) {
    if (!predicate(arr)) {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      return arr_copy;
    } else {
      return arr;
    }
  };

  // Checks if arr is fully row contiguous
  auto is_contiguous = [](const array& arr) {
    return arr.flags().row_contiguous;
  };

  // Returns true if the array is row contiguous except the sequence length
  // dimension that can be sliced but with step=1.
  auto is_contiguous_except_seq_len = [](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    return strides[3] == 1 && strides[2] == shape[3] &&
        strides[0] == strides[1] * shape[1];
  };

  // Checks that the last two dims are row contiguous.
  auto is_matrix_contiguous = [](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    return strides[3] == 1 && strides[2] == shape[3];
  };

  if (quantized_) {
    auto& q_pre = inputs[0];
    auto& k_pre = inputs[1];
    auto& k_scales_pre = inputs[2];
    auto& k_biases_pre = inputs[3];
    auto& v_pre = inputs[4];
    auto& v_scales_pre = inputs[5];
    auto& v_biases_pre = inputs[6];

    // Quantized should only be routed here for single queries
    assert(q_pre.shape(2) == 1);

    auto q = copy_unless(is_contiguous, q_pre);
    auto k = copy_unless(is_contiguous_except_seq_len, k_pre);
    auto k_scales = copy_unless(is_contiguous_except_seq_len, k_scales_pre);
    auto k_biases = copy_unless(is_contiguous_except_seq_len, k_biases_pre);
    auto v = copy_unless(is_contiguous_except_seq_len, v_pre);
    auto v_scales = copy_unless(is_contiguous_except_seq_len, v_scales_pre);
    auto v_biases = copy_unless(is_contiguous_except_seq_len, v_biases_pre);

    // Donate the query if possible
    if (q.is_donatable()) {
      o.move_shared_buffer(q);
    } else {
      o.set_data(allocator::malloc_or_wait(o.nbytes()));
    }

    quant_sdpa_vector(
        s,
        d,
        q,
        k,
        k_scales,
        k_biases,
        v,
        v_scales,
        v_biases,
        o,
        scale_,
        group_size_,
        bits_);

  }

  // Non-quantized
  else {
    assert(inputs.size() == 3);
    auto& q_pre = inputs[0];
    auto& k_pre = inputs[1];
    auto& v_pre = inputs[2];

    // We are in vector mode ie single query
    if (q_pre.shape(2) == 1) {
      auto q = copy_unless(is_contiguous, q_pre);
      auto k = copy_unless(is_contiguous_except_seq_len, k_pre);
      auto v = copy_unless(is_contiguous_except_seq_len, v_pre);

      // Donate the query if possible
      if (q.is_donatable()) {
        o.move_shared_buffer(q);
      } else {
        o.set_data(allocator::malloc_or_wait(o.nbytes()));
      }

      sdpa_vector(s, d, q, k, v, o, scale_);
    }
    // Full attention mode
    else {
      auto q = copy_unless(is_matrix_contiguous, q_pre);
      auto k = copy_unless(is_matrix_contiguous, k_pre);
      auto v = copy_unless(is_matrix_contiguous, v_pre);
      o.set_data(allocator::malloc_or_wait(o.nbytes()));

      sdpa_full_self_attention_metal(s, d, q, k, v, scale_, o);
    }
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
