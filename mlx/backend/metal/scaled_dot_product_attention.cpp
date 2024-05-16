//
//  scaled_dot_product_attention.cpp
//  mlx

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {
void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float alpha,
    array& out,
    std::vector<array>& temporaries) {
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

  constexpr const uint batch = 1;

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

  d.get_command_buffer(s.index)->addCompletedHandler(
      [temporaries](MTL::CommandBuffer*) mutable { temporaries.clear(); });
  return;
}

void sdpa_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& p_lse,
    const array& p_rowmaxes,
    const array& o_partial,
    const uint heads,
    const uint tile_size,
    const uint n_tiles,
    const float alpha,
    array& out,
    std::vector<array>& temporaries) {
  std::ostringstream kname_partials;

  kname_partials << "fast_inference_sdpa_compute_partials_";

  std::ostringstream kname_reduce;
  std::string delimiter = "_";
  kname_reduce << "fast_inference_sdpa_reduce_tiles" + delimiter;

  for (const auto& arr : {k, v, out}) {
    if (arr.dtype() != q.dtype()) {
      throw std::runtime_error(
          "[ScaledDotProductAttention::eval_gpu]: expected matching dtypes for q,k,v,o");
    }
  }

  if (q.dtype() == float32) {
    kname_partials << "float" + delimiter;
    kname_reduce << "float";
  } else if (q.dtype() == float16) {
    kname_partials << "half" + delimiter;
    kname_reduce << "half";
  } else {
    throw std::runtime_error(
        "[ScaledDotProductAttention::eval_gpu]: unexpected dtype found for queries: expected either float32 or float16.");
  }

  std::string kname_suffix_tile_size = std::to_string(tile_size) + delimiter;

  uint nsimd = 8;
  std::string kname_suffix_nsimdgroups = std::to_string(nsimd);

  // maximum number of splits == 128 at the moment (reserved tile registers in
  // reduction kernel). this is arbitrary and could be changed in the shader.

  std::string kname_suffix = kname_suffix_tile_size + kname_suffix_nsimdgroups;
  kname_partials << kname_suffix;
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname_partials.str());
  compute_encoder->setComputePipelineState(kernel);

  constexpr const uint batch = 1;
  MTL::Size grid_dims = MTL::Size(heads, n_tiles, batch);
  MTL::Size group_dims = MTL::Size(32, nsimd, 1);

  const uint64_t KV_sequence_length = k.shape(-2);
  const uint query_sequence_length = q.shape(-2);
  const uint n_q_heads = q.shape(1);
  const uint n_kv_heads = k.shape(1);

  MLXScaledDotProductAttentionParams params{
      query_sequence_length, n_q_heads, n_kv_heads, n_tiles, alpha};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder->setBytes(&KV_sequence_length, sizeof(KV_sequence_length), 3);
  compute_encoder->setBytes(
      &params, sizeof(MLXScaledDotProductAttentionParams), 4);
  compute_encoder.set_input_array(o_partial, 5);
  compute_encoder.set_input_array(p_lse, 6);
  compute_encoder.set_input_array(p_rowmaxes, 7);

  constexpr const uint tgroupMemorySize = 32768;
  compute_encoder->setThreadgroupMemoryLength(tgroupMemorySize, 0);
  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

  {
    auto kernel_accum = d.get_kernel(kname_reduce.str());
    compute_encoder->setComputePipelineState(kernel_accum);
    compute_encoder.set_input_array(o_partial, 0);
    compute_encoder.set_input_array(p_lse, 1);
    compute_encoder.set_input_array(p_rowmaxes, 2);
    compute_encoder->setBytes(
        &params, sizeof(MLXScaledDotProductAttentionParams), 3);
    compute_encoder.set_output_array(out, 4);

    MTL::Size grid_dims_reduce = MTL::Size(heads, 1, batch);
    MTL::Size group_dims_reduce = MTL::Size(128, 1, 1);

    compute_encoder->dispatchThreadgroups(grid_dims_reduce, group_dims_reduce);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [temporaries](MTL::CommandBuffer*) mutable { temporaries.clear(); });
    return;
  }
}
} // namespace

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  assert(inputs.size() >= 3);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[ScaledDotProductAttention] Does not yet support non-floating point types.");
  }

  if (inputs.size() == 4) {
    out = fallback_(inputs)[0];
    return;
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = out;
  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> temporaries;
  auto check_transpose = [&temporaries, &s](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      temporaries.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return arr_copy;
    }
  };

  auto q = check_transpose(q_pre);
  auto k = check_transpose(k_pre);
  auto v = check_transpose(v_pre);

  const int heads = q.shape(-3);

  uint query_sequence_length = q.shape(-2);
  if (query_sequence_length >= 16) {
    return sdpa_full_self_attention_metal(
        s, d, q, k, v, scale_, out, temporaries);
  }
  int tile_size = 64;
  const int kv_seq_len = k.shape(-2);
  if (kv_seq_len > 8000) {
    tile_size = 128;
  }
  if (kv_seq_len > 16000) {
    tile_size = 256;
  }
  if (kv_seq_len > 32000) {
    tile_size = 512;
  }

  const int n_tiles = (kv_seq_len + tile_size - 1) / tile_size;

  array o_partials(
      {q.shape(-4), q.shape(-3), q.shape(-2), n_tiles * v.shape(-1)},
      float32,
      nullptr,
      {});
  o_partials.set_data(allocator::malloc_or_wait(o_partials.nbytes()));

  array p_lse(
      {q.shape(-4), q.shape(-3), q.shape(-2), n_tiles}, float32, nullptr, {});
  array p_rowmaxes(
      {q.shape(-4), q.shape(-3), q.shape(-2), n_tiles}, float32, nullptr, {});
  p_lse.set_data(allocator::malloc_or_wait(p_lse.nbytes()));
  p_rowmaxes.set_data(allocator::malloc_or_wait(p_rowmaxes.nbytes()));

  temporaries.push_back(p_lse);
  temporaries.push_back(p_rowmaxes);
  temporaries.push_back(o_partials);

  return sdpa_metal(
      s,
      d,
      q,
      k,
      v,
      p_lse,
      p_rowmaxes,
      o_partials,
      heads,
      tile_size,
      n_tiles,
      scale_,
      out,
      temporaries);
}

} // namespace mlx::core::fast
