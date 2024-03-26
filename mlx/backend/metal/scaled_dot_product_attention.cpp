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
  auto compute_encoder = d.get_command_encoder(s.index);
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

  set_array_buffer(compute_encoder, q, 0);
  set_array_buffer(compute_encoder, k, 1);
  set_array_buffer(compute_encoder, v, 2);
  compute_encoder->setBytes(&KV_sequence_length, sizeof(KV_sequence_length), 3);
  compute_encoder->setBytes(
      &params, sizeof(MLXScaledDotProductAttentionParams), 4);
  set_array_buffer(compute_encoder, o_partial, 5);
  set_array_buffer(compute_encoder, p_lse, 6);
  set_array_buffer(compute_encoder, p_rowmaxes, 7);

  constexpr const uint tgroupMemorySize = 32768;
  compute_encoder->setThreadgroupMemoryLength(tgroupMemorySize, 0);
  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

  {
    auto kernel_accum = d.get_kernel(kname_reduce.str());
    compute_encoder->setComputePipelineState(kernel_accum);
    set_array_buffer(compute_encoder, o_partial, 0);
    set_array_buffer(compute_encoder, p_lse, 1);
    set_array_buffer(compute_encoder, p_rowmaxes, 2);
    compute_encoder->setBytes(
        &params, sizeof(MLXScaledDotProductAttentionParams), 3);
    set_array_buffer(compute_encoder, out, 4);

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
