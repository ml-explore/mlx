// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI in MSVC.
#define _USE_MATH_DEFINES

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include "mlx/paged_attention_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::paged_attention {

array paged_attention(
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    int max_context_len,
    float softmax_scale,
    std::optional<array> alibi_slopes = std::nullopt,
    std::optional<float> softcapping = std::nullopt,
    StreamOrDevice s_ = {}) {
  auto s = to_stream(s_);

  // supported dtypes
  if (!issubdtype(q.dtype(), floating)) {
    throw std::invalid_argument(
        "[paged_attention] Only real floating types are supported");
  }
  if (!(q.dtype() == k_cache.dtype() && k_cache.dtype() == v_cache.dtype())) {
    throw std::invalid_argument(
        "[paged_attention] q/k_cache/v_cache dtype must match");
  }
  if (!(block_tables.dtype() == uint32 && context_lens.dtype() == uint32)) {
    throw std::invalid_argument(
        "[paged_attention] block_tables/context_lens dtype must be uint32");
  }

  // rank checks
  if (q.ndim() != 3)
    throw std::invalid_argument("[paged_attention] `q` must be rank-3");
  if (k_cache.ndim() != 5)
    throw std::invalid_argument("[paged_attention] `k_cache` must be rank-5");
  if (v_cache.ndim() != 4)
    throw std::invalid_argument("[paged_attention] `v_cache` must be rank-4");
  if (block_tables.ndim() != 2)
    throw std::invalid_argument(
        "[paged_attention] `block_tables` must be rank-2");
  if (context_lens.ndim() != 1)
    throw std::invalid_argument(
        "[paged_attention] `context_lens` must be rank-1");

  // 4. Shape consistency
  const auto& q_shape = q.shape(); // [num_seqs, num_heads, head_size]
  const auto& kc_shape = k_cache.shape();
  const auto& vc_shape = v_cache.shape();
  const auto& bt_shape = block_tables.shape();
  const auto& cl_shape = context_lens.shape();

  int64_t num_seqs = q_shape[0];
  int64_t num_heads = q_shape[1];
  int64_t head_size = q_shape[2];

  // Allowed head sizes
  switch (head_size) {
    case 64:
    case 80:
    case 96:
    case 112:
    case 128:
    case 192:
    case 256:
      break;
    default:
      throw std::invalid_argument(
          "[paged_attention] `head_size` must be one of "
          "{64, 80, 96, 112, 128, 192, 256}");
  }

  // block_tables first dimension must match num_seqs
  if (bt_shape[0] != num_seqs) {
    std::stringstream ss;
    ss << "[paged_attention] block_tables.shape[0] (" << bt_shape[0]
       << ") must equal q.shape[0] (" << num_seqs << ")";
    throw std::invalid_argument(ss.str());
  }

  // Extract k_cache dimensions
  int64_t num_blocks = kc_shape[0];
  int64_t num_kv_heads = kc_shape[1];
  int64_t head_size_kc = kc_shape[2];
  int64_t block_size = kc_shape[3];
  int64_t x = kc_shape[4];

  if (head_size_kc * x != head_size) {
    std::stringstream ss;
    ss << "[paged_attention] k_cache head_size (" << head_size_kc << " * " << x
       << ") must equal q head_size (" << head_size << ")";
    throw std::invalid_argument(ss.str());
  }

  // v_cache must match the derived dimensions
  if (!(vc_shape[0] == num_blocks && vc_shape[1] == num_kv_heads &&
        vc_shape[2] == head_size && vc_shape[3] == block_size)) {
    throw std::invalid_argument(
        "[paged_attention] `v_cache` shape mismatch with `k_cache`/`q`");
  }

  // context_lens length must match num_seqs
  if (cl_shape[0] != num_seqs) {
    std::stringstream ss;
    ss << "paged_attention: context_lens length (" << cl_shape[0]
       << ") must equal q.shape[0] (" << num_seqs << ")";
    throw std::invalid_argument(ss.str());
  }

  constexpr int64_t partition_size = 512;
  int64_t max_num_partitions =
      (max_context_len + partition_size - 1) / partition_size; // ceil‑div
  bool use_v1 = ((max_num_partitions == 1) || (num_seqs * num_heads > 512)) &&
      (partition_size % block_size == 0);

  auto out_shape = q.shape();

  auto inputs = std::vector{
      std::move(q),
      std::move(k_cache),
      std::move(v_cache),
      std::move(block_tables),
      std::move(context_lens)};
  if (alibi_slopes.has_value()) {
    inputs.push_back(std::move(alibi_slopes.value()));
  }

  return array(
      std::move(out_shape),
      q.dtype(),
      std::make_shared<PagedAttention>(
          to_stream(s), use_v1, max_context_len, softmax_scale, softcapping),
      inputs);
}

} // namespace mlx::core::paged_attention
