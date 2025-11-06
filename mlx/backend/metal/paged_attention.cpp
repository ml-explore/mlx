// ABOUTME: Implements Metal paged attention decode kernel using streaming
// softmax. ABOUTME: Operates on paged KV blocks to accelerate per-token
// decoding on Apple GPUs.
#include "mlx/backend/metal/paged_attention.h"

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

struct PagedAttentionParams {
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t max_blocks_per_seq;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t has_kv_mapping;
  float scale;
  uint32_t q_batch_stride;
  uint32_t q_head_stride;
  uint32_t kv_head_stride;
  uint32_t block_stride;
  uint32_t row_stride;
  uint32_t out_batch_stride;
  uint32_t out_head_stride;
};

namespace {

std::string kernel_name(const array& q) {
  std::string name = "paged_attention_decode_";
  name += get_type_string(q.dtype());
  return name;
}

} // namespace

bool paged_attention_use_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (q.dtype() != float16 && q.dtype() != float32 && q.dtype() != bfloat16) {
    return true;
  }
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
    return true;
  }
  if (!q.flags().row_contiguous || q.ndim() != 4 || k.ndim() != 4 ||
      v.ndim() != 4) {
    return true;
  }
  if (block_tables.ndim() != 2 || context_lens.ndim() != 1) {
    return true;
  }
  // Ensure per-token dimension contiguous
  if (q.strides().back() != 1 || k.strides().back() != 1 ||
      v.strides().back() != 1) {
    return true;
  }
  auto head_dim = q.shape(3);
  if (head_dim == 0) {
    return true;
  }
  return false;
}

void paged_attention(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    array& out) {
  PagedAttentionParams params;
  params.head_dim = static_cast<uint32_t>(q.shape(3));
  params.block_size = static_cast<uint32_t>(k.shape(2));
  params.max_blocks_per_seq = static_cast<uint32_t>(block_tables.shape(1));
  params.num_q_heads = static_cast<uint32_t>(q.shape(1));
  params.num_kv_heads = static_cast<uint32_t>(k.shape(0));
  params.has_kv_mapping = kv_mapping.has_value() ? 1u : 0u;
  params.scale = scale;
  params.q_batch_stride = static_cast<uint32_t>(q.strides()[0]);
  params.q_head_stride = static_cast<uint32_t>(q.strides()[1]);
  params.kv_head_stride = static_cast<uint32_t>(k.strides()[0]);
  params.block_stride = static_cast<uint32_t>(k.strides()[1]);
  params.row_stride = static_cast<uint32_t>(k.strides()[2]);
  params.out_batch_stride = static_cast<uint32_t>(out.strides()[0]);
  params.out_head_stride = static_cast<uint32_t>(out.strides()[1]);

  auto kernel_id = kernel_name(q);
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(block_tables, 3);
  compute_encoder.set_input_array(context_lens, 4);
  if (kv_mapping.has_value()) {
    compute_encoder.set_input_array(*kv_mapping, 5);
  } else {
    compute_encoder.set_input_array(context_lens, 5);
  }
  compute_encoder.set_output_array(out, 6);
  compute_encoder.set_bytes(params, 7);

  MTL::Size grid_dims(
      static_cast<NS::UInteger>(q.shape(1)),
      static_cast<NS::UInteger>(q.shape(0)),
      1);
  MTL::Size group_dims(1, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace mlx::core::fast
