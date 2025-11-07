// ABOUTME: Implements Metal dispatchers for paged KV writes.
// ABOUTME: Handles shape checks and launch parameters for chunk kernels.

#include "mlx/backend/metal/paged_kv.h"

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

struct PagedKVWriteParams {
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t chunk_tokens;
  uint32_t num_kv_heads;
  uint32_t chunk_token_stride;
  uint32_t chunk_head_stride;
  uint32_t kv_head_stride;
  uint32_t block_stride;
  uint32_t row_stride;
};

std::string kernel_name(const array& cache) {
  std::string name = "paged_kv_write_";
  name += get_type_string(cache.dtype());
  return name;
}

} // namespace

bool paged_kv_write_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    const array& k_chunk,
    const array& v_chunk,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (k_cache.dtype() != v_cache.dtype() ||
      k_chunk.dtype() != k_cache.dtype() ||
      v_chunk.dtype() != k_cache.dtype()) {
    return true;
  }
  if (k_cache.ndim() != 4 || v_cache.ndim() != 4 || block_row.ndim() != 1 ||
      k_chunk.ndim() != 3 || v_chunk.ndim() != 3) {
    return true;
  }
  if (!k_cache.flags().row_contiguous || !v_cache.flags().row_contiguous ||
      !k_chunk.flags().row_contiguous || !v_chunk.flags().row_contiguous) {
    return true;
  }
  if (k_cache.shape() != v_cache.shape() ||
      k_chunk.shape() != v_chunk.shape()) {
    return true;
  }
  if (k_chunk.shape(2) != k_cache.shape(3)) {
    return true;
  }
  return false;
}

void paged_kv_write(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    uint32_t start_pos,
    const array& k_chunk,
    const array& v_chunk) {
  PagedKVWriteParams params;
  params.head_dim = static_cast<uint32_t>(k_chunk.shape(2));
  params.block_size = static_cast<uint32_t>(k_cache.shape(2));
  params.chunk_tokens = static_cast<uint32_t>(k_chunk.shape(0));
  params.num_kv_heads = static_cast<uint32_t>(k_chunk.shape(1));
  params.chunk_token_stride = static_cast<uint32_t>(k_chunk.strides()[0]);
  params.chunk_head_stride = static_cast<uint32_t>(k_chunk.strides()[1]);
  params.kv_head_stride = static_cast<uint32_t>(k_cache.strides()[0]);
  params.block_stride = static_cast<uint32_t>(k_cache.strides()[1]);
  params.row_stride = static_cast<uint32_t>(k_cache.strides()[2]);

  auto kernel_id = kernel_name(k_cache);
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  compute_encoder.set_input_array(block_row, 2);
  compute_encoder.set_bytes(start_pos, 3);
  compute_encoder.set_input_array(k_chunk, 4);
  compute_encoder.set_input_array(v_chunk, 5);
  compute_encoder.set_bytes(params, 6);

  const uint32_t work_items = params.chunk_tokens * params.num_kv_heads;
  MTL::Size grid_dims(work_items, 1, 1);
  MTL::Size group_dims(64, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
