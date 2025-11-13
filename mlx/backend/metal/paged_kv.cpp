// ABOUTME: Implements Metal dispatchers for paged KV writes.
// ABOUTME: Handles shape checks and launch parameters for chunk kernels.

#include "mlx/backend/metal/paged_kv.h"

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
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

struct PagedKVQuantKernelParams {
  uint32_t group_size;
  uint32_t groups_per_head;
  uint32_t bits;
  uint32_t bytes_per_token;
  uint32_t symmetric;
  uint32_t vq_head_stride;
  uint32_t vq_block_stride;
  uint32_t vq_row_stride;
  uint32_t scale_head_stride;
  uint32_t scale_block_stride;
  uint32_t scale_row_stride;
};

struct PagedKVWriteBatchParams {
  uint32_t batch_size;
  uint32_t layers;
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t max_blocks;
  uint32_t max_blocks_per_seq;
  uint32_t num_kv_heads;
  uint32_t kv_head_stride;
  uint32_t layer_stride;
  uint32_t block_stride;
  uint32_t row_stride;
  uint32_t batch_head_stride;
  uint32_t batch_seq_stride;
  uint32_t batch_layer_stride;
  uint32_t table_stride;
};

struct PagedKVWriteLayersTokensParams {
  uint32_t batch_size;
  uint32_t layers;
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t max_blocks;
  uint32_t max_blocks_per_seq;
  uint32_t num_kv_heads;
  uint32_t kv_head_stride;
  uint32_t layer_stride;
  uint32_t block_stride;
  uint32_t row_stride;
  uint32_t tokens;
  uint32_t token_layer_stride;
  uint32_t token_step_stride;
  uint32_t token_batch_stride;
  uint32_t token_head_stride;
  uint32_t token_dim_stride;
  uint32_t table_stride;
};

std::string kernel_name(const array& cache) {
  std::string name = "paged_kv_write_";
  name += get_type_string(cache.dtype());
  return name;
}

std::string quant_kernel_name(const array& cache, const array& scale) {
  std::string name = "paged_kv_write_quantized_";
  name += get_type_string(cache.dtype());
  name += "_";
  name += get_type_string(scale.dtype());
  return name;
}

} // namespace

bool paged_kv_write_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    const array& k_chunk,
    const array& v_chunk,
    array* vq_cache,
    array* v_scale_cache,
    array* v_zero_cache,
    const PagedKVQuantConfig* quant,
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
  if (quant != nullptr) {
    if (vq_cache == nullptr || v_scale_cache == nullptr ||
        v_zero_cache == nullptr) {
      return true;
    }
    if (vq_cache->ndim() != 4 || v_scale_cache->ndim() != 4 ||
        v_zero_cache->ndim() != 4) {
      return true;
    }
    if (vq_cache->dtype() != uint8) {
      return true;
    }
    if (v_scale_cache->dtype() != v_zero_cache->dtype()) {
      return true;
    }
    if (v_scale_cache->dtype() != float16) {
      return true;
    }
    if (vq_cache->shape(0) != k_cache.shape(0) ||
        v_scale_cache->shape(0) != k_cache.shape(0) ||
        v_zero_cache->shape(0) != k_cache.shape(0)) {
      return true;
    }
    if (vq_cache->shape(1) != k_cache.shape(1) ||
        v_scale_cache->shape(1) != k_cache.shape(1) ||
        v_zero_cache->shape(1) != k_cache.shape(1)) {
      return true;
    }
    if (vq_cache->shape(2) != k_cache.shape(2) ||
        v_scale_cache->shape(2) != k_cache.shape(2) ||
        v_zero_cache->shape(2) != k_cache.shape(2)) {
      return true;
    }
    if (vq_cache->shape(3) != quant->bytes_per_token) {
      return true;
    }
    if (v_scale_cache->shape(3) != quant->groups_per_head ||
        v_zero_cache->shape(3) != quant->groups_per_head) {
      return true;
    }
    if (quant->bits != 4 && quant->bits != 8) {
      return true;
    }
    if (quant->group_size <= 0 || quant->bytes_per_token <= 0 ||
        quant->groups_per_head <= 0) {
      return true;
    }
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
    const array& v_chunk,
    array* vq_cache,
    array* v_scale_cache,
    array* v_zero_cache,
    const PagedKVQuantConfig* quant) {
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

  const bool use_quant = quant != nullptr;
  auto kernel_id = use_quant ? quant_kernel_name(k_cache, *v_scale_cache)
                             : kernel_name(k_cache);
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  if (use_quant) {
    PagedKVQuantKernelParams quant_params;
    quant_params.group_size = static_cast<uint32_t>(quant->group_size);
    quant_params.groups_per_head =
        static_cast<uint32_t>(quant->groups_per_head);
    quant_params.bits = static_cast<uint32_t>(quant->bits);
    quant_params.bytes_per_token =
        static_cast<uint32_t>(quant->bytes_per_token);
    quant_params.symmetric = quant->symmetric ? 1u : 0u;
    quant_params.vq_head_stride = static_cast<uint32_t>(vq_cache->strides()[0]);
    quant_params.vq_block_stride =
        static_cast<uint32_t>(vq_cache->strides()[1]);
    quant_params.vq_row_stride = static_cast<uint32_t>(vq_cache->strides()[2]);
    quant_params.scale_head_stride =
        static_cast<uint32_t>(v_scale_cache->strides()[0]);
    quant_params.scale_block_stride =
        static_cast<uint32_t>(v_scale_cache->strides()[1]);
    quant_params.scale_row_stride =
        static_cast<uint32_t>(v_scale_cache->strides()[2]);

    compute_encoder.set_input_array(*vq_cache, 2);
    compute_encoder.set_input_array(*v_scale_cache, 3);
    compute_encoder.set_input_array(*v_zero_cache, 4);
    compute_encoder.set_input_array(block_row, 5);
    compute_encoder.set_bytes(start_pos, 6);
    compute_encoder.set_input_array(k_chunk, 7);
    compute_encoder.set_input_array(v_chunk, 8);
    compute_encoder.set_bytes(params, 9);
    compute_encoder.set_bytes(quant_params, 10);
  } else {
    compute_encoder.set_input_array(block_row, 2);
    compute_encoder.set_bytes(start_pos, 3);
    compute_encoder.set_input_array(k_chunk, 4);
    compute_encoder.set_input_array(v_chunk, 5);
    compute_encoder.set_bytes(params, 6);
  }

  const uint32_t work_items = params.chunk_tokens * params.num_kv_heads;
  MTL::Size grid_dims(work_items, 1, 1);
  MTL::Size group_dims(64, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

bool paged_kv_write_batch_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_batch,
    const array& v_batch,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (k_cache.dtype() != v_cache.dtype() ||
      k_batch.dtype() != k_cache.dtype() ||
      v_batch.dtype() != k_cache.dtype()) {
    return true;
  }
  if (k_cache.ndim() != 4 || v_cache.ndim() != 4) {
    return true;
  }
  if (block_tables.ndim() != 2 || context_lens.ndim() != 1) {
    return true;
  }
  if (block_tables.shape(0) != context_lens.shape(0)) {
    return true;
  }
  if (k_batch.ndim() != 3 || v_batch.ndim() != 3) {
    return true;
  }
  if (k_batch.shape() != v_batch.shape()) {
    return true;
  }
  if (k_batch.shape(1) != k_cache.shape(0) ||
      k_batch.shape(2) != k_cache.shape(3)) {
    return true;
  }
  return false;
}

bool paged_kv_write_layers_batch_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_batch,
    const array& v_batch,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (k_cache.dtype() != v_cache.dtype() ||
      k_batch.dtype() != k_cache.dtype() ||
      v_batch.dtype() != k_cache.dtype()) {
    return true;
  }
  if (k_cache.ndim() != 5 || v_cache.ndim() != 5) {
    return true;
  }
  if (block_tables.ndim() != 2 || context_lens.ndim() != 1) {
    return true;
  }
  if (block_tables.shape(0) != k_batch.shape(1) || block_tables.shape(1) == 0 ||
      context_lens.shape(0) != k_batch.shape(1)) {
    return true;
  }
  if (k_batch.ndim() != 4 || v_batch.ndim() != 4) {
    return true;
  }
  if (k_batch.shape() != v_batch.shape()) {
    return true;
  }
  if (k_batch.shape(2) != k_cache.shape(1) ||
      k_batch.shape(3) != k_cache.shape(4)) {
    return true;
  }
  if (k_batch.shape(0) > k_cache.shape(0)) {
    return true;
  }
  return false;
}

bool paged_kv_write_layers_tokens_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_tokens,
    const array& v_tokens,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (k_cache.dtype() != v_cache.dtype() ||
      k_tokens.dtype() != k_cache.dtype() ||
      v_tokens.dtype() != k_cache.dtype()) {
    return true;
  }
  if (k_cache.ndim() != 5 || v_cache.ndim() != 5) {
    return true;
  }
  if (block_tables.ndim() != 2 || context_lens.ndim() != 1) {
    return true;
  }
  if (k_tokens.ndim() != 5 || v_tokens.ndim() != 5) {
    return true;
  }
  if (k_tokens.shape() != v_tokens.shape()) {
    return true;
  }
  if (k_tokens.shape(2) != block_tables.shape(0) ||
      block_tables.shape(1) == 0 ||
      context_lens.shape(0) != k_tokens.shape(2)) {
    return true;
  }
  if (k_tokens.shape(3) != k_cache.shape(1) ||
      k_tokens.shape(4) != k_cache.shape(4)) {
    return true;
  }
  if (k_tokens.shape(0) > k_cache.shape(0)) {
    return true;
  }
  if (k_tokens.shape(1) == 0) {
    return true;
  }
  return false;
}

void paged_kv_write_batch(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_batch,
    const array& v_batch) {
  PagedKVWriteBatchParams params;
  params.batch_size = static_cast<uint32_t>(k_batch.shape(0));
  params.layers = 1;
  params.head_dim = static_cast<uint32_t>(k_batch.shape(2));
  params.block_size = static_cast<uint32_t>(k_cache.shape(2));
  params.max_blocks = static_cast<uint32_t>(k_cache.shape(1));
  params.max_blocks_per_seq = static_cast<uint32_t>(block_tables.shape(1));
  params.num_kv_heads = static_cast<uint32_t>(k_batch.shape(1));
  const uint32_t num_blocks = params.max_blocks;
  params.row_stride = params.head_dim;
  params.block_stride = params.block_size * params.row_stride;
  params.kv_head_stride = num_blocks * params.block_stride;
  params.batch_head_stride = params.head_dim;
  params.batch_seq_stride = params.num_kv_heads * params.head_dim;
  params.layer_stride = 0;
  params.batch_layer_stride = 0;
  params.table_stride = static_cast<uint32_t>(block_tables.strides()[0]);

  auto kernel_id =
      std::string("paged_kv_write_batch_") + get_type_string(k_cache.dtype());
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  compute_encoder.set_input_array(block_tables, 2);
  compute_encoder.set_input_array(context_lens, 3);
  compute_encoder.set_input_array(k_batch, 4);
  compute_encoder.set_input_array(v_batch, 5);
  compute_encoder.set_bytes(params, 6);

  const uint32_t work_items =
      params.batch_size * params.num_kv_heads * params.head_dim;
  constexpr uint32_t kThreadsPerGroup = 128;
  MTL::Size grid_dims(work_items, 1, 1);
  MTL::Size group_dims(kThreadsPerGroup, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void paged_kv_write_layers_batch(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_batch,
    const array& v_batch) {
  if (paged_kv_write_layers_batch_use_fallback(
          k_cache, v_cache, block_tables, context_lens, k_batch, v_batch, s)) {
    throw std::runtime_error(
        "paged_kv_write_layers_batch: configuration not supported on Metal backend.");
  }

  auto stream = to_stream(s);
  auto block_tables_i32 = mlx::core::astype(block_tables, int32, stream);
  auto context_i32 = mlx::core::astype(context_lens, int32, stream);
  auto k_cast = mlx::core::astype(k_batch, k_cache.dtype(), stream);
  auto v_cast = mlx::core::astype(v_batch, k_cache.dtype(), stream);

  PagedKVWriteBatchParams params;
  params.layers = static_cast<uint32_t>(k_batch.shape(0));
  params.batch_size = static_cast<uint32_t>(k_batch.shape(1));
  params.head_dim = static_cast<uint32_t>(k_batch.shape(3));
  params.block_size = static_cast<uint32_t>(k_cache.shape(3));
  params.max_blocks = static_cast<uint32_t>(k_cache.shape(2));
  params.max_blocks_per_seq = static_cast<uint32_t>(block_tables.shape(1));
  params.num_kv_heads = static_cast<uint32_t>(k_batch.shape(2));
  params.row_stride = params.head_dim;
  params.block_stride = params.block_size * params.row_stride;
  params.kv_head_stride = params.max_blocks * params.block_stride;
  params.layer_stride = static_cast<uint32_t>(k_cache.strides()[0]);
  params.batch_head_stride = static_cast<uint32_t>(k_batch.strides()[2]);
  params.batch_seq_stride = static_cast<uint32_t>(k_batch.strides()[1]);
  params.batch_layer_stride = static_cast<uint32_t>(k_batch.strides()[0]);
  params.table_stride = static_cast<uint32_t>(block_tables.strides()[0]);

  auto kernel_id = std::string("paged_kv_write_layers_batch_") +
      get_type_string(k_cache.dtype());
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  compute_encoder.set_input_array(block_tables_i32, 2);
  compute_encoder.set_input_array(context_i32, 3);
  compute_encoder.set_input_array(k_cast, 4);
  compute_encoder.set_input_array(v_cast, 5);
  compute_encoder.set_bytes(params, 6);

  const uint32_t work_items =
      params.layers * params.batch_size * params.num_kv_heads * params.head_dim;
  constexpr uint32_t kThreadsPerGroup = 128;
  MTL::Size grid_dims(work_items, 1, 1);
  MTL::Size group_dims(kThreadsPerGroup, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void paged_kv_write_layers_tokens(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_tokens,
    const array& v_tokens) {
  if (paged_kv_write_layers_tokens_use_fallback(
          k_cache,
          v_cache,
          block_tables,
          context_lens,
          k_tokens,
          v_tokens,
          s)) {
    throw std::runtime_error(
        "paged_kv_write_layers_tokens: configuration not supported on Metal backend.");
  }

  auto stream = to_stream(s);
  auto block_tables_i32 = mlx::core::astype(block_tables, int32, stream);
  auto context_i32 = mlx::core::astype(context_lens, int32, stream);
  auto k_cast = mlx::core::astype(k_tokens, k_cache.dtype(), stream);
  auto v_cast = mlx::core::astype(v_tokens, k_cache.dtype(), stream);

  PagedKVWriteLayersTokensParams params;
  params.layers = static_cast<uint32_t>(k_tokens.shape(0));
  params.tokens = static_cast<uint32_t>(k_tokens.shape(1));
  params.batch_size = static_cast<uint32_t>(k_tokens.shape(2));
  params.num_kv_heads = static_cast<uint32_t>(k_tokens.shape(3));
  params.head_dim = static_cast<uint32_t>(k_tokens.shape(4));
  params.block_size = static_cast<uint32_t>(k_cache.shape(3));
  params.max_blocks = static_cast<uint32_t>(k_cache.shape(2));
  params.max_blocks_per_seq = static_cast<uint32_t>(block_tables.shape(1));
  params.row_stride = params.head_dim;
  params.block_stride = params.block_size * params.row_stride;
  params.kv_head_stride = params.max_blocks * params.block_stride;
  params.layer_stride = static_cast<uint32_t>(k_cache.strides()[0]);
  params.token_layer_stride = static_cast<uint32_t>(k_tokens.strides()[0]);
  params.token_step_stride = static_cast<uint32_t>(k_tokens.strides()[1]);
  params.token_batch_stride = static_cast<uint32_t>(k_tokens.strides()[2]);
  params.token_head_stride = static_cast<uint32_t>(k_tokens.strides()[3]);
  params.token_dim_stride = static_cast<uint32_t>(k_tokens.strides()[4]);
  params.table_stride = static_cast<uint32_t>(block_tables.strides()[0]);

  auto kernel_id = std::string("paged_kv_write_layers_tokens_") +
      get_type_string(k_cache.dtype());
  metal::MTLFCList func_consts;
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_id, kernel_id, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(k_cache, 0);
  compute_encoder.set_input_array(v_cache, 1);
  compute_encoder.set_input_array(block_tables_i32, 2);
  compute_encoder.set_input_array(context_i32, 3);
  compute_encoder.set_input_array(k_cast, 4);
  compute_encoder.set_input_array(v_cast, 5);
  compute_encoder.set_bytes(params, 6);

  const uint32_t work_items = params.layers * params.tokens *
      params.batch_size * params.num_kv_heads * params.head_dim;
  constexpr uint32_t kThreadsPerGroup = 128;
  MTL::Size grid_dims(work_items, 1, 1);
  MTL::Size group_dims(kThreadsPerGroup, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
