#include "mlx/backend/metal/paged_attention.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

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

std::atomic<double> g_last_gpu_time_ms{0.0};

uint32_t read_env_u32(const char* name) {
  if (const char* value = std::getenv(name)) {
    try {
      auto parsed = std::stoul(value);
      return static_cast<uint32_t>(parsed);
    } catch (...) {
    }
  }
  return 0;
}

uint32_t clamp_threads_per_head(uint32_t tg_size, metal::Device& device) {
  auto max_threads_size = device.mtl_device()->maxThreadsPerThreadgroup();
  auto max_threads = static_cast<uint32_t>(max_threads_size.width);
  constexpr uint32_t kMaxKernelThreads = 256;
  if (tg_size == 0) {
    return 32;
  }
  tg_size = std::max<uint32_t>(32, tg_size);
  tg_size = std::min(tg_size, max_threads);
  tg_size = std::min(tg_size, kMaxKernelThreads);
  return tg_size;
}

uint32_t choose_threads_per_head(
    uint32_t head_dim,
    uint32_t batch,
    metal::Device& device) {
  if (uint32_t override = read_env_u32("MLX_PAGED_ATTN_TG_SIZE")) {
    return clamp_threads_per_head(override, device);
  }
  uint32_t guess;
  if (head_dim >= 128) {
    guess = batch >= 48 ? 128 : 96;
  } else if (head_dim >= 64) {
    guess = batch >= 32 ? 96 : 64;
  } else {
    guess = 64;
  }
  return clamp_threads_per_head(guess, device);
}

uint32_t choose_vec_width(uint32_t head_dim) {
  if (head_dim == 0) {
    return 1;
  }
  if (uint32_t override = read_env_u32("MLX_PAGED_ATTN_VEC_WIDTH")) {
    override = std::max<uint32_t>(1u, override);
    override = std::min<uint32_t>(override, 8u);
    while (override > head_dim && override > 1) {
      override >>= 1;
    }
    return std::max<uint32_t>(1u, override);
  }
  if (head_dim % 8 == 0) {
    return 8;
  }
  if (head_dim % 4 == 0) {
    return 4;
  }
  if (head_dim % 2 == 0) {
    return 2;
  }
  return 1;
}

std::string kernel_dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case float16:
      return "float16";
    case bfloat16:
      return "bfloat16";
    case float32:
      return "float32";
    default:
      throw std::invalid_argument("paged_attention: unsupported dtype");
  }
}

std::string kernel_base_name(Dtype dtype) {
  return "paged_attention_decode_" + kernel_dtype_suffix(dtype);
}

std::string kernel_specialized_name(
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head,
    uint32_t vec_width) {
  std::string name = kernel_base_name(dtype);
  name += "_bs";
  name += std::to_string(block_size);
  name += "_tg";
  name += std::to_string(threads_per_head);
  name += "_vw";
  name += std::to_string(vec_width);
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

  auto kernel_base = kernel_base_name(q.dtype());
  uint16_t block_const = static_cast<uint16_t>(params.block_size);
  uint32_t threads_per_head =
      choose_threads_per_head(params.head_dim, q.shape(0), device);
  uint16_t tg_const = static_cast<uint16_t>(threads_per_head);
  uint32_t vec_width = choose_vec_width(params.head_dim);
  uint16_t vec_const = static_cast<uint16_t>(vec_width);
  auto specialized_name = kernel_specialized_name(
      q.dtype(), params.block_size, threads_per_head, vec_width);
  metal::MTLFCList func_consts = {
      {&block_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(0)},
      {&tg_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(1)},
      {&vec_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(2)}};
  auto& compute_encoder = device.get_command_encoder(s.index);
  auto kernel = device.get_kernel(kernel_base, specialized_name, func_consts);
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
  MTL::Size group_dims(threads_per_head, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  auto command_buffer = device.get_command_buffer(s.index);
  command_buffer->addCompletedHandler([](MTL::CommandBuffer* cb) {
    double start = cb->GPUStartTime();
    double end = cb->GPUEndTime();
    double elapsed = 0.0;
    if (end > start && start > 0.0) {
      elapsed = end - start;
    } else {
      start = cb->kernelStartTime();
      end = cb->kernelEndTime();
      if (end > start && start > 0.0) {
        elapsed = end - start;
      }
    }
    if (elapsed > 0.0) {
      g_last_gpu_time_ms.store(elapsed * 1000.0, std::memory_order_relaxed);
    }
  });
}

void paged_attention_prewarm_kernel(
    metal::Device& device,
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head,
    uint32_t vec_width) {
  if (block_size == 0) {
    return;
  }
  uint16_t block_const = static_cast<uint16_t>(block_size);
  uint16_t tg_const = static_cast<uint16_t>(threads_per_head);
  uint16_t vec_const = static_cast<uint16_t>(vec_width);
  metal::MTLFCList func_consts = {
      {&block_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(0)},
      {&tg_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(1)},
      {&vec_const, MTL::DataTypeUShort, static_cast<NS::UInteger>(2)}};
  auto kernel_base = kernel_base_name(dtype);
  auto specialized_name =
      kernel_specialized_name(dtype, block_size, threads_per_head, vec_width);
  device.get_kernel(kernel_base, specialized_name, func_consts);
}

double paged_attention_last_time_ms() {
  return g_last_gpu_time_ms.load(std::memory_order_relaxed);
}

} // namespace mlx::core::fast
