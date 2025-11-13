#include "mlx/backend/metal/paged_attention.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms.h"
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
  uint32_t q_query_stride;
  uint32_t q_len;
  uint32_t kv_head_stride;
  uint32_t block_stride;
  uint32_t row_stride;
  uint32_t out_batch_stride;
  uint32_t out_head_stride;
  uint32_t out_query_stride;
  uint32_t overlay_batch_stride;
  uint32_t overlay_head_stride;
  uint32_t overlay_seq_stride;
  uint32_t overlay_len;
  uint32_t overlay_valid;
};

struct PagedAttentionQuantParams {
  uint32_t enabled;
  uint32_t bits;
  uint32_t group_size;
  uint32_t groups_per_head;
  uint32_t bytes_per_group;
  uint32_t vq_head_stride;
  uint32_t vq_block_stride;
  uint32_t vq_row_stride;
  uint32_t scale_head_stride;
  uint32_t scale_block_stride;
  uint32_t scale_row_stride;
};

namespace {

std::atomic<double> g_last_gpu_time_ms{0.0};
std::atomic<double> g_prefill_last_time_ms{0.0};
std::mutex g_trace_mutex;

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

std::string overlay_kernel_base_name(Dtype dtype) {
  return "paged_attention_decode_overlay_" + kernel_dtype_suffix(dtype);
}

std::string overlay_kernel_specialized_name(
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head,
    uint32_t vec_width) {
  std::string name = overlay_kernel_base_name(dtype);
  name += "_bs";
  name += std::to_string(block_size);
  name += "_tg";
  name += std::to_string(threads_per_head);
  name += "_vw";
  name += std::to_string(vec_width);
  return name;
}

std::string prefill_kernel_base_name(Dtype dtype) {
  return "paged_prefill_" + kernel_dtype_suffix(dtype);
}

std::string prefill_kernel_specialized_name(
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head,
    uint32_t vec_width) {
  std::string name = prefill_kernel_base_name(dtype);
  name += "_bs";
  name += std::to_string(block_size);
  name += "_tg";
  name += std::to_string(threads_per_head);
  name += "_vw";
  name += std::to_string(vec_width);
  return name;
}

bool paged_trace_enabled() {
  static bool enabled = []() {
    const char* value = std::getenv("MLX_PAGED_TRACE");
    if (value == nullptr) {
      return false;
    }
    std::string lowered(value);
    std::transform(
        lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
    return !lowered.empty() && lowered != "0" && lowered != "false" &&
        lowered != "off" && lowered != "no";
  }();
  return enabled;
}

void paged_trace_log(const std::string& message) {
  if (!paged_trace_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> guard(g_trace_mutex);
  std::cerr << "[mlx.paged] " << message << std::endl;
}

std::string describe_shape(const array& arr) {
  std::ostringstream oss;
  oss << "(";
  for (int i = 0; i < arr.ndim(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << arr.shape(i);
  }
  oss << ")";
  return oss.str();
}

std::string describe_optional_shape(const array* arr) {
  if (arr == nullptr) {
    return "None";
  }
  return describe_shape(*arr);
}

} // namespace

bool paged_attention_use_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    Stream s) {
  auto fallback = [&](const char* reason) -> bool {
    if (paged_trace_enabled()) {
      std::ostringstream oss;
      oss << "fallback reason=" << reason << " device=" << s.device
          << " q_dtype=" << q.dtype() << " k_dtype=" << k.dtype()
          << " v_dtype=" << v.dtype() << " q_shape=" << describe_shape(q)
          << " k_shape=" << describe_shape(k)
          << " v_shape=" << describe_shape(v)
          << " tables_shape=" << describe_shape(block_tables)
          << " context_shape=" << describe_shape(context_lens);
      paged_trace_log(oss.str());
    }
    return true;
  };

  if (s.device == Device::cpu) {
    return fallback("cpu_device");
  }
  if (q.dtype() != float16 && q.dtype() != float32 && q.dtype() != bfloat16) {
    return fallback("q_dtype");
  }
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
    return fallback("kv_dtype_mismatch");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return fallback("layout_or_rank");
  }
  if (block_tables.ndim() != 2 || context_lens.ndim() != 1) {
    return fallback("block_or_context_rank");
  }
  // Ensure per-token dimension contiguous
  if (k.strides().back() != 1 || v.strides().back() != 1) {
    return fallback("inner_stride");
  }
  auto head_dim = q.shape(3);
  if (head_dim == 0) {
    return fallback("head_dim_zero");
  }
  if (quant != nullptr) {
    if (quant->bits != 4 && quant->bits != 8) {
      return fallback("quant_bits");
    }
    if (quant->group_size <= 0 || quant->bytes_per_group <= 0 ||
        quant->groups_per_head <= 0) {
      return fallback("quant_group_config");
    }
    if (vq_cache == nullptr || v_scale_cache == nullptr ||
        v_zero_cache == nullptr) {
      return fallback("missing_quant_cache");
    }
    auto& vq = *vq_cache;
    auto& v_scale = *v_scale_cache;
    auto& v_zero = *v_zero_cache;
    if (vq.dtype() != uint8 || v_scale.dtype() != float16 ||
        v_zero.dtype() != float16) {
      return fallback("quant_cache_dtype");
    }
    if (vq.ndim() != 4 || v_scale.ndim() != 4 || v_zero.ndim() != 4) {
      return fallback("quant_rank");
    }
    if (vq.shape(0) != k.shape(0) || v_scale.shape(0) != k.shape(0) ||
        v_zero.shape(0) != k.shape(0)) {
      return fallback("quant_batch_mismatch");
    }
    if (vq.shape(1) != k.shape(1) || v_scale.shape(1) != k.shape(1) ||
        v_zero.shape(1) != k.shape(1)) {
      return fallback("quant_head_mismatch");
    }
    if (vq.shape(2) != k.shape(2) || v_scale.shape(2) != k.shape(2) ||
        v_zero.shape(2) != k.shape(2)) {
      return fallback("quant_block_mismatch");
    }
    if (vq.shape(3) != quant->bytes_per_group * quant->groups_per_head) {
      return fallback("quant_row_width");
    }
    if (v_scale.shape(3) != quant->groups_per_head ||
        v_zero.shape(3) != quant->groups_per_head) {
      return fallback("quant_scale_width");
    }
    if (!vq.flags().row_contiguous || !v_scale.flags().row_contiguous ||
        !v_zero.flags().row_contiguous) {
      return fallback("quant_cache_layout");
    }
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
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
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
  params.q_query_stride = static_cast<uint32_t>(q.strides()[2]);
  params.q_len = static_cast<uint32_t>(q.shape(2));
  params.kv_head_stride = static_cast<uint32_t>(k.strides()[0]);
  params.block_stride = static_cast<uint32_t>(k.strides()[1]);
  params.row_stride = static_cast<uint32_t>(k.strides()[2]);
  params.out_batch_stride = static_cast<uint32_t>(out.strides()[0]);
  params.out_head_stride = static_cast<uint32_t>(out.strides()[1]);
  params.out_query_stride = static_cast<uint32_t>(out.strides()[2]);
  params.overlay_batch_stride = 0;
  params.overlay_head_stride = 0;
  params.overlay_seq_stride = 0;
  params.overlay_len = 0;
  params.overlay_valid = 0;
  params.overlay_seq_stride = 0;
  params.overlay_len = 0;

  PagedAttentionQuantParams quant_params;
  quant_params.enabled = (quant != nullptr) ? 1u : 0u;
  if (quant != nullptr) {
    quant_params.bits = static_cast<uint32_t>(quant->bits);
    quant_params.group_size = static_cast<uint32_t>(quant->group_size);
    quant_params.groups_per_head =
        static_cast<uint32_t>(quant->groups_per_head);
    quant_params.bytes_per_group =
        static_cast<uint32_t>(quant->bytes_per_group);
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
  } else {
    std::memset(&quant_params, 0, sizeof(PagedAttentionQuantParams));
  }

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
  if (quant != nullptr) {
    compute_encoder.set_input_array(*vq_cache, 8);
    compute_encoder.set_input_array(*v_scale_cache, 9);
    compute_encoder.set_input_array(*v_zero_cache, 10);
  } else {
    compute_encoder.set_input_array(v, 8);
    compute_encoder.set_input_array(v, 9);
    compute_encoder.set_input_array(v, 10);
  }
  compute_encoder.set_bytes(quant_params, 11);

  MTL::Size grid_dims(
      static_cast<NS::UInteger>(q.shape(1)),
      static_cast<NS::UInteger>(q.shape(0)),
      1);
  MTL::Size group_dims(threads_per_head, 1, 1);
  if (paged_trace_enabled()) {
    std::ostringstream oss;
    oss << "prefill.before_dispatch grid=(" << grid_dims.width << ","
        << grid_dims.height << ") tg=" << group_dims.width;
    paged_trace_log(oss.str());
  }
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  if (paged_trace_enabled()) {
    std::ostringstream oss;
    oss << "decode.kernel dtype=" << q.dtype() << " batch=" << q.shape(0)
        << " heads=" << q.shape(1) << " q_len=" << q.shape(2)
        << " head_dim=" << params.head_dim
        << " block_size=" << params.block_size
        << " blocks_per_seq=" << params.max_blocks_per_seq
        << " threads_per_head=" << threads_per_head
        << " vec_width=" << vec_width << " quant=" << (quant != nullptr ? 1 : 0)
        << " grid=(" << grid_dims.width << "," << grid_dims.height << ")"
        << " out_stride=" << params.out_query_stride
        << " kv_stride=" << params.row_stride;
    if (kv_mapping.has_value()) {
      oss << " kv_mapping=" << describe_shape(*kv_mapping);
    }
    paged_trace_log(oss.str());
  }

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

void paged_attention_with_overlay(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    const array& k_overlay,
    const array& v_overlay,
    array& out,
    std::optional<uint32_t> overlay_len_override) {
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
  params.q_query_stride = static_cast<uint32_t>(q.strides()[2]);
  params.q_len = static_cast<uint32_t>(q.shape(2));
  params.kv_head_stride = static_cast<uint32_t>(k.strides()[0]);
  params.block_stride = static_cast<uint32_t>(k.strides()[1]);
  params.row_stride = static_cast<uint32_t>(k.strides()[2]);
  params.out_batch_stride = static_cast<uint32_t>(out.strides()[0]);
  params.out_head_stride = static_cast<uint32_t>(out.strides()[1]);
  params.out_query_stride = static_cast<uint32_t>(out.strides()[2]);
  params.overlay_batch_stride = 0;
  params.overlay_head_stride = 0;
  params.overlay_seq_stride = 0;
  params.overlay_len = 0;
  params.overlay_valid = overlay_len_override.value_or(0);
  if (k_overlay.ndim() >= 4) {
    params.overlay_len = static_cast<uint32_t>(k_overlay.shape(0));
    params.overlay_seq_stride = static_cast<uint32_t>(k_overlay.strides()[0]);
    params.overlay_batch_stride = static_cast<uint32_t>(k_overlay.strides()[1]);
    params.overlay_head_stride = static_cast<uint32_t>(k_overlay.strides()[2]);
  } else if (k_overlay.ndim() >= 3) {
    params.overlay_len = 1;
    params.overlay_batch_stride = static_cast<uint32_t>(k_overlay.strides()[0]);
    params.overlay_head_stride = static_cast<uint32_t>(k_overlay.strides()[1]);
  }
  if (params.overlay_valid > params.overlay_len && params.overlay_len > 0) {
    params.overlay_valid = params.overlay_len;
  }

  PagedAttentionQuantParams quant_params;
  quant_params.enabled = (quant != nullptr) ? 1u : 0u;
  if (quant != nullptr) {
    quant_params.bits = static_cast<uint32_t>(quant->bits);
    quant_params.group_size = static_cast<uint32_t>(quant->group_size);
    quant_params.groups_per_head =
        static_cast<uint32_t>(quant->groups_per_head);
    quant_params.bytes_per_group =
        static_cast<uint32_t>(quant->bytes_per_group);
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
  } else {
    std::memset(&quant_params, 0, sizeof(PagedAttentionQuantParams));
  }

  auto kernel_base = overlay_kernel_base_name(q.dtype());
  uint16_t block_const = static_cast<uint16_t>(params.block_size);
  uint32_t threads_per_head =
      choose_threads_per_head(params.head_dim, q.shape(0), device);
  uint16_t tg_const = static_cast<uint16_t>(threads_per_head);
  uint32_t vec_width = choose_vec_width(params.head_dim);
  uint16_t vec_const = static_cast<uint16_t>(vec_width);
  auto specialized_name = overlay_kernel_specialized_name(
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
  if (quant != nullptr) {
    compute_encoder.set_input_array(*vq_cache, 8);
    compute_encoder.set_input_array(*v_scale_cache, 9);
    compute_encoder.set_input_array(*v_zero_cache, 10);
  } else {
    compute_encoder.set_input_array(v, 8);
    compute_encoder.set_input_array(v, 9);
    compute_encoder.set_input_array(v, 10);
  }
  compute_encoder.set_bytes(quant_params, 11);
  compute_encoder.set_input_array(k_overlay, 12);
  compute_encoder.set_input_array(v_overlay, 13);

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

void paged_prefill(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& base_lens,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    array& out) {
  if (paged_trace_enabled()) {
    std::ostringstream oss;
    oss << "prefill.launch batch=" << q.shape(0) << " heads=" << q.shape(1)
        << " q_len=" << q.shape(2) << " head_dim=" << q.shape(3);
    paged_trace_log(oss.str());
  }
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
  params.q_query_stride = static_cast<uint32_t>(q.strides()[2]);
  params.q_len = static_cast<uint32_t>(q.shape(2));
  params.kv_head_stride = static_cast<uint32_t>(k.strides()[0]);
  params.block_stride = static_cast<uint32_t>(k.strides()[1]);
  params.row_stride = static_cast<uint32_t>(k.strides()[2]);
  params.out_batch_stride = static_cast<uint32_t>(out.strides()[0]);
  params.out_head_stride = static_cast<uint32_t>(out.strides()[1]);
  params.out_query_stride = static_cast<uint32_t>(out.strides()[2]);
  params.overlay_batch_stride = 0;
  params.overlay_head_stride = 0;

  PagedAttentionQuantParams quant_params;
  quant_params.enabled = (quant != nullptr) ? 1u : 0u;
  if (quant != nullptr) {
    quant_params.bits = static_cast<uint32_t>(quant->bits);
    quant_params.group_size = static_cast<uint32_t>(quant->group_size);
    quant_params.groups_per_head =
        static_cast<uint32_t>(quant->groups_per_head);
    quant_params.bytes_per_group =
        static_cast<uint32_t>(quant->bytes_per_group);
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
  } else {
    std::memset(&quant_params, 0, sizeof(PagedAttentionQuantParams));
  }

  auto kernel_base = prefill_kernel_base_name(q.dtype());
  uint16_t block_const = static_cast<uint16_t>(params.block_size);
  uint32_t threads_per_head =
      choose_threads_per_head(params.head_dim, q.shape(0), device);
  uint16_t tg_const = static_cast<uint16_t>(threads_per_head);
  uint32_t vec_width = choose_vec_width(params.head_dim);
  uint16_t vec_const = static_cast<uint16_t>(vec_width);
  auto specialized_name = prefill_kernel_specialized_name(
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
  compute_encoder.set_input_array(base_lens, 5);
  if (kv_mapping.has_value()) {
    compute_encoder.set_input_array(*kv_mapping, 6);
  } else {
    compute_encoder.set_input_array(context_lens, 6);
  }
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(params, 8);
  if (quant != nullptr) {
    compute_encoder.set_input_array(*vq_cache, 9);
    compute_encoder.set_input_array(*v_scale_cache, 10);
    compute_encoder.set_input_array(*v_zero_cache, 11);
  } else {
    compute_encoder.set_input_array(v, 9);
    compute_encoder.set_input_array(v, 10);
    compute_encoder.set_input_array(v, 11);
  }
  compute_encoder.set_bytes(quant_params, 12);

  MTL::Size grid_dims(
      static_cast<NS::UInteger>(q.shape(1)),
      static_cast<NS::UInteger>(q.shape(0)),
      1);
  MTL::Size group_dims(threads_per_head, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  if (paged_trace_enabled()) {
    std::ostringstream oss;
    oss << "prefill.kernel dtype=" << q.dtype() << " batch=" << q.shape(0)
        << " heads=" << q.shape(1) << " q_len=" << q.shape(2)
        << " head_dim=" << params.head_dim
        << " block_size=" << params.block_size
        << " blocks_per_seq=" << params.max_blocks_per_seq
        << " threads_per_head=" << threads_per_head
        << " vec_width=" << vec_width << " quant=" << (quant != nullptr ? 1 : 0)
        << " base_lens_shape=" << describe_shape(base_lens)
        << " context_shape=" << describe_shape(context_lens) << " grid=("
        << grid_dims.width << "," << grid_dims.height << ")";
    if (kv_mapping.has_value()) {
      oss << " kv_mapping=" << describe_shape(*kv_mapping);
    }
    paged_trace_log(oss.str());
  }

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
      g_prefill_last_time_ms.store(elapsed * 1000.0, std::memory_order_relaxed);
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

void paged_prefill_prewarm_kernel(
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
  auto kernel_base = prefill_kernel_base_name(dtype);
  auto specialized_name = prefill_kernel_specialized_name(
      dtype, block_size, threads_per_head, vec_width);
  device.get_kernel(kernel_base, specialized_name, func_consts);
}

double paged_prefill_last_time_ms() {
  return g_prefill_last_time_ms.load(std::memory_order_relaxed);
}

namespace {

inline const array* get_optional_input(
    const std::vector<array>& inputs,
    int index) {
  if (index < 0) {
    return nullptr;
  }
  if (index >= static_cast<int>(inputs.size())) {
    throw std::out_of_range(
        "PagedAttentionPrimitive input index out of range.");
  }
  return &inputs[index];
}

} // namespace

void PagedAttentionPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  if (inputs.size() < 5) {
    throw std::invalid_argument(
        "PagedAttentionPrimitive requires at least five inputs.");
  }
  auto& q_cast = inputs[0];
  auto& k_cast = inputs[1];
  auto& v_cast = inputs[2];
  auto& tables_i32 = inputs[3];
  auto& lens_i32 = inputs[4];
  const array* mapping = get_optional_input(inputs, mapping_index_);
  const array* vq_view = get_optional_input(inputs, vq_index_);
  const array* v_scale_view = get_optional_input(inputs, v_scale_index_);
  const array* v_zero_view = get_optional_input(inputs, v_zero_index_);
  const array* overlay_k = get_optional_input(inputs, overlay_k_index_);
  const array* overlay_v = get_optional_input(inputs, overlay_v_index_);
  const array* overlay_len_arr = get_optional_input(inputs, overlay_len_index_);

  if ((overlay_k == nullptr) != (overlay_v == nullptr)) {
    throw std::invalid_argument(
        "PagedAttentionPrimitive requires both overlay tensors.");
  }
  if (has_quant_) {
    if (vq_view == nullptr || v_scale_view == nullptr ||
        v_zero_view == nullptr) {
      throw std::invalid_argument(
          "PagedAttentionPrimitive quantization inputs missing.");
    }
  }

  if (mapping) {
    mlx::core::eval(*mapping);
  }
  if (has_quant_) {
    mlx::core::eval(*vq_view, *v_scale_view, *v_zero_view);
  }
  if (overlay_k) {
    mlx::core::eval(*overlay_k, *overlay_v);
  }
  mlx::core::eval(q_cast, k_cast, v_cast, tables_i32, lens_i32);

  out.set_data(allocator::malloc(out.nbytes()));

  auto stream = out.primitive().stream();
  auto& device = metal::device(stream.device);

  std::optional<array> mapping_opt;
  if (mapping) {
    mapping_opt = *mapping;
  }
  const PagedAttentionQuantConfig* quant_ptr =
      has_quant_ ? &quant_cfg_ : nullptr;

  uint32_t overlay_override = 0;
  if (overlay_len_arr) {
    mlx::core::eval(*overlay_len_arr);
    overlay_override = static_cast<uint32_t>(overlay_len_arr->item<int64_t>());
  }
  if (overlay_k) {
    paged_attention_with_overlay(
        stream,
        device,
        q_cast,
        k_cast,
        v_cast,
        tables_i32,
        lens_i32,
        mapping_opt,
        scale_,
        vq_view,
        v_scale_view,
        v_zero_view,
        quant_ptr,
        *overlay_k,
        *overlay_v,
        out,
        overlay_override > 0 ? std::optional<uint32_t>(overlay_override)
                             : std::nullopt);
  } else {
    paged_attention(
        stream,
        device,
        q_cast,
        k_cast,
        v_cast,
        tables_i32,
        lens_i32,
        mapping_opt,
        scale_,
        vq_view,
        v_scale_view,
        v_zero_view,
        quant_ptr,
        out);
  }
  out.set_status(array::Status::available);
}

} // namespace mlx::core::fast
