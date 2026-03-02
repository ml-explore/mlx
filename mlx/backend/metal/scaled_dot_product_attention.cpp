// Copyright © 2024 Apple Inc.
#include <cassert>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

void sdpa_full_self_attention_nax(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 64;
  int bk = 32;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = d.get_command_encoder(s.index);

  auto kernel = get_steel_attention_nax_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  // NAX doesn't support logsumexp output - provide dummy strides
  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
      /* int64_t LSE_strides[2] = */ {0, 0}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks,
    bool output_logsumexp_ = false,
    array* lse_out = nullptr) {
  // NAX path does not support logsumexp output - skip when VJP needs it
  if (metal::is_nax_available() && q.shape(3) != 80 &&
      (env::enable_tf32() || q.dtype() != float32) && !output_logsumexp_) {
    return sdpa_full_self_attention_nax(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& q = */ q,
        /* const array& k = */ k,
        /* const array& v = */ v,
        /* const float scale = */ scale,
        /* array& o = */ o,
        /* bool do_causal_ = */ do_causal_,
        /* const std::optional<array>& mask = */ mask,
        /* const std::optional<array>& sinks = */ sinks);
  }

  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 32;
  int bk = bd < 128 ? 32 : 16;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();
  const bool output_logsumexp = output_logsumexp_;

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302},
      {&output_logsumexp, MTL::DataType::DataTypeBool, 303}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'),
      "_lse_",
      (output_logsumexp ? 't' : 'n'));

  auto& compute_encoder = d.get_command_encoder(s.index);

  auto kernel = get_steel_attention_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  // Compute LSE strides if outputting logsumexp: shape [B, H, qL, 1]
  // The VJP kernel expects strides as:
  //   LSE_strides[0] = qL (stride between heads within same batch)
  //   LSE_strides[1] = 1 (stride between query positions)
  // Linear index = (batch * H + head) * qL + query_pos
  int64_t lse_str_head = qL; // Stride between heads
  int64_t lse_str_qpos = 1; // Stride between query positions

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
      /* int64_t LSE_strides[2] = */ {lse_str_head, lse_str_qpos}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }
  if (output_logsumexp && lse_out != nullptr) {
    compute_encoder.set_output_array(*lse_out, 8);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);

  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 16);
    compute_encoder.set_bytes(q.shape(1), 17);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int n_simds = gqa_factor * q.shape(2);

  char devc = d.get_architecture().back();
  int N = k.shape(2);
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    if (n_simds >= 4) {
      blocks = 64;
    } else {
      blocks = 32;
    }
  }
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];
  MTL::Size group_dims(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);

  // Allocate the intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  d.add_temporary(intermediate, s.index);
  d.add_temporary(sums, s.index);
  d.add_temporary(maxs, s.index);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 18);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Final pass
  kname.clear();
  kname = "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Get the kernel
  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  // Launch
  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp,
    Stream s) {
  // Note: When output_logsumexp is true, the caller (fast.cpp) has already
  // verified VJP availability with proper has_mask/has_sinks parameters.
  // No redundant check needed here.
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = k.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256);
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);

  const bool sdpa_full_supported_mask = !has_mask || has_arr_mask ||
      (query_sequence_length <= key_sequence_length && do_causal);

  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim &&
      (query_sequence_length * gqa_factor) <= 32;

  return !(supports_sdpa_full || supports_sdpa_vector);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = outputs[0];

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(inputs.size());
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // We are in vector mode ie single query
  // NOTE: Vector mode doesn't support logsumexp output needed for VJP.
  // When output_logsumexp_ is true (training mode), use full attention instead.
  if (q_pre.shape(2) <= 8 && !output_logsumexp_) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    bool q_copied = !q_copy_unless(q_pre);
    array q = (q_copied) ? contiguous_copy_gpu(q_pre, s) : q_pre;
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      if (q_copied) {
        copies.push_back(q);
      }
      o.set_data(allocator::malloc(o.nbytes()));
    }

    auto mask_copy_unless = [&q](const array& arr) {
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
          (strides[0] == strides[1] * shape[1]);
    };

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    }
  }

  // Full attention mode
  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    // Handle logsumexp output for VJP backward pass
    array* lse_out = nullptr;
    if (output_logsumexp_ && outputs.size() > 1) {
      auto& lse = outputs[1];
      lse.set_data(allocator::malloc(lse.nbytes()));
      lse_out = &outputs[1];
    }

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
        : std::nullopt;

    sdpa_full_self_attention_metal(
        s,
        d,
        q,
        k,
        v,
        scale_,
        o,
        do_causal_,
        mask,
        sinks,
        output_logsumexp_,
        lse_out);
  }

  d.add_temporaries(std::move(copies), s.index);
}

bool ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s,
    bool has_mask,
    bool has_sinks,
    int n_kv_heads) {
  // Use fallback on CPU
  if (s.device == Device::cpu) {
    return true;
  }

  const int query_head_dim = q.shape(-1);
  const int query_seq_len = q.shape(2);

  // Vector VJP uses exp2() matching forward pass's log2 domain.
  // Note: D=256 exceeds Metal's 32KB threadgroup memory limit for vector VJP.
  // Note: The accumulate variant (used for half/bfloat16) does NOT support
  // sinks.
  const bool vector_supported_head_dim =
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128);
  const bool is_float32 = (q.dtype() == float32);

  // For short sequences (seq <= 8), prefer vector VJP if head dim is supported.
  // However, sinks are only supported in the float32 vector VJP kernel,
  // not in the accumulate variant used for half/bfloat16.
  if (query_seq_len <= 8 && vector_supported_head_dim) {
    // If sinks are present and dtype is not float32, must use fallback
    // because sdpa_vector_vjp_accumulate doesn't support sinks.
    if (has_sinks && !is_float32) {
      return true; // Must use unfused attention for sinks with half/bfloat16
    }
    return false; // Use vector VJP
  }

  // For longer sequences (L > 8), use fallback (unfused attention)
  // STEEL VJP for longer sequences will be added in a future PR
  return true;
}

namespace {

void sdpa_vector_vjp_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& out,
    const array& d_out,
    const array& logsumexp,
    array& d_q,
    array& d_k,
    array& d_v,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name (matching forward pattern)
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_vjp_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes (same as forward)
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // Vector VJP kernel uses input strides for output pointer arithmetic.
  // Verify output strides match input strides to prevent memory corruption.
  // Stride requirements:
  //   d_k head stride must match k head stride
  //   d_k seq stride must match k seq stride
  //   d_v head stride must match v head stride
  //   d_v seq stride must match v seq stride
  size_t d_k_head_stride = d_k.shape(1) == 1 ? d_k.strides(0) : d_k.strides(1);
  size_t d_v_head_stride = d_v.shape(1) == 1 ? d_v.strides(0) : d_v.strides(1);
  if (d_k_head_stride != k_head_stride || d_k.strides()[2] != k_seq_stride ||
      d_v_head_stride != v_head_stride || d_v.strides()[2] != v_seq_stride) {
    throw std::runtime_error(
        "Stride mismatch in vector VJP kernel: "
        "output array strides must match input array strides. "
        "This may occur with non-contiguous array views.");
  }

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks_flag = sinks.has_value();

  // Function constants (same indices as forward)
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks_flag, MTL::DataType::DataTypeBool, 25},
  };

  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks_flag ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set kernel arguments
  // Inputs: Q, K, V, O, dO, logsumexp
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);

  // Outputs: dQ, dK, dV
  compute_encoder.set_output_array(d_q, 6);
  compute_encoder.set_output_array(d_k, 7);
  compute_encoder.set_output_array(d_v, 8);

  // Parameters
  compute_encoder.set_bytes(gqa_factor, 9);
  compute_encoder.set_bytes(N, 10);
  compute_encoder.set_bytes(k_head_stride, 11);
  compute_encoder.set_bytes(k_seq_stride, 12);
  compute_encoder.set_bytes(v_head_stride, 13);
  compute_encoder.set_bytes(v_seq_stride, 14);
  compute_encoder.set_bytes(scale, 15);

  // Output (O/dO) stride parameters - handle BLHV physical layout from STEEL
  // For BLHV layout: strides are [L*H*V, V, H*V, 1] vs logical [B, H, L, V]
  int num_q_heads = q.shape(1);
  size_t o_batch_stride = out.strides(0);
  size_t o_head_stride = out.shape(1) == 1 ? 0 : out.strides(1);
  size_t o_seq_stride = out.strides(2);
  compute_encoder.set_bytes(num_q_heads, 16);
  compute_encoder.set_bytes(o_batch_stride, 17);
  compute_encoder.set_bytes(o_head_stride, 18);
  compute_encoder.set_bytes(o_seq_stride, 19);

  // Optional mask inputs (buffer indices shifted by 4)
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(
        m, 20 + float_mask); // 20 for bool, 21 for float
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 22);
    compute_encoder.set_bytes(q_seq_stride, 23);
    compute_encoder.set_bytes(head_stride, 24);
  }

  // Optional sinks (buffer index shifted by 4)
  if (has_sinks_flag) {
    compute_encoder.set_input_array(*sinks, 25);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Dispatch function for vector VJP with float32 accumulators (for
// half/bfloat16) This variant uses the sdpa_vector_vjp_accumulate kernel which
// has device float* signature for dK and dV buffers, allowing correct pointer
// arithmetic for atomic float operations.
void sdpa_vector_vjp_accumulate_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& out,
    const array& d_out,
    const array& logsumexp,
    array& d_q,
    array& d_k_accum, // float32 accumulator buffer
    array& d_v_accum, // float32 accumulator buffer
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name (uses accumulate variant)
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_vjp_accumulate_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int Q_seq = q.shape(2); // Number of query positions
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // Vector VJP kernel uses input strides for output pointer arithmetic.
  // Verify accumulator buffer strides match input strides to prevent memory
  // corruption. Stride requirements:
  //   d_k_accum head stride must match k head stride
  //   d_k_accum seq stride must match k seq stride
  //   d_v_accum head stride must match v head stride
  //   d_v_accum seq stride must match v seq stride
  size_t d_k_head_stride =
      d_k_accum.shape(1) == 1 ? d_k_accum.strides(0) : d_k_accum.strides(1);
  size_t d_v_head_stride =
      d_v_accum.shape(1) == 1 ? d_v_accum.strides(0) : d_v_accum.strides(1);
  if (d_k_head_stride != k_head_stride ||
      d_k_accum.strides()[2] != k_seq_stride ||
      d_v_head_stride != v_head_stride ||
      d_v_accum.strides()[2] != v_seq_stride) {
    throw std::runtime_error(
        "Stride mismatch in vector VJP kernel: "
        "output array strides must match input array strides. "
        "This may occur with non-contiguous array views.");
  }

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks_flag = sinks.has_value();

  // Function constants (same indices as forward)
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks_flag, MTL::DataType::DataTypeBool, 25},
  };

  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks_flag ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set kernel arguments (accumulate variant has slightly different buffer
  // layout) Inputs: Q, K, V, O, dO, logsumexp
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);

  // Outputs: dQ, dK_accum (float32), dV_accum (float32)
  compute_encoder.set_output_array(d_q, 6);
  compute_encoder.set_output_array(d_k_accum, 7);
  compute_encoder.set_output_array(d_v_accum, 8);

  // Parameters (note: buffer indices shifted from regular VJP kernel)
  compute_encoder.set_bytes(gqa_factor, 9);
  compute_encoder.set_bytes(N, 10);
  compute_encoder.set_bytes(
      Q_seq, 11); // Extra parameter for accumulate variant
  compute_encoder.set_bytes(k_head_stride, 12);
  compute_encoder.set_bytes(k_seq_stride, 13);
  compute_encoder.set_bytes(v_head_stride, 14);
  compute_encoder.set_bytes(v_seq_stride, 15);
  compute_encoder.set_bytes(scale, 16);

  // Output (O/dO) stride parameters - handle BLHV physical layout from STEEL
  int num_q_heads = q.shape(1);
  size_t o_batch_stride = out.strides(0);
  size_t o_head_stride = out.shape(1) == 1 ? 0 : out.strides(1);
  size_t o_seq_stride = out.strides(2);
  compute_encoder.set_bytes(num_q_heads, 17);
  compute_encoder.set_bytes(o_batch_stride, 18);
  compute_encoder.set_bytes(o_head_stride, 19);
  compute_encoder.set_bytes(o_seq_stride, 20);

  // Optional mask inputs (buffer indices shifted by 4)
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(
        m, 21 + float_mask); // 21 for bool, 22 for float
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 23);
    compute_encoder.set_bytes(q_seq_stride, 24);
    compute_encoder.set_bytes(head_stride, 25);
  }

  // Note: sinks not supported in accumulate variant.
  // use_fallback() returns true for sinks with non-float32 dtypes,
  // so this code path should never be reached with has_sinks_flag=true.

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Parse inputs:
  // inputs = [Q, K, V, (optional mask), (optional sinks), O, logsumexp, dO]
  // The last 3 are always O, logsumexp, dO
  const auto& q_pre = inputs[0];
  const auto& k_pre = inputs[1];
  const auto& v_pre = inputs[2];

  // Determine indices based on optional inputs
  // primals can have mask and/or sinks appended
  size_t num_primals = inputs.size() - 3; // Subtract O, logsumexp, dO
  const auto& out = inputs[num_primals];
  const auto& logsumexp = inputs[num_primals + 1];
  const auto& d_out = inputs[num_primals + 2];

  auto& d_q = outputs[0];
  auto& d_k = outputs[1];
  auto& d_v = outputs[2];

  std::vector<array> copies;
  copies.reserve(inputs.size());

  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  // Handle optional sinks
  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs[num_primals - 1]);
  }

  // Determine if we have a mask
  bool has_arr_mask = num_primals > (3 + has_sinks_);

  // Determine early whether to use vector VJP (needed for K/V copy decisions)
  // Vector VJP uses input K/V strides for output dK/dV pointer arithmetic,
  // so K/V must be row-contiguous when using vector VJP.
  // Note: D=256 exceeds Metal's 32KB threadgroup memory limit for vector VJP.
  const int query_head_dim_pre = q_pre.shape(-1);
  const int value_head_dim_pre = v_pre.shape(-1);
  const bool vector_supported_head_dim =
      query_head_dim_pre == value_head_dim_pre &&
      (query_head_dim_pre == 64 || query_head_dim_pre == 96 ||
       query_head_dim_pre == 128);
  bool use_vector_vjp = (q_pre.shape(2) <= 8) && vector_supported_head_dim;

  // Copy predicates for Q (same as forward)
  auto q_copy_unless = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return true;
    }
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (shape[0] == 1 || shape[1] == 1) {
      auto bidx = shape[0] == 1 ? 1 : 0;
      return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
          (strides[bidx] == shape[3]);
    }
    return false;
  };

  auto kv_copy_unless = [](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (strides.back() != 1) {
      return false;
    }
    if (shape[0] == 1 || shape[1] == 1) {
      return true;
    }
    return (strides[0] == strides[1] * shape[1]);
  };

  auto is_row_contiguous = [](const array& arr) {
    return arr.flags().row_contiguous;
  };

  const auto& q = copy_unless(q_copy_unless, q_pre);
  // Vector VJP requires row-contiguous K/V because the kernel uses input
  // strides for output pointer arithmetic, and dK/dV are always contiguous.
  const auto& k = use_vector_vjp ? copy_unless(is_row_contiguous, k_pre)
                                 : copy_unless(kv_copy_unless, k_pre);
  const auto& v = use_vector_vjp ? copy_unless(is_row_contiguous, v_pre)
                                 : copy_unless(kv_copy_unless, v_pre);
  const auto& o = copy_unless(is_matrix_contiguous, out);
  const auto& dO = copy_unless(is_matrix_contiguous, d_out);
  const auto& lse = copy_unless(is_matrix_contiguous, logsumexp);

  // Allocate output gradient arrays
  // The vector VJP kernel uses atomic adds to accumulate dK and dV,
  // so we must zero-initialize these arrays.
  const int query_head_dim = q.shape(-1);
  const int value_head_dim = v.shape(-1);

  d_q.set_data(allocator::malloc(d_q.nbytes()));

  // CRITICAL FIX: The vector VJP kernel uses mlx_atomic<float> for dK/dV
  // accumulation. This works correctly ONLY when the output dtype is float32.
  // For half/bfloat16, reinterpret_cast<device mlx_atomic<float>*>(d_keys)
  // causes memory corruption because half is 2 bytes but float is 4 bytes.
  //
  // Solution: For non-float32 dtypes with vector VJP, we:
  // 1. Allocate float32 temporary accumulators for dK and dV
  // 2. Run the kernel with these float32 buffers
  // 3. Copy/convert from float32 to the original dtype after kernel completion
  bool needs_float32_accumulators = use_vector_vjp && (q.dtype() != float32);
  std::optional<array> dk_accum = std::nullopt;
  std::optional<array> dv_accum = std::nullopt;

  if (use_vector_vjp) {
    if (needs_float32_accumulators) {
      // Allocate float32 accumulator buffers with same shape as dK/dV
      // Note: zeros() creates lazy arrays with null data pointer.
      // We must explicitly allocate and zero-initialize for GPU kernel use.
      size_t dk_bytes = d_k.size() * sizeof(float);
      size_t dv_bytes = d_v.size() * sizeof(float);
      dk_accum = array(allocator::malloc(dk_bytes), d_k.shape(), float32);
      dv_accum = array(allocator::malloc(dv_bytes), d_v.shape(), float32);

      // Zero-initialize the accumulator buffers
      array zero_f32 = array(0.0f, float32);
      fill_gpu(zero_f32, dk_accum.value(), s);
      fill_gpu(zero_f32, dv_accum.value(), s);
      copies.push_back(std::move(zero_f32));

      // Allocate the actual output arrays (will be written after kernel)
      d_k.set_data(allocator::malloc(d_k.nbytes()));
      d_v.set_data(allocator::malloc(d_v.nbytes()));
    } else {
      // No float32 accumulators needed: zero-initialize dK/dV directly
      // Must allocate memory before fill_gpu
      d_k.set_data(allocator::malloc(d_k.nbytes()));
      d_v.set_data(allocator::malloc(d_v.nbytes()));
      array zero = array(0.0f, d_k.dtype());
      fill_gpu(zero, d_k, s);
      fill_gpu(zero, d_v, s);
      copies.push_back(std::move(zero));
    }
  }

  // Handle mask
  auto mask_copy_unless = [&q](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
        (strides[0] == strides[1] * shape[1]);
  };

  std::optional<array> mask = std::nullopt;
  if (has_arr_mask) {
    mask = copy_unless(mask_copy_unless, inputs[3]);
  }

  bool do_causal = do_causal_ && q.shape(2) > 1;

  // Dispatch to appropriate kernel based on sequence length
  if (use_vector_vjp) {
    if (needs_float32_accumulators) {
      // Use float32 accumulator buffers with the accumulate kernel variant
      // This variant has device float* signature for dK/dV, ensuring correct
      // pointer arithmetic (sizeof(float)=4) instead of sizeof(T)=2 for
      // half/bfloat16
      array& dk_acc = dk_accum.value();
      array& dv_acc = dv_accum.value();
      sdpa_vector_vjp_accumulate_dispatch(
          s,
          d,
          q,
          k,
          v,
          o,
          dO,
          lse,
          d_q,
          dk_acc,
          dv_acc,
          scale_,
          do_causal,
          mask,
          sinks);

      // Convert float32 accumulators to original dtype
      // This uses the standard copy primitive with type conversion
      copy_gpu(dk_acc, d_k, CopyType::General, s);
      copy_gpu(dv_acc, d_v, CopyType::General, s);

      // Add accumulators as temporaries for cleanup
      d.add_temporary(dk_acc, s.index);
      d.add_temporary(dv_acc, s.index);
    } else {
      // Float32: pass dK/dV directly (already zero-initialized above)
      sdpa_vector_vjp_dispatch(
          s,
          d,
          q,
          k,
          v,
          o,
          dO,
          lse,
          d_q,
          d_k,
          d_v,
          scale_,
          do_causal,
          mask,
          sinks);
    }
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
