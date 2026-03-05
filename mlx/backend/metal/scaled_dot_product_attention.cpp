// Copyright © 2024 Apple Inc.
#include <cassert>
#include <cstdlib>
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

// Copy predicates shared between forward and VJP eval_gpu methods.

// Returns true if the array's last dimension has stride 1.
bool is_matrix_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

// Returns true if Q doesn't need a contiguous copy for the vector path.
// Allows row-contiguous or transposed layouts where batch/head dims are
// interchangeable with sequence when one is a singleton.
bool q_is_vector_compatible(const array& arr) {
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
}

// Returns true if K/V doesn't need a contiguous copy for the vector path.
// Requires last dim stride=1 and contiguous batch/head dimensions.
bool kv_is_vector_compatible(const array& arr) {
  auto& strides = arr.strides();
  auto& shape = arr.shape();
  if (strides.back() != 1) {
    return false;
  }
  if (shape[0] == 1 || shape[1] == 1) {
    return true;
  }
  return (strides[0] == strides[1] * shape[1]);
}

// Returns true if mask doesn't need a contiguous copy.
// Checks row-contiguity or batch/head dimension compatibility.
bool mask_is_compatible(const array& q, const array& arr) {
  auto& strides = arr.strides();
  auto& shape = arr.shape();
  return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
      (strides[0] == strides[1] * shape[1]);
}

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
    const std::optional<array>& sinks,
    array* lse_out = nullptr) {
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
  bool do_output_lse = (lse_out != nullptr);
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&do_output_lse, MTL::DataType::DataTypeBool, 28},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";
  hash_name += do_output_lse ? "_lse" : "_nolse";

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
  if (do_output_lse) {
    compute_encoder.set_output_array(*lse_out, 18);
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
    const std::optional<array>& sinks,
    array* lse_out = nullptr) {
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

  bool do_output_lse = (lse_out != nullptr);
  metal::MTLFCList pass2_func_consts = {
      {&do_output_lse, MTL::DataType::DataTypeBool, 28},
  };
  std::string pass2_hash_name = kname;
  pass2_hash_name += do_output_lse ? "_lse" : "_nolse";

  // Get the kernel
  kernel = d.get_kernel(kname, pass2_hash_name, pass2_func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);
  if (do_output_lse) {
    compute_encoder.set_output_array(*lse_out, 5);
  }

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

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // We are in vector mode ie single query
  if (q_pre.shape(2) <= 8) {
    bool q_copied = !q_is_vector_compatible(q_pre);
    array q = (q_copied) ? contiguous_copy_gpu(q_pre, s) : q_pre;
    const auto& k = copy_unless(kv_is_vector_compatible, k_pre);
    const auto& v = copy_unless(kv_is_vector_compatible, v_pre);


    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      if (q_copied) {
        copies.push_back(q);
      }
      o.set_data(allocator::malloc(o.nbytes()));
    }

    // Handle logsumexp output for VJP backward pass
    array* lse_out = nullptr;
    if (output_logsumexp_ && outputs.size() > 1) {
      auto& lse = outputs[1];
      lse.set_data(allocator::malloc(lse.nbytes()));
      lse_out = &outputs[1];
    }

    auto mask_pred = [&q](const array& arr) {
      return mask_is_compatible(q, arr);
    };
    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(mask_pred, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(
          s, d, q, k, v, o, scale_, do_causal, mask, sinks, lse_out);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks, lse_out);
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

  // Vector VJP supports D=64,96,128,256
  // D=256 uses two-stage tiling (128-wide passes) to fit in 32KB threadgroup memory.
  const bool vector_supported_head_dim =
      (query_head_dim == 64 || query_head_dim == 96 ||
       query_head_dim == 128 || query_head_dim == 256);

  // For short sequences (seq <= 8), prefer vector VJP if head dim is supported.
  // Sinks are not yet implemented in the vector VJP kernel, so fall back
  // to unfused attention when sinks are present.
  if (query_seq_len <= 8 && vector_supported_head_dim) {
    if (has_sinks) {
      return true; // Sinks not supported in vector VJP kernel
    }
    return false; // Use vector VJP
  }

  // STEEL VJP dispatch policy.
  //
  // On Apple Silicon with NAX-optimized matmul kernels (macOS 26.2+),
  // unfused backward is faster at all sequence lengths for typical L
  // (BQ=32 tiles: 1.9 TFLOPS vs NAX large tiles: 10.7 TFLOPS).
  // However, fused VJP avoids materializing the O(L^2) attention matrix,
  // providing 84-96% memory savings at L>=1024. This matters for long
  // sequences where the attention matrix would exceed available memory.
  //
  // See: Draw Things' Metal FlashAttention discussion for Apple Silicon
  // backward pass tradeoffs and Metal-specific tiling constraints.
  //
  // Policy controlled by MLX_SDPA_VJP_MODE env var:
  //   "auto"    (default) - use fused when L or memory pressure is high
  //   "unfused" - always use unfused backward (fastest on Apple Silicon)
  //   "fused"   - always use fused backward (memory-efficient)
  //
  // Auto mode triggers:
  //   - L >= MLX_SDPA_VJP_LONG_L_THRESHOLD (default 8192), OR
  //   - estimated attention matrix >= MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD
  //     (default 1073741824 = 1 GB)

  const bool steel_supported_head_dim = (query_head_dim == 64);
  const bool steel_supported_dtype =
      (q.dtype() == float16 || q.dtype() == bfloat16);
  const bool steel_eligible =
      steel_supported_head_dim && steel_supported_dtype && !has_mask && !has_sinks;

  if (!steel_eligible) {
    return true; // Not eligible for fused VJP, use unfused
  }

  // Read dispatch policy from environment
  const char* mode_env = std::getenv("MLX_SDPA_VJP_MODE");
  std::string mode = mode_env ? mode_env : "auto";

  if (mode == "unfused") {
    return true;
  }
  if (mode == "fused") {
    return false; // Force fused VJP
  }

  // Auto mode: use fused when sequence length or memory pressure is high.
  // This avoids materializing the O(L^2) attention matrix for backward.

  // Check 1: sequence length threshold
  const char* thresh_env = std::getenv("MLX_SDPA_VJP_LONG_L_THRESHOLD");
  int l_threshold = thresh_env ? std::atoi(thresh_env) : 8192;

  if (query_seq_len >= l_threshold) {
    return false; // Use fused VJP for long sequences
  }

  // Check 2: estimated attention matrix size.
  // Unfused backward materializes [B, n_q_heads, qL, kL] as an intermediate.
  // dtype_size: float16/bfloat16 = 2 bytes.
  const int B = q.shape(0);
  const int n_q_heads = q.shape(1);
  const size_t dtype_size = q.dtype() == float16 ? 2 : 2; // both fp16/bf16
  const size_t attn_bytes =
      static_cast<size_t>(B) * n_q_heads * query_seq_len * query_seq_len * dtype_size;

  const char* bytes_env = std::getenv("MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD");
  // Default: 1 GB (1073741824 bytes)
  const size_t bytes_threshold = bytes_env
      ? static_cast<size_t>(std::atoll(bytes_env))
      : static_cast<size_t>(1) << 30;

  if (attn_bytes >= bytes_threshold) {
    return false; // Use fused VJP to avoid large attention matrix
  }

  return true; // Default: unfused (faster on Apple Silicon)
}

namespace {

// Dispatch for the vector VJP kernel (sdpa_vector_vjp).
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
  // Build kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_vjp_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute KV sizes and strides
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // Verify output strides match input strides (kernel uses input strides
  // for output pointer arithmetic)
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

  // Function constants (same indices as forward)
  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks_flag = sinks.has_value();

  // When L=1 and no GQA sharing, each simdgroup writes to unique KV positions.
  // Skip atomics and use direct writes for better performance.
  int gqa_factor = q.shape(1) / k.shape(1);
  bool use_direct_write = (q.shape(2) == 1) && (gqa_factor == 1);
  bool use_dq_only = false;

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks_flag, MTL::DataType::DataTypeBool, 25},
      {&use_direct_write, MTL::DataType::DataTypeBool, 27},
      {&use_dq_only, MTL::DataType::DataTypeBool, 29},
  };

  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks_flag ? "_sinks" : "_nosinks";
  hash_name += use_direct_write ? "_dw" : "_aw";

  // Get kernel and set pipeline
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Inputs: Q, K, V, O, dO, logsumexp (buffers 0-5)
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);

  // Outputs: dQ, dK, dV (buffers 6-8)
  compute_encoder.set_output_array(d_q, 6);
  compute_encoder.set_output_array(d_k, 7);
  compute_encoder.set_output_array(d_v, 8);

  // Parameters: gqa_factor, N always at buffers 9-10
  compute_encoder.set_bytes(gqa_factor, 9);
  compute_encoder.set_bytes(N, 10);

  compute_encoder.set_bytes(k_head_stride, 11);
  compute_encoder.set_bytes(k_seq_stride, 12);
  compute_encoder.set_bytes(v_head_stride, 13);
  compute_encoder.set_bytes(v_seq_stride, 14);
  compute_encoder.set_bytes(scale, 15);

  // Output (O/dO) stride parameters - handle BLHV physical layout from STEEL.
  // The kernel uses the same strides for both out and d_out pointer arithmetic,
  // so they must match. Both are made row-contiguous in eval_gpu via
  // copy_unless(is_row_contiguous, ...), guaranteeing identical strides.
  assert(out.strides(0) == d_out.strides(0));
  assert(out.strides(1) == d_out.strides(1));
  assert(out.strides(2) == d_out.strides(2));
  int num_q_heads = q.shape(1);
  size_t o_batch_stride = out.strides(0);
  size_t o_head_stride = out.shape(1) == 1 ? 0 : out.strides(1);
  size_t o_seq_stride = out.strides(2);
  compute_encoder.set_bytes(num_q_heads, 16);
  compute_encoder.set_bytes(o_batch_stride, 17);
  compute_encoder.set_bytes(o_head_stride, 18);
  compute_encoder.set_bytes(o_seq_stride, 19);

  // Optional mask
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 20 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 22);
    compute_encoder.set_bytes(q_seq_stride, 23);
    compute_encoder.set_bytes(head_stride, 24);
  }

  // Optional sinks
  if (has_sinks_flag) {
    compute_encoder.set_input_array(*sinks, 25);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Dispatch the dQ kernel (first pass of 2-kernel split).
// Computes dQ and writes delta[B*H_q, L] for the dK/dV kernel.
void sdpa_vector_vjp_dq_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& out,
    const array& d_out,
    const array& logsumexp,
    array& d_q,
    array& delta,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Build kernel name (same kernel as before, with dq_only=true)
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_vjp_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

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
  bool has_sinks_flag = sinks.has_value();
  // dq_only=true: skip dK/dV writes, output delta
  bool use_dq_only = true;
  // direct_write is irrelevant when dq_only=true, set false
  bool use_direct_write = false;

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks_flag, MTL::DataType::DataTypeBool, 25},
      {&use_direct_write, MTL::DataType::DataTypeBool, 27},
      {&use_dq_only, MTL::DataType::DataTypeBool, 29},
  };

  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks_flag ? "_sinks" : "_nosinks";
  hash_name += "_dqonly";

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Inputs
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);

  // dQ output (buffers 6); buffers 7,8 not written in dq_only mode
  // but still need to be bound (they exist in the kernel signature)
  compute_encoder.set_output_array(d_q, 6);
  // Bind dummy buffers for dK/dV (not written but signature requires them)
  compute_encoder.set_output_array(d_q, 7); // reuse dQ as dummy
  compute_encoder.set_output_array(d_q, 8); // reuse dQ as dummy

  int gqa_factor = q.shape(1) / k.shape(1);
  compute_encoder.set_bytes(gqa_factor, 9);
  compute_encoder.set_bytes(N, 10);

  compute_encoder.set_bytes(k_head_stride, 11);
  compute_encoder.set_bytes(k_seq_stride, 12);
  compute_encoder.set_bytes(v_head_stride, 13);
  compute_encoder.set_bytes(v_seq_stride, 14);
  compute_encoder.set_bytes(scale, 15);

  // The kernel uses the same strides for both out and d_out pointer arithmetic,
  // so they must match. Both are made row-contiguous in eval_gpu via
  // copy_unless(is_row_contiguous, ...), guaranteeing identical strides.
  assert(out.strides(0) == d_out.strides(0));
  assert(out.strides(1) == d_out.strides(1));
  assert(out.strides(2) == d_out.strides(2));
  int num_q_heads = q.shape(1);
  size_t o_batch_stride = out.strides(0);
  size_t o_head_stride = out.shape(1) == 1 ? 0 : out.strides(1);
  size_t o_seq_stride = out.strides(2);
  compute_encoder.set_bytes(num_q_heads, 16);
  compute_encoder.set_bytes(o_batch_stride, 17);
  compute_encoder.set_bytes(o_head_stride, 18);
  compute_encoder.set_bytes(o_seq_stride, 19);

  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 20 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 22);
    compute_encoder.set_bytes(q_seq_stride, 23);
    compute_encoder.set_bytes(head_stride, 24);
  }

  if (has_sinks_flag) {
    compute_encoder.set_input_array(*sinks, 25);
  }

  // Delta output at buffer 26
  compute_encoder.set_output_array(delta, 26);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Dispatch the dK/dV kernel (second pass of 2-kernel split).
// No atomics - each simdgroup exclusively owns one KV position.
void sdpa_vector_vjp_dkdv_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& d_out,
    const array& logsumexp,
    const array& delta,
    array& d_k,
    array& d_v,
    float scale,
    bool do_causal,
    const std::optional<array>& mask) {
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_vjp_dkdv_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int L = q.shape(2);
  int num_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  int B = q.shape(0);

  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // Grid: each threadgroup handles BN=32 KV positions for one (batch, kv_head)
  constexpr int BN = 32;
  int n_kv_blocks = (N + BN - 1) / BN;
  MTL::Size grid_dims(B * n_kv_heads, n_kv_blocks, 1);
  MTL::Size group_dims(1024, 1, 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
  };

  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set buffers
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(d_out, 3);
  compute_encoder.set_input_array(logsumexp, 4);
  compute_encoder.set_input_array(delta, 5);
  compute_encoder.set_output_array(d_k, 6);
  compute_encoder.set_output_array(d_v, 7);

  compute_encoder.set_bytes(gqa_factor, 8);
  compute_encoder.set_bytes(N, 9);
  compute_encoder.set_bytes(L, 10);
  compute_encoder.set_bytes(num_q_heads, 11);
  compute_encoder.set_bytes(k_head_stride, 12);
  compute_encoder.set_bytes(k_seq_stride, 13);
  compute_encoder.set_bytes(v_head_stride, 14);
  compute_encoder.set_bytes(v_seq_stride, 15);
  compute_encoder.set_bytes(scale, 16);

  size_t o_batch_stride = d_out.strides(0);
  size_t o_head_stride = d_out.shape(1) == 1 ? 0 : d_out.strides(1);
  size_t o_seq_stride = d_out.strides(2);
  compute_encoder.set_bytes(o_batch_stride, 17);
  compute_encoder.set_bytes(o_head_stride, 18);
  compute_encoder.set_bytes(o_seq_stride, 19);

  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 20 + float_mask);
    int32_t kv_seq_stride_m = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride_m = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride_m =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride_m, 22);
    compute_encoder.set_bytes(q_seq_stride_m, 23);
    compute_encoder.set_bytes(head_stride_m, 24);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Dispatch the STEEL VJP dQ kernel.
// Computes dQ gradients using tiled matrix multiply on the GPU.
// Grid: [NQ, H, B] - one threadgroup per (query_block, head, batch)
void sdpa_steel_vjp_dq_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& out,
    const array& d_out,
    const array& logsumexp,
    array& d_q,
    float scale,
    bool do_causal,
    int qL_off_override = -1) {
  using namespace mlx::steel;

  constexpr int bq = 32;
  constexpr int wm = 4;
  constexpr int wn = 1;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  // Select BK based on head dimension:
  // D=64: BK=32, D=96/128: BK=16
  int bk = (D <= 64) ? 32 : 16;

  int qL = q.shape(2);
  int kL = k.shape(2);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;

  // Function constants (same indices as forward kernel)
  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
  };

  // Kernel name: matches host_name from instantiation macro
  // Format: attention_vjp_dq_{type}_{bq}_{bk}_{bd}
  std::string kname = "attention_vjp_dq_";
  kname += type_to_name(q);
  kname += "_";
  kname += std::to_string(bq);
  kname += "_";
  kname += std::to_string(bk);
  kname += "_";
  kname += std::to_string(D);

  std::string hash_name = kname;
  hash_name += "_align_Q_";
  hash_name += (align_Q ? 't' : 'n');
  hash_name += "_align_K_";
  hash_name += (align_K ? 't' : 'n');
  hash_name += "_causal_";
  hash_name += (do_causal ? 't' : 'n');

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // LSE strides: shape [B, H, qL] stored linearly as (batch * H + head) * qL +
  // pos
  int64_t lse_str_head = qL;
  int64_t lse_str_qpos = 1;

  // qL_off: causal mask offset. When caller pads Q/K to different block sizes,
  // the padded kL - qL may differ from the original. Use override if provided.
  int qL_off = (qL_off_override >= 0) ? qL_off_override : (kL - qL);

  AttnVJPParams params{
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
      /* int qL_off = */ qL_off,

      /* int64_t Q_strides[3] = */
      {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */
      {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */
      {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */
      {out.strides(0), out.strides(1), out.strides(2)},
      /* int64_t LSE_strides[2] = */ {lse_str_head, lse_str_qpos},

      /* int64_t dQ_strides[3] = */
      {d_q.strides(0), d_q.strides(1), d_q.strides(2)},
      /* int64_t dK_strides[3] = */ {0, 0, 0},
      /* int64_t dV_strides[3] = */ {0, 0, 0},
  };

  // Set buffers (must match kernel signature in steel_attention_vjp_dq.h)
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);
  compute_encoder.set_output_array(d_q, 6);
  compute_encoder.set_bytes(params, 7);

  // Grid: [NQ, H, B] - one threadgroup per (query_block, head, batch)
  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  // Group: WM * WN * 32 threads = 4 * 1 * 32 = 128
  MTL::Size group_dims = MTL::Size(wm * wn * 32, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Dispatch the STEEL VJP dKV kernel.
// Computes dK and dV gradients using tiled matrix multiply on the GPU.
// Grid: [NK, n_kv_heads, B] - one threadgroup per (kv_block, kv_head, batch)
void sdpa_steel_vjp_dkv_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& out,
    const array& d_out,
    const array& logsumexp,
    array& d_k,
    array& d_v,
    float scale,
    bool do_causal,
    int qL_off_override = -1) {
  using namespace mlx::steel;

  constexpr int bq = 32;
  constexpr int wm = 4;
  constexpr int wn = 1;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);
  int n_kv_heads = k.shape(1);

  // Select BK based on head dimension:
  // D=64: BK=32, D=96/128: BK=16
  int bk = (D <= 64) ? 32 : 16;

  int qL = q.shape(2);
  int kL = k.shape(2);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;

  // Function constants
  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
  };

  // Kernel name: matches host_name from instantiation macro
  // Format: attention_vjp_dkv_{type}_{bq}_{bk}_{bd}
  std::string kname = "attention_vjp_dkv_";
  kname += type_to_name(q);
  kname += "_";
  kname += std::to_string(bq);
  kname += "_";
  kname += std::to_string(bk);
  kname += "_";
  kname += std::to_string(D);

  std::string hash_name = kname;
  hash_name += "_align_Q_";
  hash_name += (align_Q ? 't' : 'n');
  hash_name += "_align_K_";
  hash_name += (align_K ? 't' : 'n');
  hash_name += "_causal_";
  hash_name += (do_causal ? 't' : 'n');

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  int64_t lse_str_head = qL;
  int64_t lse_str_qpos = 1;

  // qL_off: causal mask offset. When caller pads Q/K to different block sizes,
  // the padded kL - qL may differ from the original. Use override if provided.
  int qL_off = (qL_off_override >= 0) ? qL_off_override : (kL - qL);

  AttnVJPParams params{
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
      /* int qL_off = */ qL_off,

      /* int64_t Q_strides[3] = */
      {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */
      {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */
      {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */
      {out.strides(0), out.strides(1), out.strides(2)},
      /* int64_t LSE_strides[2] = */ {lse_str_head, lse_str_qpos},

      /* int64_t dQ_strides[3] = */ {0, 0, 0},
      /* int64_t dK_strides[3] = */
      {d_k.strides(0), d_k.strides(1), d_k.strides(2)},
      /* int64_t dV_strides[3] = */
      {d_v.strides(0), d_v.strides(1), d_v.strides(2)},
  };

  // Set buffers (must match kernel signature in steel_attention_vjp_dkv.h)
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(out, 3);
  compute_encoder.set_input_array(d_out, 4);
  compute_encoder.set_input_array(logsumexp, 5);
  compute_encoder.set_output_array(d_k, 6);
  compute_encoder.set_output_array(d_v, 7);
  compute_encoder.set_bytes(params, 8);

  // Grid: [NK, n_kv_heads, B] - one threadgroup per (kv_block, kv_head, batch)
  MTL::Size grid_dims = MTL::Size(NK, n_kv_heads, B);
  // Group: WM * WN * 32 threads = 4 * 1 * 32 = 128
  MTL::Size group_dims = MTL::Size(wm * wn * 32, 1, 1);

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
  // D=256 uses two-stage tiling (128-wide passes) to fit in 32KB.
  const int query_head_dim_pre = q_pre.shape(-1);
  const int value_head_dim_pre = v_pre.shape(-1);
  const bool vector_supported_head_dim =
      query_head_dim_pre == value_head_dim_pre &&
      (query_head_dim_pre == 64 || query_head_dim_pre == 96 ||
       query_head_dim_pre == 128 || query_head_dim_pre == 256);
  bool use_vector_vjp = (q_pre.shape(2) <= 8) && vector_supported_head_dim;

  // STEEL VJP: re-enabled behind policy control. On Apple Silicon with
  // NAX-optimized matmuls, unfused is faster for typical L. Fused VJP
  // avoids materializing O(L^2) attention matrix (84-96% memory savings
  // at L>=1024). Dispatch controlled by MLX_SDPA_VJP_MODE env var.
  // See use_fallback() for policy details.
  const bool steel_supported_head_dim_eval = (query_head_dim_pre == 64);
  const bool steel_supported_dtype_eval =
      (q_pre.dtype() == float16 || q_pre.dtype() == bfloat16);
  bool use_steel_vjp =
      steel_supported_head_dim_eval &&
      steel_supported_dtype_eval &&
      (q_pre.shape(2) > 8) &&
      !has_arr_mask &&
      !has_sinks_;

  auto is_row_contiguous = [](const array& arr) {
    return arr.flags().row_contiguous;
  };

  // STEEL VJP requires row-contiguous inputs because the kernel uses a single
  // set of O_strides for both O and dO pointer arithmetic. The forward SDPA
  // "full attention" kernel stores O in BLHD layout (non-standard strides),
  // so O must be copied to standard BHLD layout to match dO's strides.
  // Vector VJP requires row-contiguous K/V for output pointer arithmetic.
  const auto& q = use_steel_vjp ? copy_unless(is_row_contiguous, q_pre)
                                : copy_unless(q_is_vector_compatible, q_pre);
  const auto& k = use_vector_vjp ? copy_unless(is_row_contiguous, k_pre)
                                 : copy_unless(is_matrix_contiguous, k_pre);
  const auto& v = use_vector_vjp ? copy_unless(is_row_contiguous, v_pre)
                                 : copy_unless(is_matrix_contiguous, v_pre);
  // STEEL VJP needs row-contiguous O because the forward pass may store O with
  // non-standard strides (BLHD layout from sdpa_full_self_attention_metal).
  // Using the same O_strides for dO (which has standard BHLD strides) would
  // cause head cross-talk. Row-contiguous ensures both O and dO have matching
  // standard strides.
  const auto& o = copy_unless(is_row_contiguous, out);
  const auto& dO = use_steel_vjp ? copy_unless(is_row_contiguous, d_out)
                                 : copy_unless(is_row_contiguous, d_out);
  const auto& lse = copy_unless(is_matrix_contiguous, logsumexp);

  // Handle mask
  auto mask_pred = [&q](const array& arr) {
    return mask_is_compatible(q, arr);
  };
  std::optional<array> mask = std::nullopt;
  if (has_arr_mask) {
    mask = copy_unless(mask_pred, inputs[3]);
  }

  bool do_causal = do_causal_ && q.shape(2) > 1;

  // Dispatch to appropriate kernel
  if (use_steel_vjp) {
    // STEEL VJP: two-kernel split for D=64/96/128, L>8
    //
    // Pad Q/K/V sequence lengths to block boundaries so the kernels never
    // hit the slow load_safe (bounds-checked) path.  Padded positions are
    // filled with zeros which is safe:
    //   - Causal mask: padded Q rows have row_pos > kL → fully masked out.
    //   - Padded K cols: softmax([-inf, …, -inf]) → 0 attention weight.
    //
    // After dispatch, gradient outputs are trimmed back to original sizes.

    int D = q.shape(3);
    int B = q.shape(0);
    int H = q.shape(1);
    int n_kv_heads = k.shape(1);

    constexpr int bq = 32;
    int bk = (D <= 64) ? 32 : 16;

    int qL = q.shape(2);
    int kL = k.shape(2);
    int qL_padded = ((qL + bq - 1) / bq) * bq;
    int kL_padded = ((kL + bk - 1) / bk) * bk;

    bool need_q_pad = (qL != qL_padded);
    bool need_k_pad = (kL != kL_padded);

    // Original causal mask offset — must be preserved through padding.
    // When bq != bk, padded kL - padded qL differs from kL - qL.
    int orig_qL_off = kL - qL;
    // Only pass override when padding changes the offset
    int qL_off_arg =
        (need_q_pad || need_k_pad) ? orig_qL_off : -1;

    // Flags for trimmed outputs: not contiguous due to padding gaps.
    const auto trim_flags = array::Flags{false, false, false};

    // References to (possibly padded) arrays for dispatch
    auto q_use = q;
    auto o_use = o;
    auto dO_use = dO;
    auto lse_use = lse;
    auto k_use = k;
    auto v_use = v;

    // Scalar zero for fill_gpu
    array zero_scalar = array(0, q.dtype());

    // --- Pad Q-side arrays (Q, O, dO, LSE) if needed ---
    if (need_q_pad) {
      // Helper: pad a 4D array [B, H_dim, qL, D] → [B, H_dim, qL_padded, D]
      auto pad_4d_q = [&](const array& src, int H_dim) -> array {
        array padded(
            {src.shape(0), H_dim, qL_padded, src.shape(3)},
            src.dtype(),
            nullptr,
            {});
        fill_gpu(zero_scalar, padded, s);

        // Create a slice view into padded for the first qL rows (offset 0)
        array slice(src.shape(), padded.dtype(), nullptr, {});
        slice.copy_shared_buffer(
            padded, padded.strides(), padded.flags(), slice.size(), 0);

        // Copy original data into the slice
        copy_gpu_inplace(src, slice, CopyType::GeneralGeneral, s);

        copies.push_back(std::move(slice));
        return padded;
      };

      q_use = pad_4d_q(q, H);
      copies.push_back(q_use);

      o_use = pad_4d_q(o, H);
      copies.push_back(o_use);

      dO_use = pad_4d_q(dO, H);
      copies.push_back(dO_use);

      // Pad LSE: shape [B, H, qL, 1] → [B, H, qL_padded, 1]
      {
        array lse_pad(
            {lse.shape(0), lse.shape(1), qL_padded, lse.shape(3)},
            lse.dtype(),
            nullptr,
            {});
        array zero_f32 = array(0, lse.dtype());
        fill_gpu(zero_f32, lse_pad, s);
        copies.push_back(zero_f32);
        array lse_slice(lse.shape(), lse_pad.dtype(), nullptr, {});
        lse_slice.copy_shared_buffer(
            lse_pad, lse_pad.strides(), lse_pad.flags(), lse_slice.size(), 0);
        copy_gpu_inplace(lse, lse_slice, CopyType::GeneralGeneral, s);
        copies.push_back(std::move(lse_slice));
        lse_use = std::move(lse_pad);
        copies.push_back(lse_use);
      }
    }

    // --- Pad K/V arrays if needed ---
    if (need_k_pad) {
      auto pad_kv = [&](const array& src) -> array {
        array padded(
            {src.shape(0), n_kv_heads, kL_padded, src.shape(3)},
            src.dtype(),
            nullptr,
            {});
        fill_gpu(zero_scalar, padded, s);
        array slice(src.shape(), padded.dtype(), nullptr, {});
        slice.copy_shared_buffer(
            padded, padded.strides(), padded.flags(), slice.size(), 0);
        copy_gpu_inplace(src, slice, CopyType::GeneralGeneral, s);
        copies.push_back(std::move(slice));
        return padded;
      };

      k_use = pad_kv(k);
      copies.push_back(k_use);
      v_use = pad_kv(v);
      copies.push_back(v_use);
    }

    copies.push_back(zero_scalar);

    // --- Allocate output gradient arrays ---
    // If padded, allocate padded-size outputs; otherwise use original sizes.
    if (need_q_pad) {
      // d_q: padded size [B, H, qL_padded, D]
      array d_q_padded({B, H, qL_padded, D}, d_q.dtype(), nullptr, {});
      d_q_padded.set_data(allocator::malloc(d_q_padded.nbytes()));

      if (need_k_pad) {
        array d_k_padded(
            {B, n_kv_heads, kL_padded, D}, d_k.dtype(), nullptr, {});
        d_k_padded.set_data(allocator::malloc(d_k_padded.nbytes()));
        array d_v_padded(
            {B, n_kv_heads, kL_padded, D}, d_v.dtype(), nullptr, {});
        d_v_padded.set_data(allocator::malloc(d_v_padded.nbytes()));

        // dQ and dKV have independent outputs — dispatch concurrently
        // to avoid the unnecessary memory barrier between them.
        {
          auto& enc = d.get_command_encoder(s.index);
          auto concurrent = enc.start_concurrent();
          sdpa_steel_vjp_dq_dispatch(
              s,
              d,
              q_use,
              k_use,
              v_use,
              o_use,
              dO_use,
              lse_use,
              d_q_padded,
              scale_,
              do_causal,
              qL_off_arg);
          sdpa_steel_vjp_dkv_dispatch(
              s,
              d,
              q_use,
              k_use,
              v_use,
              o_use,
              dO_use,
              lse_use,
              d_k_padded,
              d_v_padded,
              scale_,
              do_causal,
              qL_off_arg);
        }

        // Trim outputs: share padded buffer but expose only original rows.
        d_q.copy_shared_buffer(
            d_q_padded, d_q_padded.strides(), trim_flags, d_q.size(), 0);
        d_k.copy_shared_buffer(
            d_k_padded, d_k_padded.strides(), trim_flags, d_k.size(), 0);
        d_v.copy_shared_buffer(
            d_v_padded, d_v_padded.strides(), trim_flags, d_v.size(), 0);

        copies.push_back(std::move(d_q_padded));
        copies.push_back(std::move(d_k_padded));
        copies.push_back(std::move(d_v_padded));
      } else {
        // Only Q padded, K/V aligned
        d_k.set_data(allocator::malloc(d_k.nbytes()));
        d_v.set_data(allocator::malloc(d_v.nbytes()));

        {
          auto& enc = d.get_command_encoder(s.index);
          auto concurrent = enc.start_concurrent();
          sdpa_steel_vjp_dq_dispatch(
              s,
              d,
              q_use,
              k_use,
              v_use,
              o_use,
              dO_use,
              lse_use,
              d_q_padded,
              scale_,
              do_causal,
              qL_off_arg);
          sdpa_steel_vjp_dkv_dispatch(
              s,
              d,
              q_use,
              k_use,
              v_use,
              o_use,
              dO_use,
              lse_use,
              d_k,
              d_v,
              scale_,
              do_causal,
              qL_off_arg);
        }

        d_q.copy_shared_buffer(
            d_q_padded, d_q_padded.strides(), trim_flags, d_q.size(), 0);
        copies.push_back(std::move(d_q_padded));
      }
    } else if (need_k_pad) {
      // Only K/V padded, Q aligned
      d_q.set_data(allocator::malloc(d_q.nbytes()));
      array d_k_padded(
          {B, n_kv_heads, kL_padded, D}, d_k.dtype(), nullptr, {});
      d_k_padded.set_data(allocator::malloc(d_k_padded.nbytes()));
      array d_v_padded(
          {B, n_kv_heads, kL_padded, D}, d_v.dtype(), nullptr, {});
      d_v_padded.set_data(allocator::malloc(d_v_padded.nbytes()));

      {
        auto& enc = d.get_command_encoder(s.index);
        auto concurrent = enc.start_concurrent();
        sdpa_steel_vjp_dq_dispatch(
            s,
            d,
            q_use,
            k_use,
            v_use,
            o_use,
            dO_use,
            lse_use,
            d_q,
            scale_,
            do_causal,
            qL_off_arg);
        sdpa_steel_vjp_dkv_dispatch(
            s,
            d,
            q_use,
            k_use,
            v_use,
            o_use,
            dO_use,
            lse_use,
            d_k_padded,
            d_v_padded,
            scale_,
            do_causal,
            qL_off_arg);
      }

      d_k.copy_shared_buffer(
          d_k_padded, d_k_padded.strides(), trim_flags, d_k.size(), 0);
      d_v.copy_shared_buffer(
          d_v_padded, d_v_padded.strides(), trim_flags, d_v.size(), 0);
      copies.push_back(std::move(d_k_padded));
      copies.push_back(std::move(d_v_padded));
    } else {
      // Both aligned — no padding needed
      d_q.set_data(allocator::malloc(d_q.nbytes()));
      d_k.set_data(allocator::malloc(d_k.nbytes()));
      d_v.set_data(allocator::malloc(d_v.nbytes()));

      {
        auto& enc = d.get_command_encoder(s.index);
        auto concurrent = enc.start_concurrent();
        sdpa_steel_vjp_dq_dispatch(
            s, d, q, k, v, o, dO, lse, d_q, scale_, do_causal);
        sdpa_steel_vjp_dkv_dispatch(
            s, d, q, k, v, o, dO, lse, d_k, d_v, scale_, do_causal);
      }
    }
  } else if (use_vector_vjp) {
    // Allocate output gradient arrays for vector path
    d_q.set_data(allocator::malloc(d_q.nbytes()));
    d_k.set_data(allocator::malloc(d_k.nbytes()));
    d_v.set_data(allocator::malloc(d_v.nbytes()));

    // When L=1, no GQA sharing, and no explicit mask, each simdgroup writes
    // unique KV positions. The non-accumulate kernel uses direct writes (no
    // atomics) and is instantiated for float32, float16, and bfloat16.
    //
    // When an explicit mask is present, we cannot use the direct-write path
    // because some KV positions may be masked and never written. Instead, the
    // 2-kernel split is used: the dK/dV kernel parallelizes over KV positions
    // with exclusive ownership, so all positions are written (no zero-init
    // needed). Note: do_causal is not a concern here because it requires
    // q.shape(2) > 1, which is incompatible with can_direct_write requiring
    // q.shape(2) == 1.
    int gqa_factor = q.shape(1) / k.shape(1);
    bool can_direct_write =
        (q.shape(2) == 1) && (gqa_factor == 1) && !has_arr_mask;

    if (can_direct_write) {
      // Direct write: use non-accumulate kernel, writes T directly (no atomics)
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
    } else {
      // 2-kernel split: dQ kernel + atomic-free dK/dV kernel

      // Allocate delta buffer [B*H_q*L] for inter-kernel communication
      int B = q.shape(0);
      int H_q = q.shape(1);
      int L = q.shape(2);
      array delta_buf({B * H_q * L}, float32, nullptr, {});
      delta_buf.set_data(allocator::malloc(delta_buf.nbytes()));

      // Pass 1: Compute dQ and delta (skips dK/dV writes)
      sdpa_vector_vjp_dq_dispatch(
          s,
          d,
          q,
          k,
          v,
          o,
          dO,
          lse,
          d_q,
          delta_buf,
          scale_,
          do_causal,
          mask,
          sinks);

      // Pass 2: Compute dK/dV using delta (no atomics)
      sdpa_vector_vjp_dkdv_dispatch(
          s, d, q, k, v, dO, lse, delta_buf, d_k, d_v, scale_, do_causal, mask);

      d.add_temporary(delta_buf, s.index);
    }
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
