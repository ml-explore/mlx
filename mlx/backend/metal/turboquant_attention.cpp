// Copyright © 2024-25 Apple Inc.
// Metal dispatch for TurboQuant fused attention kernel.

#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params_turboquant.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

void sdpa_turboquant_1pass(
    const Stream& s,
    metal::Device& d,
    const array& q_r,
    const array& q_s,
    const array& kp,
    const array& ks,
    const array& kn,
    const array& krn,
    const array& cen,
    const array& vp,
    const array& vs,
    const array& vz,
    array& o,
    array& o_m,
    array& o_l,
    const mlx::steel::TurboQuantAttnParams& params,
    int B,
    int H_q,
    int qL,
    int D) {
  std::string kname = "sdpa_vector_turboquant_";
  kname += get_type_string(q_r.dtype());
  kname += "_" + std::to_string(D);
  kname += "_" + std::to_string(params.mse_bits);
  kname += "_" + std::to_string(params.v_bits);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q_r, 0);
  compute_encoder.set_input_array(q_s, 1);
  compute_encoder.set_input_array(kp, 2);
  compute_encoder.set_input_array(ks, 3);
  compute_encoder.set_input_array(kn, 4);
  compute_encoder.set_input_array(krn, 5);
  compute_encoder.set_input_array(cen, 6);
  compute_encoder.set_input_array(vp, 7);
  compute_encoder.set_input_array(vs, 8);
  compute_encoder.set_input_array(vz, 9);
  compute_encoder.set_output_array(o, 10);
  compute_encoder.set_output_array(o_m, 11);
  compute_encoder.set_output_array(o_l, 12);
  compute_encoder.set_bytes(params, 13);

  MTL::Size grid_dims = MTL::Size(B * H_q, qL, 1);
  MTL::Size group_dims = MTL::Size(1024, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_turboquant_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q_r,
    const array& q_s,
    const array& kp,
    const array& ks,
    const array& kn,
    const array& krn,
    const array& cen,
    const array& vp,
    const array& vs,
    const array& vz,
    array& o,
    array& o_m,
    array& o_l,
    const mlx::steel::TurboQuantAttnParams& params,
    int B,
    int H_q,
    int H_kv,
    int qL,
    int D,
    int kL) {
  // Determine number of blocks based on sequence length
  int blocks = 32;
  if (kL > 4096) {
    blocks = 64;
  }
  if (kL > 16384) {
    blocks = 128;
  }

  int gqa_factor = H_q / H_kv;

  // Pass 1: partial results per block
  std::string kname1 = "sdpa_vector_turboquant_2pass_1_";
  kname1 += get_type_string(q_r.dtype());
  kname1 += "_" + std::to_string(D);
  kname1 += "_" + std::to_string(params.mse_bits);
  kname1 += "_" + std::to_string(params.v_bits);

  // Allocate intermediates
  Shape inter_shape = {B * H_q, blocks, D};
  array intermediate(inter_shape, q_r.dtype(), nullptr, {});
  Shape scalar_shape = {B * H_q, blocks};
  array inter_sums(scalar_shape, float32, nullptr, {});
  array inter_maxs(scalar_shape, float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  inter_sums.set_data(allocator::malloc(inter_sums.nbytes()));
  inter_maxs.set_data(allocator::malloc(inter_maxs.nbytes()));
  d.add_temporary(intermediate, s.index);
  d.add_temporary(inter_sums, s.index);
  d.add_temporary(inter_maxs, s.index);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel1 = d.get_kernel(kname1);
  compute_encoder.set_compute_pipeline_state(kernel1);

  compute_encoder.set_input_array(q_r, 0);
  compute_encoder.set_input_array(q_s, 1);
  compute_encoder.set_input_array(kp, 2);
  compute_encoder.set_input_array(ks, 3);
  compute_encoder.set_input_array(kn, 4);
  compute_encoder.set_input_array(krn, 5);
  compute_encoder.set_input_array(cen, 6);
  compute_encoder.set_input_array(vp, 7);
  compute_encoder.set_input_array(vs, 8);
  compute_encoder.set_input_array(vz, 9);
  compute_encoder.set_output_array(intermediate, 10);
  compute_encoder.set_output_array(inter_sums, 11);
  compute_encoder.set_output_array(inter_maxs, 12);
  compute_encoder.set_bytes(params, 13);

  // Grid: (H_kv, B, blocks), Threadgroup: (32, gqa_factor, 1)
  MTL::Size grid_dims1 = MTL::Size(H_kv, B, blocks);
  MTL::Size group_dims1 = MTL::Size(32, gqa_factor, 1);
  compute_encoder.dispatch_threadgroups(grid_dims1, group_dims1);

  // Pass 2: merge blocks
  std::string kname2 = "sdpa_vector_turboquant_2pass_2_";
  kname2 += get_type_string(q_r.dtype());
  kname2 += "_";
  kname2 += std::to_string(D);

  auto kernel2 = d.get_kernel(kname2);
  compute_encoder.set_compute_pipeline_state(kernel2);

  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(inter_sums, 1);
  compute_encoder.set_input_array(inter_maxs, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_output_array(o_m, 4);
  compute_encoder.set_output_array(o_l, 5);
  compute_encoder.set_bytes(blocks, 6);

  MTL::Size grid_dims2 = MTL::Size(B * H_q, 1, 1);
  MTL::Size group_dims2 = MTL::Size(1024, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims2, group_dims2);
}

} // namespace

void TurboQuantAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_rot = inputs[0];
  auto& q_sketch = inputs[1];
  auto& k_packed = inputs[2];
  auto& k_signs = inputs[3];
  auto& k_norms = inputs[4];
  auto& k_res_norms = inputs[5];
  auto& centroids = inputs[6];
  auto& v_packed = inputs[7];
  auto& v_scales = inputs[8];
  auto& v_zeros = inputs[9];

  auto& o = outputs[0];
  auto& o_m = outputs[1];
  auto& o_l = outputs[2];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  auto ensure_contiguous = [&copies, &s](const array& arr) -> const array& {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
  };

  const auto& q_r = ensure_contiguous(q_rot);
  const auto& q_s = ensure_contiguous(q_sketch);
  const auto& kp = ensure_contiguous(k_packed);
  const auto& ks = ensure_contiguous(k_signs);
  const auto& kn = ensure_contiguous(k_norms);
  const auto& krn = ensure_contiguous(k_res_norms);
  const auto& cen = ensure_contiguous(centroids);
  const auto& vp = ensure_contiguous(v_packed);
  const auto& vs = ensure_contiguous(v_scales);
  const auto& vz = ensure_contiguous(v_zeros);

  o.set_data(allocator::malloc(o.nbytes()));
  o_m.set_data(allocator::malloc(o_m.nbytes()));
  o_l.set_data(allocator::malloc(o_l.nbytes()));

  int B = q_r.shape(0);
  int H_q = q_r.shape(1);
  int qL = q_r.shape(2);
  int D = q_r.shape(3);
  int H_kv = kp.shape(1);
  int kL = kp.shape(2);

  mlx::steel::TurboQuantAttnParams params;
  params.N = kL;
  params.gqa_factor = H_q / H_kv;
  params.scale = scale_;
  params.qjl_scale = qjl_scale_;
  params.packed_d_mse = kp.shape(3);
  params.packed_d_signs = ks.shape(3);
  params.packed_d_v = vp.shape(3);
  params.n_groups = D / group_size_;
  params.group_size = group_size_;
  params.n_centroids = cen.shape(0);
  params.mse_bits = mse_bits_;
  params.v_bits = v_bits_;

  // Route: 2-pass for long sequences, 1-pass otherwise
  // 2-pass requires blocks to be a multiple of 32 (for pass 2 reduction)
  if (kL >= 1024 && qL == 1) {
    sdpa_turboquant_2pass(
        s,
        d,
        q_r,
        q_s,
        kp,
        ks,
        kn,
        krn,
        cen,
        vp,
        vs,
        vz,
        o,
        o_m,
        o_l,
        params,
        B,
        H_q,
        H_kv,
        qL,
        D,
        kL);
  } else {
    sdpa_turboquant_1pass(
        s,
        d,
        q_r,
        q_s,
        kp,
        ks,
        kn,
        krn,
        cen,
        vp,
        vs,
        vz,
        o,
        o_m,
        o_l,
        params,
        B,
        H_q,
        qL,
        D);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
