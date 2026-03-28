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

void TurboQuantAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Unpack inputs
  auto& q_rot = inputs[0];     // (B, H_q, qL, D)
  auto& q_sketch = inputs[1];  // (B, H_q, qL, D)
  auto& k_packed = inputs[2];  // (B, H_kv, kL, packed_d_mse)
  auto& k_signs = inputs[3];   // (B, H_kv, kL, packed_d_signs)
  auto& k_norms = inputs[4];   // (B, H_kv, kL)
  auto& k_res_norms = inputs[5]; // (B, H_kv, kL)
  auto& centroids = inputs[6]; // (n_centroids,)
  auto& v_packed = inputs[7];  // (B, H_kv, kL, packed_d_v)
  auto& v_scales = inputs[8];  // (B, H_kv, kL, n_groups)
  auto& v_zeros = inputs[9];   // (B, H_kv, kL, n_groups)

  auto& o = outputs[0]; // (B, H_q, qL, D)

  // Ensure contiguous layout via copies
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

  // Allocate output
  o.set_data(allocator::malloc(o.nbytes()));

  // Extract dimensions
  int B = q_r.shape(0);
  int H_q = q_r.shape(1);
  int qL = q_r.shape(2);
  int D = q_r.shape(3);
  int H_kv = kp.shape(1);
  int kL = kp.shape(2);

  // Build params
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

  // Build kernel name: sdpa_vector_turboquant_<type>_<D>
  std::string kname = "sdpa_vector_turboquant_";
  kname += get_type_string(q_r.dtype());
  kname += "_";
  kname += std::to_string(D);

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set kernel arguments (must match buffer indices in shader)
  compute_encoder.set_input_array(q_r, 0);     // q_rot
  compute_encoder.set_input_array(q_s, 1);     // q_sketch
  compute_encoder.set_input_array(kp, 2);      // k_packed
  compute_encoder.set_input_array(ks, 3);      // k_signs
  compute_encoder.set_input_array(kn, 4);      // k_norms
  compute_encoder.set_input_array(krn, 5);     // k_res_norms
  compute_encoder.set_input_array(cen, 6);     // centroids
  compute_encoder.set_input_array(vp, 7);      // v_packed
  compute_encoder.set_input_array(vs, 8);      // v_scales
  compute_encoder.set_input_array(vz, 9);      // v_zeros
  compute_encoder.set_output_array(o, 10);     // output
  compute_encoder.set_bytes(params, 11);       // params struct

  // Grid: one threadgroup per (batch×head, query_position)
  // Threadgroup: 1024 = 32 SIMD groups × 32 threads
  MTL::Size grid_dims = MTL::Size(B * H_q, qL, 1);
  MTL::Size group_dims = MTL::Size(1024, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
