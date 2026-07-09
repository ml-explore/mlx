// Copyright © 2024 Apple Inc.
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

bool GatedDeltaUpdate::use_fallback(Stream s) {
  // TODO: finish implementation
  return false;
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& g = inputs[3];
  auto& beta = inputs[4];
  auto& h0 = inputs[5];

  auto& out = outputs[0];
  auto& hf = outputs[1];

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);

  // printf("%d %d %d %d %d %d\n",B,T,Hk,Hv,Dk,Dv);

  int C = chunk_size;

  std::string suffix = get_type_string(q.dtype()) // "float"
      + "_" + get_type_string(h0.dtype()) // "float"
      + "_" + std::to_string(Dk) + "_" + std::to_string(Dv) + "_" +
      std::to_string(Hk) + "_" + std::to_string(Hv);

  auto& compute_encoder = metal::get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));
  hf.set_data(allocator::malloc(hf.nbytes()));

  fill_gpu(array(0, out.dtype()), out, s);

  switch (C) {
    case 16: {
      std::string kernel_name = "gated_delta_fused_nax_";

      // printf("C: %d\nname: %s\n",C,kernel_name.c_str());
      std::string base_name = kernel_name + suffix;

      base_name += "_" + std::to_string(C);

      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel = get_steel_gated_delta_forward_kernel(
          d, base_name, hash_name, func_consts);

      compute_encoder.set_compute_pipeline_state(delta_kernel);
      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_input_array(h0, 3); // initial state in
      compute_encoder.set_input_array(g, 4);
      compute_encoder.set_input_array(beta, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder.set_output_array(hf, 7); // final state out
      compute_encoder.set_bytes(T, 8);

      auto grid = MTL::Size(32, Dv / 16, B * Hv);
      auto threads = MTL::Size(32, 1, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    case 8: {
      std::string kernel_name = "gated_delta_fused_chunk_";

      // printf("C: %d\nname: %s\n",C,kernel_name.c_str());
      std::string base_name = kernel_name + suffix;

      base_name += "_" + std::to_string(C);

      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel = get_steel_gated_delta_forward_kernel(
          d, base_name, hash_name, func_consts);

      compute_encoder.set_compute_pipeline_state(delta_kernel);
      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_input_array(h0, 3); // initial state in
      compute_encoder.set_input_array(g, 4);
      compute_encoder.set_input_array(beta, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder.set_output_array(hf, 7); // final state out
      compute_encoder.set_bytes(T, 8);

      auto grid = MTL::Size(32, Dv / 8, B * Hv);
      auto threads = MTL::Size(32, 1, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    case 1:
    case 0: {
      std::string kernel_name = "seq_gated_delta_";
      std::string base_name = kernel_name + suffix;
      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel = get_steel_gated_delta_forward_kernel(
          d, base_name, hash_name, func_consts);

      compute_encoder.set_compute_pipeline_state(delta_kernel);

      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_input_array(g, 3);
      compute_encoder.set_input_array(beta, 4);
      compute_encoder.set_input_array(h0, 5);
      compute_encoder.set_bytes(T, 6);
      compute_encoder.set_output_array(out, 7);
      compute_encoder.set_output_array(hf, 8);

      auto grid = MTL::Size(32, Dv, B * Hv);
      auto threads = MTL::Size(32, 4, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    default: {
      throw std::runtime_error(
          "NYI: Only sequential and chunk size 8 are supported");
    }
  }
}

bool GatedDeltaUpdate::is_equivalent(const Primitive& other) const {
  const auto* p = dynamic_cast<const GatedDeltaUpdate*>(&other);
  if (p == nullptr) {
    return false;
  }
  // TODO: finish implementation
  return true;
}

} // namespace mlx::core::fast
