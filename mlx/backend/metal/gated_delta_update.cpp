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

  int C = chunk_size;
  int n_chunks = T / C; // TODO: make general

  std::string kernel_name = C < 1 ? "seq_gated_delta_" : "chunk_gated_delta_";
  std::string suffix = get_type_string(q.dtype()) // "float"
      + "_" + get_type_string(h0.dtype()) // "float"
      + "_" + std::to_string(Dk) + "_" + std::to_string(Dv) + "_" +
      std::to_string(Hk) + "_" + std::to_string(Hv);

  // printf("C: %d\nname: %s\n",C,kernel_name.c_str());
  std::string base_name = kernel_name + suffix;

  base_name += C >= 1 ? "_" + std::to_string(C) : "";

  std::string hash_name = base_name;

  metal::MTLFCList func_consts = {};

  auto delta_kernel = get_steel_gated_delta_forward_kernel(
      d, base_name, hash_name, func_consts);

  auto& compute_encoder = metal::get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));
  hf.set_data(allocator::malloc(hf.nbytes()));

  if (C > 1) {
    // allocate full W and U -- [B, T, H, D]
    array W({B, T, Hk, Dk}, q.dtype(), nullptr, {});
    array U({B, T, Hv, Dv}, q.dtype(), nullptr, {});
    W.set_data(allocator::malloc(W.nbytes()));
    U.set_data(allocator::malloc(U.nbytes()));

    compute_encoder.add_temporary(W);
    compute_encoder.add_temporary(U);

    // kernel 1: compute full W and U
    std::string make_wy_name = "make_wy_" +
        get_type_string(q.dtype()) // "float"
        + "_" + std::to_string(Dk) + "_" + std::to_string(Dv) + "_" +
        std::to_string(Hk) + "_" + std::to_string(Hv) + "_" + std::to_string(C);

    auto wy_kernel = d.get_kernel(make_wy_name);

    compute_encoder.set_compute_pipeline_state(wy_kernel);
    compute_encoder.set_input_array(k, 0);
    compute_encoder.set_input_array(v, 1);
    compute_encoder.set_input_array(g, 2);
    compute_encoder.set_input_array(beta, 3);
    compute_encoder.set_output_array(W, 4);
    compute_encoder.set_output_array(U, 5);
    compute_encoder.set_bytes(T, 6);

    auto grid_wy = MTL::Size(32, 4, B * Hv * n_chunks);
    auto threads_wy = MTL::Size(32, 4, 1);
    compute_encoder.dispatch_threads(grid_wy, threads_wy);

    // kernel 2: gated delta
    compute_encoder.set_compute_pipeline_state(delta_kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(W, 2);
    compute_encoder.set_input_array(U, 3);
    compute_encoder.set_input_array(h0, 4); // initial state in
    compute_encoder.set_input_array(g, 5);
    compute_encoder.set_output_array(out, 6);
    compute_encoder.set_output_array(hf, 7); // final state out
    compute_encoder.set_bytes(T, 8);
    compute_encoder.set_bytes(n_chunks, 9);

    auto grid = MTL::Size(32, Dv / 8, B * Hv);
    auto threads = MTL::Size(32, 1, 1); // get one simdgroup to work
    // auto grid   = MTL::Size(1, 1, 1);
    // auto threads = MTL::Size(1, 1, 1);
    compute_encoder.dispatch_threads(grid, threads);

  } else {
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

    // auto grid   = MTL::Size(1, 1, 1);
    // auto threads = MTL::Size(1, 1, 1);
    auto grid = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
    compute_encoder.dispatch_threads(grid, threads);
  }
  // throw std::runtime_error("NYI");
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
