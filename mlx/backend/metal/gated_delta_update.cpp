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

bool GatedDeltaUpdate::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    const bool has_mask,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }

  if (has_mask) {
    return true;
  }

  if (Dk != 128 || Dv != 128) {
    return true;
  }

  const bool supported_heads = (Hk == 24 && Hv == 24) ||
      (Hk == 32 && Hv == 32) || (Hk == 16 && Hv == 32) ||
      (Hk == 16 && Hv == 48);
  if (!supported_heads) {
    return true;
  }

  return false;
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto q = ensure_row_contiguous(inputs[0], d, s);
  auto k = ensure_row_contiguous(inputs[1], d, s);
  auto v = ensure_row_contiguous(inputs[2], d, s);
  auto g = ensure_row_contiguous(inputs[3], d, s);
  auto beta = ensure_row_contiguous(inputs[4], d, s);
  auto h0 = ensure_row_contiguous(inputs[5], d, s);

  auto& out = outputs[0];
  auto& hf = outputs[1];

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);

  int C = 1;
  const char* threashold_env = std::getenv("GATED_DELTA_THRESH");
  int threshold = threashold_env ? std::stoi(threashold_env) : 16;
  if (T > threshold) {
    if (metal::is_nax_available())
      C = 16;
    else
      C = 8;
  }
  const char* chunk_env = std::getenv("GATED_DELTA_CHUNK");
  C = chunk_env ? std::stoi(chunk_env) : C;

  if (!metal::is_nax_available())
    C = std::min(C, 8); // override in case nax is not available.

  std::string suffix = get_type_string(q.dtype()) + "_" + std::to_string(Dk) +
      "_" + std::to_string(Dv) + "_" + std::to_string(Hk) + "_" +
      std::to_string(Hv);

  auto& compute_encoder = metal::get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));
  hf.set_data(allocator::malloc(hf.nbytes()));

  switch (C) {
    case 16: {
      std::string kernel_name = "gated_delta_fused_nax_";
      std::string base_name = kernel_name + suffix;

      base_name += "_" + std::to_string(C);

      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel =
          get_gated_delta_nax_kernel(d, base_name, hash_name, func_consts);

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
      auto threads = MTL::Size(32, 4, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    case 8: {
      std::string kernel_name = "gated_delta_fused_chunk_";
      std::string base_name = kernel_name + suffix;

      base_name += "_" + std::to_string(C);

      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel =
          get_gated_delta_kernel(d, base_name, hash_name, func_consts);

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
      auto threads = MTL::Size(32, 4, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    case 1:
    case 0: {
      std::string kernel_name = "seq_gated_delta_";
      std::string base_name = kernel_name + suffix;
      std::string hash_name = base_name;

      metal::MTLFCList func_consts = {};

      auto delta_kernel =
          get_gated_delta_kernel(d, base_name, hash_name, func_consts);

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
          "NYI: Only sequential and chunk size 8,16 are supported");
    }
  }
}

} // namespace mlx::core::fast
