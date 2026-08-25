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

  fill_gpu(array(0, out.dtype()), out, s);

  switch (C) {
    case 16: {
      std::string kernel_name = "gated_delta_fused_nax_";
      std::string base_name = kernel_name + suffix;

      base_name += "_" + std::to_string(C);

      std::string hash_name = base_name;

      bool save_state = false;
      metal::MTLFCList func_consts = {
          {&save_state, MTL::DataType::DataTypeBool, 200},
      };

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

      bool save_state = false;
      metal::MTLFCList func_consts = {
          {&save_state, MTL::DataType::DataTypeBool, 200},
      };

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

      bool save_state = false;
      metal::MTLFCList func_consts = {
          {&save_state, MTL::DataType::DataTypeBool, 200},
      };

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

/////////////////////////
// BACKWARD PASS STUFF //
/////////////////////////
bool GatedDeltaUpdateVJP::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }

  if (const char* e = std::getenv("GATED_DELTA_VJP_FALLBACK");
      e != nullptr && std::stoi(e) != 0) {
    return true;
  }

  // like the forward
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

#define PRINT_STRIDES(arr)                 \
  printf(                                  \
      "%s strides: %lld %lld %lld %lld\n", \
      #arr,                                \
      arr.strides()[0],                    \
      arr.strides()[1],                    \
      arr.strides()[2],                    \
      arr.strides()[3])

#define PRINT_SHAPES(arr)                 \
  printf(                                 \
      "%s shapes: %lld %lld %lld %lld\n", \
      #arr,                               \
      arr.shape()[0],                     \
      arr.shape()[1],                     \
      arr.shape()[2],                     \
      arr.shape()[3])

#define PRINT_ARR(arr)                      \
  if (arr.flags().row_contiguous)           \
    printf("%s is row contiguous\n", #arr); \
  PRINT_SHAPES(arr);                        \
  PRINT_STRIDES(arr);                       \
  printf("\n");

void GatedDeltaUpdateVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Inputs are q, k, v, g, b, h0, cot_o, cot_h
  // Outputs are dq, dk, dv, dg, db, dh0
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto q = ensure_row_contiguous(inputs[0], d, s);
  auto k = ensure_row_contiguous(inputs[1], d, s);
  auto v = ensure_row_contiguous(inputs[2], d, s);
  auto g = ensure_row_contiguous(inputs[3], d, s);
  auto beta = ensure_row_contiguous(inputs[4], d, s);
  auto h0 = ensure_row_contiguous(inputs[5], d, s);
  auto cot_o = ensure_row_contiguous(inputs[6], d, s);
  auto cot_h = ensure_row_contiguous(inputs[7], d, s);

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);

  // VJP implementation: 16 = chunked NAX, anything else = sequential.
  const char* vjp_chunk_env = std::getenv("GATED_DELTA_VJP_CHUNK");
  int C = vjp_chunk_env ? std::stoi(vjp_chunk_env) : 0;
  if (C == 16 && !metal::is_nax_available()) {
    C = 0;
  }

  // No checkpoint interval: the chunked path stores one state per chunk, the
  // sequential path one state per timestep.
  int n_chunks = (T + 15) / 16;
  int n_states = (C == 16) ? n_chunks : T;

  auto& dq = outputs[0];
  auto& dk = outputs[1];
  auto& dv = outputs[2];
  auto& dg = outputs[3];
  auto& db = outputs[4];
  auto& dh = outputs[5];

  auto& compute_encoder = metal::get_command_encoder(s);

  dq.set_data(allocator::malloc(dq.nbytes()));
  dk.set_data(allocator::malloc(dk.nbytes()));
  dv.set_data(allocator::malloc(dv.nbytes()));
  dg.set_data(allocator::malloc(dg.nbytes()));
  db.set_data(allocator::malloc(db.nbytes()));
  dh.set_data(allocator::malloc(dh.nbytes()));

  fill_gpu(array(0, dq.dtype()), dq, s);
  fill_gpu(array(0, dk.dtype()), dk, s);
  fill_gpu(array(0, dv.dtype()), dv, s);
  fill_gpu(array(0, dg.dtype()), dg, s);
  fill_gpu(array(0, db.dtype()), db, s);
  fill_gpu(array(0, dh.dtype()), dh, s);

  array state_cache({B, Hv, n_states, Dv, Dk}, float32, nullptr, {});
  state_cache.set_data(allocator::malloc(state_cache.nbytes()));
  fill_gpu(array(0, state_cache.dtype()), state_cache, s);
  compute_encoder.add_temporary(state_cache);

  array scratch({1}, float32, nullptr, {});
  scratch.set_data(allocator::malloc(scratch.nbytes()));
  fill_gpu(array(0, scratch.dtype()), scratch, s);
  compute_encoder.add_temporary(scratch);

  std::string suffix = get_type_string(q.dtype()) + "_" + std::to_string(Dk) +
      "_" + std::to_string(Dv) + "_" + std::to_string(Hk) + "_" +
      std::to_string(Hv);

  switch (C) {
    case 16: {
      // Forward save pass: NAX chunked kernel, one checkpoint per chunk.
      {
        std::string base_name =
            "gated_delta_fused_nax_" + suffix + "_" + std::to_string(C);
        std::string hash_name = base_name + "_save";

        bool save_state = true;
        metal::MTLFCList func_consts = {
            {&save_state, MTL::DataType::DataTypeBool, 200},
        };

        auto delta_kernel =
            get_gated_delta_nax_kernel(d, base_name, hash_name, func_consts);

        compute_encoder.set_compute_pipeline_state(delta_kernel);
        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k, 1);
        compute_encoder.set_input_array(v, 2);
        compute_encoder.set_input_array(h0, 3);
        compute_encoder.set_input_array(g, 4);
        compute_encoder.set_input_array(beta, 5);
        compute_encoder.set_output_array(scratch, 6);
        compute_encoder.set_output_array(scratch, 7);
        compute_encoder.set_bytes(T, 8);
        compute_encoder.set_output_array(state_cache, 9);

        auto grid = MTL::Size(32, Dv / 16, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

      // Backward pass: NAX chunked VJP.
      {
        std::string base_name =
            "gated_delta_vjp_fused_nax_" + suffix + "_" + std::to_string(C);
        std::string hash_name = base_name;

        metal::MTLFCList func_consts = {};

        auto delta_kernel = get_gated_delta_vjp_nax_kernel(
            d, base_name, hash_name, func_consts);

        compute_encoder.set_compute_pipeline_state(delta_kernel);
        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k, 1);
        compute_encoder.set_input_array(v, 2);
        compute_encoder.set_input_array(g, 3);
        compute_encoder.set_input_array(beta, 4);
        compute_encoder.set_input_array(cot_o, 5);
        compute_encoder.set_input_array(cot_h, 6);
        compute_encoder.set_input_array(state_cache, 7);
        compute_encoder.set_bytes(T, 8);
        compute_encoder.set_output_array(dq, 9);
        compute_encoder.set_output_array(dk, 10);
        compute_encoder.set_output_array(dv, 11);
        compute_encoder.set_output_array(dg, 12);
        compute_encoder.set_output_array(db, 13);
        compute_encoder.set_output_array(dh, 14);

        auto grid = MTL::Size(32, Dv / 16, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

      // The chunked kernel leaves dL/dgamma in dg. Convert it to dL/dg with a
      // per-chunk reverse cumulative sum followed by the 1/g factor.
      {
        int n_total = B * Hv;

        std::string base_name = "gated_delta_dgamma_to_dg_" +
            get_type_string(q.dtype()) + "_" + std::to_string(C);
        std::string hash_name = base_name;

        metal::MTLFCList func_consts = {};

        auto dgamma_kernel = get_gated_delta_vjp_nax_kernel(
            d, base_name, hash_name, func_consts);

        compute_encoder.set_compute_pipeline_state(dgamma_kernel);
        compute_encoder.set_input_array(g, 0);
        compute_encoder.set_output_array(dg, 1);
        compute_encoder.set_bytes(T, 2);
        compute_encoder.set_bytes(Hv, 3);
        compute_encoder.set_bytes(n_total, 4);

        auto grid = MTL::Size(n_total, n_chunks, 1);
        auto threads = MTL::Size(std::min(n_total, 32), 1, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }
      break;
    }
    case 1:
    case 0: {
      // Forward save pass: sequential kernel, one state per timestep.
      {
        std::string base_name = "seq_gated_delta_" + suffix;
        std::string hash_name = base_name + "_save";

        bool save_state = true;
        metal::MTLFCList func_consts = {
            {&save_state, MTL::DataType::DataTypeBool, 200},
        };

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
        compute_encoder.set_output_array(scratch, 7);
        compute_encoder.set_output_array(scratch, 8);
        compute_encoder.set_output_array(state_cache, 9);

        auto grid = MTL::Size(32, Dv, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

      // Backward pass: sequential VJP.
      {
        std::string base_name = "seq_gated_delta_vjp_" + suffix;
        std::string hash_name = base_name;

        metal::MTLFCList func_consts = {};

        auto delta_kernel =
            get_gated_delta_vjp_kernel(d, base_name, hash_name, func_consts);

        compute_encoder.set_compute_pipeline_state(delta_kernel);
        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k, 1);
        compute_encoder.set_input_array(v, 2);
        compute_encoder.set_input_array(g, 3);
        compute_encoder.set_input_array(beta, 4);
        compute_encoder.set_input_array(cot_o, 5);
        compute_encoder.set_input_array(cot_h, 6);
        compute_encoder.set_input_array(state_cache, 7);
        compute_encoder.set_bytes(T, 8);
        compute_encoder.set_output_array(dq, 9);
        compute_encoder.set_output_array(dk, 10);
        compute_encoder.set_output_array(dv, 11);
        compute_encoder.set_output_array(dg, 12);
        compute_encoder.set_output_array(db, 13);
        compute_encoder.set_output_array(dh, 14);

        auto grid = MTL::Size(32, Dv, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }
      break;
    }
    default: {
      throw std::runtime_error(
          "NYI: Only sequential and chunk size 16 are supported for vjp");
    }
  }
}

} // namespace mlx::core::fast
