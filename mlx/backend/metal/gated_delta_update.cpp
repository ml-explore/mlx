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
      (Hk == 16 && Hv == 48) || (Hk == 16 && Hv == 16) ||
      (Hk == 16 && Hv == 64);
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

  int C = gated_delta_chunk_size(T);
  if (!metal::is_nax_available()) {
    C = std::min(C, 8);
  }

  std::string suffix = get_type_string(q.dtype()) + "_" + std::to_string(Dk) +
      "_" + std::to_string(Dv) + "_" + std::to_string(Hk) + "_" +
      std::to_string(Hv);

  auto& compute_encoder = metal::get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));
  hf.set_data(allocator::malloc(hf.nbytes()));

  fill_gpu(array(0, out.dtype()), out, s);

  // The forward is inference-only here: the vjp runs its own save pass. The
  // kernel declares buffers 9, 10 and 11 unconditionally, so bind a placeholder
  // and turn the stores off with the function constant.
  bool save_state = false;

  array dummy({1}, float32, nullptr, {});
  dummy.set_data(allocator::malloc(dummy.nbytes()));
  compute_encoder.add_temporary(dummy);

  switch (C) {
    case 16: {
      const int ckpt = gated_delta_ckpt();
      std::string base_name = "gated_delta_fused_nax_" + suffix + "_" +
          std::to_string(C) + "_" + std::to_string(ckpt);
      std::string hash_name = base_name + (save_state ? "_save" : "");

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
      compute_encoder.set_output_array(dummy, 9); // state_cache
      compute_encoder.set_output_array(dummy, 10); // chunk_mats
      compute_encoder.set_output_array(dummy, 11); // chunk_delta

      auto grid = MTL::Size(32, Dv / 16, B * Hv);
      auto threads = MTL::Size(32, 4, 1);
      compute_encoder.dispatch_threads(grid, threads);
      break;
    }
    case 8: {
      std::string base_name =
          "gated_delta_fused_chunk_" + suffix + "_" + std::to_string(C);
      std::string hash_name = base_name;

      bool no_save = false;
      metal::MTLFCList func_consts = {
          {&no_save, MTL::DataType::DataTypeBool, 200},
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
      // Ckpt is a template parameter now, so even the inference path has to
      // name one; it just never stores.
      const int ckpt = gated_delta_ckpt();
      std::string base_name =
          "seq_gated_delta_" + suffix + "_" + std::to_string(ckpt);
      std::string hash_name = base_name;

      bool no_save = false;
      metal::MTLFCList func_consts = {
          {&no_save, MTL::DataType::DataTypeBool, 200},
      };

      auto delta_kernel =
          get_gated_delta_kernel(d, base_name, hash_name, func_consts);

      compute_encoder.set_compute_pipeline_state(delta_kernel);

      // Order must match the kernel signature: state_in is slot 3, g is 4.
      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_input_array(h0, 3);
      compute_encoder.set_input_array(g, 4);
      compute_encoder.set_input_array(beta, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder.set_output_array(hf, 7);
      compute_encoder.set_bytes(T, 8);
      // Slot 9 is gated off by the function constant, so it stays unbound.

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
  return false;

  // like the forward
  if (Dk != 128 || Dv != 128) {
    return true;
  }

  const bool supported_heads = (Hk == 24 && Hv == 24) ||
      (Hk == 32 && Hv == 32) || (Hk == 16 && Hv == 32) ||
      (Hk == 16 && Hv == 48) || (Hk == 16 && Hv == 16) ||
      (Hk == 16 && Hv == 64);
  if (!supported_heads) {
    return true;
  }

  return false;
}

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

  int n_chunks = (T + 15) / 16;

  const int ckpt = (C == 16) ? gated_delta_ckpt() : 1;
  const int n_states =
      (C == 16) ? gated_delta_n_ckpt(n_chunks, ckpt) : ((T + ckpt - 1) / ckpt);

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

  bool narrow = (q.dtype() != float32);

  auto accum_for = [&](const array& out) {
    if (!narrow) {
      return out;
    }
    array a(out.shape(), float32, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    compute_encoder.add_temporary(a);
    return a;
  };

  auto dq_acc = accum_for(dq);
  auto dk_acc = accum_for(dk);
  auto dv_acc = accum_for(dv);
  auto dg_acc = accum_for(dg);
  auto db_acc = accum_for(db);

  fill_gpu(array(0, float32), dq_acc, s);
  fill_gpu(array(0, float32), dk_acc, s);
  fill_gpu(array(0, float32), dv_acc, s);
  fill_gpu(array(0, float32), dg_acc, s);
  fill_gpu(array(0, float32), db_acc, s);
  fill_gpu(array(0, dh.dtype()), dh, s);

  array state_cache({B, Hv, n_states, Dv, Dk}, float32, nullptr, {});
  array chunk_mats({B, Hv, n_chunks, 3, 16, 16}, float32, nullptr, {});
  array chunk_delta({B, Hv, n_chunks, 16, Dv}, float32, nullptr, {});
  array seg_states(
      (C == 16) ? Shape({B, Hv, ckpt, Dv, Dk}) : Shape({1}),
      float32,
      nullptr,
      {});

  state_cache.set_data(allocator::malloc(state_cache.nbytes()));
  if (C == 16) {
    fill_gpu(array(0, state_cache.dtype()), state_cache, s);
  }
  compute_encoder.add_temporary(state_cache);

  seg_states.set_data(allocator::malloc(seg_states.nbytes()));
  compute_encoder.add_temporary(seg_states);

  if (C == 16) {
    chunk_mats.set_data(allocator::malloc(chunk_mats.nbytes()));
    fill_gpu(array(0, chunk_mats.dtype()), chunk_mats, s);
    compute_encoder.add_temporary(chunk_mats);

    chunk_delta.set_data(allocator::malloc(chunk_delta.nbytes()));
    fill_gpu(array(0, chunk_delta.dtype()), chunk_delta, s);
    compute_encoder.add_temporary(chunk_delta);
  }

  array y_scratch({B, T, Hv, Dv}, q.dtype(), nullptr, {});
  array hf_scratch({B, Hv, Dv, Dk}, float32, nullptr, {});

  y_scratch.set_data(allocator::malloc(y_scratch.nbytes()));
  compute_encoder.add_temporary(y_scratch);

  hf_scratch.set_data(allocator::malloc(hf_scratch.nbytes()));
  compute_encoder.add_temporary(hf_scratch);

  std::string suffix = get_type_string(q.dtype()) + "_" + std::to_string(Dk) +
      "_" + std::to_string(Dv) + "_" + std::to_string(Hk) + "_" +
      std::to_string(Hv);

  // printf("state_cache %.2f  chunk_delta %.2f  chunk_mats %.2f GiB\n",
  //     state_cache.nbytes() / double(1 << 30),
  //     chunk_delta.nbytes() / double(1 << 30),
  //     chunk_mats.nbytes() / double(1 << 30));
  switch (C) {
    case 16: {
      const std::string ckpt_suffix =
          "_" + std::to_string(C) + "_" + std::to_string(ckpt);

      // Forward save pass.
      {
        std::string base_name = "gated_delta_fused_nax_" + suffix + ckpt_suffix;
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
        compute_encoder.set_output_array(y_scratch, 6);
        compute_encoder.set_output_array(hf_scratch, 7);
        compute_encoder.set_bytes(T, 8);
        compute_encoder.set_output_array(state_cache, 9);
        compute_encoder.set_output_array(chunk_mats, 10);
        compute_encoder.set_output_array(chunk_delta, 11);

        auto grid = MTL::Size(32, Dv / 16, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

      {
        std::string base_name =
            "gated_delta_vjp_fused_nax_" + suffix + ckpt_suffix;
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
        compute_encoder.set_output_array(dq_acc, 9);
        compute_encoder.set_output_array(dk_acc, 10);
        compute_encoder.set_output_array(dv_acc, 11);
        compute_encoder.set_output_array(dg_acc, 12);
        compute_encoder.set_output_array(db_acc, 13);
        compute_encoder.set_output_array(dh, 14);
        compute_encoder.set_input_array(chunk_mats, 15);
        compute_encoder.set_input_array(chunk_delta, 16);
        compute_encoder.set_output_array(seg_states, 17);

        auto grid = MTL::Size(32, Dv / 16, B * Hv);
        auto threads = MTL::Size(32, Dv / 16, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

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
        compute_encoder.set_output_array(dg_acc, 1);
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
      const std::string ckpt_suffix = "_" + std::to_string(ckpt);

      {
        std::string base_name = "seq_gated_delta_" + suffix + ckpt_suffix;
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
        compute_encoder.set_input_array(h0, 3);
        compute_encoder.set_input_array(g, 4);
        compute_encoder.set_input_array(beta, 5);
        compute_encoder.set_output_array(y_scratch, 6);
        compute_encoder.set_output_array(hf_scratch, 7);
        compute_encoder.set_bytes(T, 8);
        compute_encoder.set_output_array(state_cache, 9);

        auto grid = MTL::Size(32, Dv, B * Hv);
        auto threads = MTL::Size(32, 4, 1);
        compute_encoder.dispatch_threads(grid, threads);
      }

      {
        std::string base_name = "seq_gated_delta_vjp_" + suffix + ckpt_suffix;
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
        compute_encoder.set_output_array(dq_acc, 9);
        compute_encoder.set_output_array(dk_acc, 10);
        compute_encoder.set_output_array(dv_acc, 11);
        compute_encoder.set_output_array(dg_acc, 12);
        compute_encoder.set_output_array(db_acc, 13);
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

  if (narrow) {
    copy_gpu(dq_acc, dq, CopyType::General, s);
    copy_gpu(dk_acc, dk, CopyType::General, s);
    copy_gpu(dv_acc, dv, CopyType::General, s);
    copy_gpu(dg_acc, dg, CopyType::General, s);
    copy_gpu(db_acc, db, CopyType::General, s);
  }
}

} // namespace mlx::core::fast
