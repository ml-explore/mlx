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

namespace {

inline int gated_delta_chunk_size(int T) {
  int C = metal::is_nax_available() ? 16 : 8;
  C = T > 8 ? C : 1;
  return env::get_var("GATED_DELTA_CHUNK", C);
}

inline int gated_delta_chunk_size_vjp(int T) {
  int C = metal::is_nax_available() ? 16 : 1;
  return env::get_var("GATED_DELTA_CHUNK_VJP", C);
}

inline int gated_delta_ckpt(int chunk) {
  int c = env::get_var("GATED_DELTA_CKPT", 16);
  if (c != 1 && c != 4 && c != 8 && c != 16) {
    c = 16;
  }
  return (chunk == 16) ? c : 1;
}

inline int gated_delta_n_ckpt(int n_chunks, int ckpt) {
  return (n_chunks + ckpt - 1) / ckpt;
}

bool supported_gated_delta_shape(int Hk, int Dk, int Hv, int Dv) {
  if (Dk != 128 || Dv != 128) {
    return false;
  }
  return (Hk == 24 && Hv == 24) || (Hk == 32 && Hv == 32) ||
      (Hk == 16 && Hv == 32) || (Hk == 16 && Hv == 48) ||
      (Hk == 16 && Hv == 16) || (Hk == 16 && Hv == 64);
}

array ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

array scratch_alloc(Shape shape, Dtype dt, const Stream& s) {
  array a(std::move(shape), dt, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  metal::get_command_encoder(s).add_temporary(a);
  return a;
}

} // namespace

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
  return !supported_gated_delta_shape(Hk, Dk, Hv, Dv);
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
  const int ckpt = gated_delta_ckpt(C);

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
  metal::MTLFCList func_consts = {
      {&save_state, MTL::DataType::DataTypeBool, 200},
  };

  array dummy = scratch_alloc({1}, float32, s);

  switch (C) {
    case 16: {
      std::string base_name = "gated_delta_fused_nax_" + suffix + "_" +
          std::to_string(C) + "_" + std::to_string(ckpt);

      auto delta_kernel =
          get_gated_delta_nax_kernel(d, base_name, base_name, func_consts);

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

      auto delta_kernel =
          get_gated_delta_kernel(d, base_name, base_name, func_consts);

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
      std::string base_name =
          "seq_gated_delta_" + suffix + "_" + std::to_string(ckpt);

      auto delta_kernel =
          get_gated_delta_kernel(d, base_name, base_name, func_consts);

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
  if (env::get_var("GATED_DELTA_VJP_FALLBACK", 0) != 0) {
    return true;
  }
  return !supported_gated_delta_shape(Hk, Dk, Hv, Dv);
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

  // 16 = chunked NAX, anything else = sequential.
  int C = gated_delta_chunk_size_vjp(T);
  const bool chunked = (C == 16);

  const int n_chunks = (T + 15) / 16;
  const int ckpt = gated_delta_ckpt(C);
  const int n_states = chunked ? gated_delta_n_ckpt(n_chunks, ckpt) : T;

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

  // The kernels accumulate their gradients as fp32, so we may need a cast.
  const bool stage_fp32 = (q.dtype() != float32);

  auto dq_acc = stage_fp32 ? scratch_alloc(dq.shape(), float32, s) : dq;
  auto dk_acc = stage_fp32 ? scratch_alloc(dk.shape(), float32, s) : dk;
  auto dv_acc = stage_fp32 ? scratch_alloc(dv.shape(), float32, s) : dv;
  auto dg_acc = stage_fp32 ? scratch_alloc(dg.shape(), float32, s) : dg;
  auto db_acc = stage_fp32 ? scratch_alloc(db.shape(), float32, s) : db;

  fill_gpu(array(0, float32), dq_acc, s);
  fill_gpu(array(0, float32), dk_acc, s);
  fill_gpu(array(0, float32), dv_acc, s);
  fill_gpu(array(0, float32), dg_acc, s);
  fill_gpu(array(0, float32), db_acc, s);
  fill_gpu(array(0, dh.dtype()), dh, s);

  array state_cache = scratch_alloc({B, Hv, n_states, Dv, Dk}, float32, s);
  array seg_states = scratch_alloc(
      chunked ? Shape({B, Hv, ckpt, Dv, Dk}) : Shape({1}), float32, s);
  array chunk_mats = scratch_alloc(
      chunked ? Shape({B, Hv, n_chunks, 3, 16, 16}) : Shape({1}), float32, s);
  array chunk_delta = scratch_alloc(
      chunked ? Shape({B, Hv, n_chunks, 16, Dv}) : Shape({1}), float32, s);

  if (chunked) {
    fill_gpu(array(0, float32), state_cache, s);
    fill_gpu(array(0, float32), chunk_mats, s);
    fill_gpu(array(0, float32), chunk_delta, s);
  }

  array y_scratch = scratch_alloc({B, T, Hv, Dv}, q.dtype(), s);
  array hf_scratch = scratch_alloc({B, Hv, Dv, Dk}, float32, s);

  std::string suffix = get_type_string(q.dtype()) + "_" + std::to_string(Dk) +
      "_" + std::to_string(Dv) + "_" + std::to_string(Hk) + "_" +
      std::to_string(Hv);

  bool save_state = true;
  metal::MTLFCList save_consts = {
      {&save_state, MTL::DataType::DataTypeBool, 200},
  };
  metal::MTLFCList no_consts = {};

  switch (C) {
    case 16: {
      const std::string ckpt_suffix =
          "_" + std::to_string(C) + "_" + std::to_string(ckpt);

      // Forward save pass.
      {
        std::string base_name = "gated_delta_fused_nax_" + suffix + ckpt_suffix;

        auto delta_kernel = get_gated_delta_nax_kernel(
            d, base_name, base_name + "_save", save_consts);

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

        auto delta_kernel =
            get_gated_delta_vjp_nax_kernel(d, base_name, base_name, no_consts);

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

        auto dgamma_kernel =
            get_gated_delta_vjp_nax_kernel(d, base_name, base_name, no_consts);

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

        auto delta_kernel = get_gated_delta_kernel(
            d, base_name, base_name + "_save", save_consts);

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

        auto delta_kernel =
            get_gated_delta_vjp_kernel(d, base_name, base_name, no_consts);

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

  if (stage_fp32) {
    copy_gpu(dq_acc, dq, CopyType::General, s);
    copy_gpu(dk_acc, dk, CopyType::General, s);
    copy_gpu(dv_acc, dv, CopyType::General, s);
    copy_gpu(dg_acc, dg, CopyType::General, s);
    copy_gpu(db_acc, db, CopyType::General, s);
  }
}

} // namespace mlx::core::fast
