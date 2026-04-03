// Fused quantized MLA SDPA — Metal GPU dispatch
// Based on scaled_dot_product_attention.cpp dispatch pattern

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void MLAFusedSDPA::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q_nope    = inputs[0];  // [B, H, 256]
  auto& q_pe      = inputs[1];  // [B, H, 64]
  auto& lat_packed = inputs[2]; // [B, S, 32]
  auto& lat_scales = inputs[3]; // [B, S, 4]
  auto& lat_biases = inputs[4]; // [B, S, 4]
  auto& k_pe      = inputs[5];  // [B, S, 64]
  auto& out       = outputs[0]; // [B, H, 256]

  const uint32_t B = static_cast<uint32_t>(q_nope.shape(0));
  const uint32_t H = static_cast<uint32_t>(q_nope.shape(1));
  const uint32_t S = static_cast<uint32_t>(lat_packed.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname;
  if (q_nope.dtype() == float16) {
    kname = "mla_fused_sdpa_f16";
  } else if (q_nope.dtype() == bfloat16) {
    kname = "mla_fused_sdpa_bf16";
  } else {
    throw std::runtime_error(
        "MLAFusedSDPA: q_nope must be float16 or bfloat16");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(q_nope, 0);
  enc.set_input_array(q_pe, 1);
  enc.set_input_array(lat_packed, 2);
  enc.set_input_array(lat_scales, 3);
  enc.set_input_array(lat_biases, 4);
  enc.set_input_array(k_pe, 5);
  enc.set_output_array(out, 6);

  enc.set_bytes(B, 7);
  enc.set_bytes(H, 8);
  enc.set_bytes(S, 9);
  enc.set_bytes(scale_, 10);

  // One threadgroup per (head, batch)
  // BN=32 simdgroups × BD=32 threads = 1024 threads per threadgroup
  // Grid: (H, B, 1) — one threadgroup per head per batch
  // Group: (1024, 1, 1) — flat, simdgroup_index handles the rest
  enc.dispatch_threadgroups(
      MTL::Size::Make(H, B, 1),
      MTL::Size::Make(1024, 1, 1));
}

// V2: Fused SDPA + direct cache update (eliminates SliceUpdate)
void MLAFusedSDPAWithCacheUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q_nope      = inputs[0];  // [B, H, 256]
  auto& q_pe        = inputs[1];  // [B, H, 64]
  auto& cache_packed = inputs[2]; // [B, S_alloc, 32]
  auto& cache_scales = inputs[3]; // [B, S_alloc, 4]
  auto& cache_biases = inputs[4]; // [B, S_alloc, 4]
  auto& cache_kpe    = inputs[5]; // [B, S_alloc, 64]
  auto& new_latent   = inputs[6]; // [B, 1, 256]
  auto& new_kpe      = inputs[7]; // [B, 1, 64]

  auto& sdpa_out     = outputs[0]; // [B, H, 256] — new allocation
  auto& out_packed   = outputs[1]; // [B, S_alloc, 32] — aliased to cache_packed
  auto& out_scales   = outputs[2]; // [B, S_alloc, 4]  — aliased to cache_scales
  auto& out_biases   = outputs[3]; // [B, S_alloc, 4]  — aliased to cache_biases
  auto& out_kpe      = outputs[4]; // [B, S_alloc, 64] — aliased to cache_kpe

  const uint32_t B = static_cast<uint32_t>(q_nope.shape(0));
  const uint32_t H = static_cast<uint32_t>(q_nope.shape(1));
  const uint32_t S_alloc = static_cast<uint32_t>(cache_packed.shape(1));

  // SDPA output — new allocation
  sdpa_out.set_data(allocator::malloc(sdpa_out.nbytes()));

  // Cache outputs — alias to input buffers (zero-copy, eliminates SliceUpdate)
  out_packed.copy_shared_buffer(cache_packed);
  out_scales.copy_shared_buffer(cache_scales);
  out_biases.copy_shared_buffer(cache_biases);
  out_kpe.copy_shared_buffer(cache_kpe);

  std::string kname;
  if (q_nope.dtype() == float16) {
    kname = "mla_fused_sdpa_v2_f16";
  } else if (q_nope.dtype() == bfloat16) {
    kname = "mla_fused_sdpa_v2_bf16";
  } else {
    throw std::runtime_error(
        "MLAFusedSDPAWithCacheUpdate: q_nope must be float16 or bfloat16");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(q_nope, 0);
  enc.set_input_array(q_pe, 1);
  // Cache arrays: set as OUTPUT (write tracking for barriers)
  enc.set_output_array(out_packed, 2);
  enc.set_output_array(out_scales, 3);
  enc.set_output_array(out_biases, 4);
  enc.set_output_array(out_kpe, 5);
  enc.set_input_array(new_latent, 6);
  enc.set_input_array(new_kpe, 7);
  enc.set_output_array(sdpa_out, 8);

  enc.set_bytes(B, 9);
  enc.set_bytes(H, 10);
  enc.set_bytes(seq_offset_, 11);   // S: current occupancy
  enc.set_bytes(S_alloc, 12);
  enc.set_bytes(scale_, 13);

  enc.dispatch_threadgroups(
      MTL::Size::Make(H, B, 1),
      MTL::Size::Make(1024, 1, 1));
}

}  // namespace mlx::core::fast
