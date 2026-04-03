// MLA shared-latent nope score — Metal GPU dispatch
// Following the pattern from scaled_dot_product_attention.cpp

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void MLANopeScores::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q_nope   = inputs[0];  // [B, H, 256]
  auto& k_packed = inputs[1];  // [B, S, 32]
  auto& k_scales = inputs[2];  // [B, S, 4]
  auto& k_biases = inputs[3];  // [B, S, 4]
  auto& out      = outputs[0]; // [B, H, S]

  const uint32_t B = static_cast<uint32_t>(q_nope.shape(0));
  const uint32_t H = static_cast<uint32_t>(q_nope.shape(1));
  const uint32_t S = static_cast<uint32_t>(k_packed.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  // Select kernel based on dtype
  std::string kname;
  if (q_nope.dtype() == float16) {
    kname = "mla_nope_scores_shared_latent_f16";
  } else if (q_nope.dtype() == bfloat16) {
    kname = "mla_nope_scores_shared_latent_bf16";
  } else {
    throw std::runtime_error(
        "MLANopeScores: q_nope must be float16 or bfloat16");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  // The kernel is compiled into MLX's metallib
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(q_nope, 0);
  enc.set_input_array(k_packed, 1);
  enc.set_input_array(k_scales, 2);
  enc.set_input_array(k_biases, 3);
  enc.set_output_array(out, 4);

  enc.set_bytes(B, 5);
  enc.set_bytes(H, 6);
  enc.set_bytes(S, 7);
  enc.set_bytes(scale_, 8);

  constexpr size_t H_TILE = 8;
  auto ceil_div = [](size_t x, size_t y) { return (x + y - 1) / y; };

  enc.dispatch_threadgroups(
      MTL::Size::Make(
          ceil_div(static_cast<size_t>(H), H_TILE),
          static_cast<size_t>(S),
          static_cast<size_t>(B)),
      MTL::Size::Make(32, H_TILE, 1));
}

}  // namespace mlx::core::fast
