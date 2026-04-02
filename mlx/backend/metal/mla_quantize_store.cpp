// Fused MLA quantize-on-store — Metal GPU dispatch

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void MLAQuantizeStore::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& input      = inputs[0];   // [..., 256]
  auto& packed     = outputs[0];  // [..., 32]
  auto& out_scales = outputs[1];  // [..., 4]
  auto& out_biases = outputs[2];  // [..., 4]

  // N = total vectors = product of all dims except last
  uint32_t N = 1;
  for (int i = 0; i < input.ndim() - 1; i++) {
    N *= static_cast<uint32_t>(input.shape(i));
  }

  packed.set_data(allocator::malloc(packed.nbytes()));
  out_scales.set_data(allocator::malloc(out_scales.nbytes()));
  out_biases.set_data(allocator::malloc(out_biases.nbytes()));

  std::string kname;
  if (input.dtype() == float16) {
    kname = "mla_quantize_store_f16";
  } else if (input.dtype() == bfloat16) {
    kname = "mla_quantize_store_bf16";
  } else {
    throw std::runtime_error(
        "MLAQuantizeStore: input must be float16 or bfloat16");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(input, 0);
  enc.set_output_array(packed, 1);
  enc.set_output_array(out_scales, 2);
  enc.set_output_array(out_biases, 3);

  enc.set_bytes(N, 4);

  // One threadgroup (32 threads = 1 simdgroup) per vector
  enc.dispatch_threadgroups(
      MTL::Size::Make(N, 1, 1),
      MTL::Size::Make(32, 1, 1));
}

}  // namespace mlx::core::fast
