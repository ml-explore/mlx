#include <iostream>
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/mlx.h"
#include "mlx/stream.h"

namespace mx = mlx::core;

int main() {
  int group_size = 16;
  int bits = 4;
  int M = 128;
  int N = 128;
  int K = 256;
  std::string quantization_mode = "nvfp4";

  mx::Device device(mx::Device::gpu, 0);
  auto s = mx::default_stream(device);
  auto& encoder = mx::cu::get_command_encoder(s);

  mx::array a = mx::random::uniform({M, K}, mx::bfloat16); // (M, K)
  mx::array b = mx::random::uniform({N, K}, mx::bfloat16); // (N, K)

  auto scaled_a = mx::quantize(a, group_size, bits, quantization_mode);
  auto scaled_b = mx::quantize(b, group_size, bits, quantization_mode);

  mx::array a_quantized = scaled_a[0];
  mx::array a_scale = scaled_a[1];
  mx::array b_quantized = scaled_b[0];
  mx::array b_scale = scaled_b[1];

  mx::array out = mx::qqmm(
      a_quantized,
      b_quantized,
      a_scale,
      b_scale,
      true,
      group_size,
      bits,
      quantization_mode);

  mx::array a_dequantized =
      mx::dequantize(a_quantized, a_scale, {}, 16, 4, "nvfp4");
  mx::array b_dequantized =
      mx::dequantize(b_quantized, b_scale, {}, 16, 4, "nvfp4");

  mx::array reference_deq =
      mx::matmul(a_dequantized, mx::transpose(b_dequantized));
  mx::array isclose = mx::allclose(out, reference_deq, 1e-1f);

  std::cout << isclose << std::endl;
  return 0;
}