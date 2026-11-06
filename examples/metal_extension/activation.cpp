// Copyright © 2026 Apple Inc.

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary directory.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

class SiluAndMul : public mx::UnaryPrimitive {
 public:
  explicit SiluAndMul(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("SiluAndMul is only implemented for the GPU.");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& input = inputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
      return;
    }

    const uint32_t last_dim = input.shape(-1);
    const uint32_t d = last_dim / 2;
    const uint32_t num_tokens = input.size() / last_dim;
    const uint32_t num_chunks = (d + 7) / 8;

    auto& device = mx::metal::device(stream().device);
    auto library =
        device.get_library(MLX_METAL_LIBRARY_NAME, current_binary_dir());
    auto kernel = device.get_kernel("silu_and_mul_f16", library);

    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_output_array(out, 0);
    encoder.set_input_array(input, 1);
    encoder.set_bytes(d, 2);

    const size_t group_width = std::min(
        static_cast<size_t>(num_chunks),
        static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()));
    encoder.dispatch_threads(
        MTL::Size(num_chunks, num_tokens, 1), MTL::Size(group_width, 1, 1));
  }

  const char* name() const override {
    return "SiluAndMul";
  }

  bool is_equivalent(const mx::Primitive&) const override {
    return true;
  }
};

mx::array silu_and_mul(const mx::array& input, mx::StreamOrDevice s = {}) {
  if (input.ndim() == 0) {
    throw std::invalid_argument(
        "silu_and_mul input must have at least one dimension.");
  }
  if (input.shape(-1) % 2 != 0) {
    throw std::invalid_argument(
        "silu_and_mul requires an even-sized last dimension.");
  }
  if (input.dtype() != mx::float16) {
    throw std::invalid_argument("silu_and_mul only supports float16.");
  }

  auto stream = mx::to_stream(s);
  auto contiguous_input = mx::contiguous(input, false, stream);
  auto output_shape = input.shape();
  output_shape.back() /= 2;
  return mx::array(
      output_shape,
      input.dtype(),
      std::make_shared<SiluAndMul>(stream),
      {contiguous_input});
}

} // namespace

NB_MODULE(MLX_EXTENSION_NAME, m) {
  m.doc() =
      "Fused float16 SiLU and multiplication implemented with MLX and Metal.";
  m.def(
      "silu_and_mul",
      &silu_and_mul,
      "input"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def silu_and_mul(input: mlx.core.array, *, stream: "
          "mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | "
          "None = None) -> mlx.core.array"),
      R"pbdoc(
        Apply a fused SiLU activation and multiplication.

        Splits the last dimension into equal halves ``x`` and ``y``, then
        computes ``silu(x) * y``. The input must be a float16 array with an
        even-sized last dimension.

        Args:
            input (mlx.core.array): Input array.
            stream (Stream or Device, optional): Stream or device on which to
              schedule the operation. Defaults to ``None``.

        Returns:
            mlx.core.array: An array whose last dimension is half that of the
            input.
      )pbdoc");
}
