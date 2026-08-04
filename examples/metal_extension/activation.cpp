// Copyright © 2026 Apple Inc.

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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

class Activation : public mx::UnaryPrimitive {
 public:
  Activation(
      mx::Stream stream,
      std::string kernel_name,
      bool gated,
      std::optional<float> threshold = std::nullopt)
      : mx::UnaryPrimitive(stream),
        kernel_name_(std::move(kernel_name)),
        gated_(gated),
        threshold_(threshold) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Activation is only implemented for the GPU.");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& input = inputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
      return;
    }

    const uint32_t last_dim = input.shape(-1);
    const uint32_t d = gated_ ? last_dim / 2 : last_dim;
    const uint32_t num_tokens = input.size() / last_dim;
    const uint32_t num_chunks = (d + 7) / 8;

    auto& device = mx::metal::device(stream().device);
    auto library = device.get_library("_ext", current_binary_dir());
    auto kernel = device.get_kernel(kernel_name_ + "_f16", library);

    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_output_array(out, 0);
    encoder.set_input_array(input, 1);
    encoder.set_bytes(d, 2);
    if (threshold_) {
      encoder.set_bytes(*threshold_, 3);
    }

    const size_t group_width = std::min(
        static_cast<size_t>(num_chunks),
        static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()));
    encoder.dispatch_threads(
        MTL::Size(num_chunks, num_tokens, 1), MTL::Size(group_width, 1, 1));
  }

  const char* name() const override {
    return "Activation";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& primitive = static_cast<const Activation&>(other);
    return kernel_name_ == primitive.kernel_name_ &&
        gated_ == primitive.gated_ && threshold_ == primitive.threshold_;
  }

 private:
  std::string kernel_name_;
  bool gated_;
  std::optional<float> threshold_;
};

mx::array activation(
    const mx::array& input,
    std::string kernel_name,
    bool gated,
    std::optional<float> threshold,
    mx::StreamOrDevice s) {
  if (input.ndim() == 0) {
    throw std::invalid_argument(
        "activation input must have at least one dimension.");
  }
  if (gated && input.shape(-1) % 2 != 0) {
    throw std::invalid_argument(
        "gated activations require an even-sized last dimension.");
  }
  if (input.dtype() != mx::float16) {
    throw std::invalid_argument("activation extension only supports float16.");
  }

  auto stream = mx::to_stream(s);
  auto contiguous_input = mx::contiguous(input, false, stream);
  auto output_shape = input.shape();
  if (gated) {
    output_shape.back() /= 2;
  }
  return mx::array(
      output_shape,
      input.dtype(),
      std::make_shared<Activation>(
          stream, std::move(kernel_name), gated, threshold),
      {contiguous_input});
}

mx::array silu_and_mul(const mx::array& input, mx::StreamOrDevice s = {}) {
  return activation(input, "silu_and_mul", true, std::nullopt, s);
}

mx::array fatrelu_and_mul(
    const mx::array& input,
    float threshold,
    mx::StreamOrDevice s = {}) {
  return activation(input, "fatrelu_and_mul", true, threshold, s);
}

mx::array gelu(const mx::array& input, mx::StreamOrDevice s = {}) {
  return activation(input, "gelu", false, std::nullopt, s);
}

} // namespace

NB_MODULE(_ext, m) {
  m.doc() = "Fused float16 activation kernels implemented with MLX and Metal.";
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
  m.def(
      "fatrelu_and_mul",
      &fatrelu_and_mul,
      "input"_a,
      "threshold"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def fatrelu_and_mul(input: mlx.core.array, threshold: float, *, "
          "stream: mlx.core.Stream | mlx.core.ThreadLocalStream | "
          "mlx.core.Device | None = None) -> mlx.core.array"),
      R"pbdoc(
        Apply a fused thresholded ReLU activation and multiplication.

        Splits the last dimension into equal halves ``x`` and ``y``, then
        computes ``where(x > threshold, x, 0) * y``. The input must be a
        float16 array with an even-sized last dimension.

        Args:
            input (mlx.core.array): Input array.
            threshold (float): Values of ``x`` at or below this threshold are
              replaced with zero.
            stream (Stream or Device, optional): Stream or device on which to
              schedule the operation. Defaults to ``None``.

        Returns:
            mlx.core.array: An array whose last dimension is half that of the
            input.
      )pbdoc");
  m.def(
      "gelu",
      &gelu,
      "input"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gelu(input: mlx.core.array, *, stream: mlx.core.Stream | "
          "mlx.core.ThreadLocalStream | mlx.core.Device | None = None) -> "
          "mlx.core.array"),
      R"pbdoc(
        Apply the elementwise Gaussian Error Linear Unit activation.

        Computes ``0.5 * x * (1 + erf(x / sqrt(2)))`` for a float16 input
        array. The output has the same shape and dtype as the input.

        Args:
            input (mlx.core.array): Input array.
            stream (Stream or Device, optional): Stream or device on which to
              schedule the operation. Defaults to ``None``.

        Returns:
            mlx.core.array: The activated array.
      )pbdoc");
}
