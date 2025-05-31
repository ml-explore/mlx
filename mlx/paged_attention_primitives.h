// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI in MSVC.
#define _USE_MATH_DEFINES

#include <optional>

#include "mlx/primitives.h"

namespace mlx::core::paged_attention {

class PagedAttention : public UnaryPrimitive {
 public:
  explicit PagedAttention(
      Stream stream,
      bool use_v1,
      int max_context_len,
      float softmax_scale,
      std::optional<float> softcapping = std::nullopt)
      : UnaryPrimitive(stream),
        use_v1_(use_v1),
        max_context_len_(max_context_len),
        softmax_scale_(softmax_scale),
        softcapping_(softcapping) {}

  void eval_cpu(const std::vector<array>& inputs, array& outputs) override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, array& outputs) override;

  DEFINE_PRINT(PagedAttention);

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(max_context_len_, softmax_scale_, softcapping_);
  }

 private:
  bool use_v1_;
  int max_context_len_;
  float softmax_scale_;
  std::optional<float> softcapping_ = std::nullopt;
};

} // namespace mlx::core::paged_attention