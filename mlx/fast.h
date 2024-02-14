// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core::fast {

// Custom primitive accepts a fallback function which it uses for
// transformations. Transformations are virtual so that derived classes may to
// override the default behavior
class Custom : public Primitive {
 public:
  explicit Custom(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback)
      : Primitive(stream), fallback_(fallback){};

  virtual std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
};

array rope(
    const array& x,
    int dims,
    bool traditional,
    float base,
    float scale,
    int offset,
    StreamOrDevice s /* = {} */);

class RoPE : public Custom {
 public:
  RoPE(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int dims,
      bool traditional,
      float base,
      float scale,
      int offset)
      : Custom(stream, fallback),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        offset_(offset){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(RoPE)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  int offset_;
};

} // namespace mlx::core::fast
