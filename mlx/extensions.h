// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core::ext {

class Extended : public Primitive {
 public:
  explicit Extended(Stream stream)
      : Primitive(stream) {} //, std::function<> fallback);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_GRADS()

 private:
  //  std::function<std::vector<array>(std::vector<array>);
};

array rms_norm(
    array x,
    const array& w,
    float eps = 1e-5,
    bool precise = false,
    StreamOrDevice s = {});

} // namespace mlx::core::ext
