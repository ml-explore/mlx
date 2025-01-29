// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include <Accelerate/Accelerate.h>

#include "mlx/allocator.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/unary.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Scan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (reduce_type_ == Scan::Sum && out.dtype() == float32 &&
      in.flags().row_contiguous && in.strides()[axis_] == 1 && !inclusive_) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    int stride = in.shape(axis_);
    int count = in.size() / stride;
    const float* input = in.data<float>();
    float* output = out.data<float>();
    float s = 1.0;
    if (!reverse_) {
      for (int i = 0; i < count; i++) {
        vDSP_vrsum(input - 1, 1, &s, output, 1, stride);
        input += stride;
        output += stride;
      }
    } else {
      for (int i = 0; i < count; i++) {
        input += stride - 1;
        output += stride - 1;
        vDSP_vrsum(input + 1, -1, &s, output, -1, stride);
        input++;
        output++;
      }
    }
  } else {
    eval(inputs, out);
  }
}

} // namespace mlx::core
