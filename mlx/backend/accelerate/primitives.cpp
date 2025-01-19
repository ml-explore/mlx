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

void Add::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary_op<float>(
        a,
        b,
        out,
        [](auto x, auto y) { return x + y; },
        [](const auto* s, const auto* vec, auto* o, auto n) {
          vDSP_vsadd((const float*)vec, 1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* vec, const auto* s, auto* o, auto n) {
          vDSP_vsadd((const float*)vec, 1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vadd((const float*)a, 1, (const float*)b, 1, (float*)o, 1, n);
        });
  } else if (a.dtype() == int32) {
    binary_op<int>(
        a,
        b,
        out,
        [](auto x, auto y) { return x + y; },
        [](const auto* s, const auto* vec, auto* o, auto n) {
          vDSP_vsaddi((const int*)vec, 1, (const int*)s, (int*)o, 1, n);
        },
        [](const auto* vec, const auto* s, auto* o, auto n) {
          vDSP_vsaddi((const int*)vec, 1, (const int*)s, (int*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vaddi((const int*)a, 1, (const int*)b, 1, (int*)o, 1, n);
        });
  } else {
    eval(inputs, out);
  }
}

void ArcTan2::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (out.dtype() == float32 && a.flags().row_contiguous &&
      b.flags().row_contiguous) {
    if (a.is_donatable()) {
      out.copy_shared_buffer(a);
    } else if (b.is_donatable()) {
      out.copy_shared_buffer(b);
    } else {
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
    }
    int size = a.data_size();
    vvatan2f(out.data<float>(), a.data<float>(), b.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void AsType::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  if (in.flags().contiguous) {
    // Use accelerate functions if possible
    if (in.dtype() == float32 && out.dtype() == uint32) {
      set_unary_output_data(in, out);
      vDSP_vfixu32(
          in.data<float>(), 1, out.data<uint32_t>(), 1, in.data_size());
      return;
    } else if (in.dtype() == float32 && out.dtype() == int32) {
      set_unary_output_data(in, out);
      vDSP_vfix32(in.data<float>(), 1, out.data<int32_t>(), 1, in.data_size());
      return;
    } else if (in.dtype() == uint32 && out.dtype() == float32) {
      set_unary_output_data(in, out);
      vDSP_vfltu32(
          in.data<uint32_t>(), 1, out.data<float>(), 1, in.data_size());
      return;
    } else if (in.dtype() == int32 && out.dtype() == float32) {
      set_unary_output_data(in, out);
      vDSP_vflt32(in.data<int32_t>(), 1, out.data<float>(), 1, in.data_size());
      return;
    }
  }
  eval(inputs, out);
}

void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == int32) {
    binary_op<int>(
        a,
        b,
        out,
        [](auto x, auto y) { return x / y; },
        UseDefaultBinaryOp(),
        [](const auto* vec, const auto* s, auto* o, auto n) {
          vDSP_vsdivi((const int*)vec, 1, (const int*)s, (int*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vdivi((const int*)b, 1, (const int*)a, 1, (int*)o, 1, n);
        });
  } else if (a.dtype() == float32) {
    binary_op<float>(
        a,
        b,
        out,
        [](auto x, auto y) { return x / y; },
        [](const auto* s, const auto* vec, auto* o, auto n) {
          vDSP_svdiv((const float*)s, (const float*)vec, 1, (float*)o, 1, n);
        },
        [](const auto* vec, const auto* s, auto* o, auto n) {
          vDSP_vsdiv((const float*)vec, 1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vdiv((const float*)b, 1, (const float*)a, 1, (float*)o, 1, n);
        });
  } else {
    eval(inputs, out);
  }
}

void Full::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  assert(in.dtype() == out.dtype());
  if (in.data_size() == 1 && out.dtype() == float32) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    vDSP_vfill(in.data<float>(), out.data<float>(), 1, out.size());
  } else {
    eval(inputs, out);
  }
}

void Multiply::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary_op<float>(
        a,
        b,
        out,
        [](auto x, auto y) { return x * y; },
        [](const auto* s, const auto* vec, auto* o, auto n) {
          vDSP_vsmul((const float*)vec, 1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* vec, const auto* s, auto* o, auto n) {
          vDSP_vsmul((const float*)vec, 1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vmul((const float*)a, 1, (const float*)b, 1, (float*)o, 1, n);
        });
  } else {
    eval(inputs, out);
  }
}

void Power::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (out.dtype() == float32 && a.flags().row_contiguous &&
      b.flags().row_contiguous) {
    int size = a.size();
    if (a.is_donatable() && a.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(a);
    } else if (b.is_donatable() && b.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(b);
    } else {
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
    }
    vvpowf(out.data<float>(), b.data<float>(), a.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

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

void Subtract::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary_op<float>(
        a,
        b,
        out,
        [](auto x, auto y) { return x - y; },
        [](const auto* s, const auto* vec, auto* o, auto n) {
          float minus_1 = -1;
          vDSP_vsmsa(
              (const float*)vec, 1, &minus_1, (const float*)s, (float*)o, 1, n);
        },
        [](const auto* vec, const auto* s, auto* o, auto n) {
          float val = -(*s);
          vDSP_vsadd((const float*)vec, 1, &val, (float*)o, 1, n);
        },
        [](const auto* a, const auto* b, auto* o, auto n) {
          vDSP_vsub((const float*)b, 1, (const float*)a, 1, (float*)o, 1, n);
        });
  } else if (a.dtype() == int32) {
    binary_op<int>(
        a,
        b,
        out,
        [](auto x, auto y) { return x - y; },
        UseDefaultBinaryOp(),
        [](const auto* vec, const auto* s, auto* o, auto n) {
          int val = -(*s);
          vDSP_vsaddi((const int*)vec, 1, &val, (int*)o, 1, n);
        },
        UseDefaultBinaryOp());
  } else {
    eval(inputs, out);
  }
}

} // namespace mlx::core
