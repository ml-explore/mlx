// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>

#include <vecLib/vDSP.h>
#include <vecLib/vForce.h>

#include "mlx/allocator.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/unary.h"
#include "mlx/primitives.h"

#define DEFAULT(primitive)                                                 \
  void primitive::eval_cpu(const std::vector<array>& inputs, array& out) { \
    primitive::eval(inputs, out);                                          \
  }

namespace mlx::core {

// Use the default implementation for the following primitives
DEFAULT(Arange)
DEFAULT(ArgPartition)
DEFAULT(ArgReduce)
DEFAULT(ArgSort)
DEFAULT(AsStrided)
DEFAULT(Broadcast)
DEFAULT(Ceil)
DEFAULT(Concatenate)
DEFAULT(Copy)
DEFAULT(Equal)
DEFAULT(Erf)
DEFAULT(ErfInv)
DEFAULT(FFT)
DEFAULT(Floor)
DEFAULT(Gather)
DEFAULT(Greater)
DEFAULT(GreaterEqual)
DEFAULT(Less)
DEFAULT(LessEqual)
DEFAULT(Load)
DEFAULT(LogicalNot)
DEFAULT(LogAddExp)
DEFAULT(NotEqual)
DEFAULT(Pad)
DEFAULT(Partition)
DEFAULT(RandomBits)
DEFAULT(Reshape)
DEFAULT(Round)
DEFAULT(Scatter)
DEFAULT(Sigmoid)
DEFAULT(Sign)
DEFAULT(Slice)
DEFAULT(Sort)
DEFAULT(StopGradient)
DEFAULT(Transpose)

void Abs::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vDSP_vabs(in.data<float>(), 1, out.data<float>(), 1, size);
  } else if (in.dtype() == int32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vDSP_vabsi(in.data<int>(), 1, out.data<int>(), 1, size);
  } else if (is_unsigned(in.dtype())) {
    // No-op for unsigned types
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, AbsOp());
  }
}

void Add::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary(
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
    binary(
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
    binary(a, b, out, [](auto x, auto y) { return x + y; });
  }
}

void ArcCos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvacosf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void ArcCosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvacoshf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void ArcSin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvasinf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void ArcSinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvasinhf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void ArcTan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvatanf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void ArcTanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvatanhf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void AsType::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  if (in.flags().contiguous) {
    auto allocfn = [&in, &out]() {
      out.set_data(
          allocator::malloc_or_wait(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    };
    // Use accelerate functions if possible
    if (in.dtype() == float32 && out.dtype() == uint32) {
      allocfn();
      vDSP_vfixu32(
          in.data<float>(), 1, out.data<uint32_t>(), 1, in.data_size());
      return;
    } else if (in.dtype() == float32 && out.dtype() == int32) {
      allocfn();
      vDSP_vfix32(in.data<float>(), 1, out.data<int32_t>(), 1, in.data_size());
      return;
    } else if (in.dtype() == uint32 && out.dtype() == float32) {
      allocfn();
      vDSP_vfltu32(
          in.data<uint32_t>(), 1, out.data<float>(), 1, in.data_size());
      return;
    } else if (in.dtype() == int32 && out.dtype() == float32) {
      allocfn();
      vDSP_vflt32(in.data<int32_t>(), 1, out.data<float>(), 1, in.data_size());
      return;
    }
  }
  eval(inputs, out);
}

void Cos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvcosf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void Cosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvcoshf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == int32) {
    binary(
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
    binary(
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
    binary(a, b, out, [](auto x, auto y) { return x / y; });
  }
}

// TODO: Avoid code duplication with the common backend.
struct RemainderFn {
  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(
      T numerator,
      T denominator) {
    return std::fmod(numerator, denominator);
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(
      T numerator,
      T denominator) {
    return numerator % denominator;
  }
};

void Remainder::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary(
        a,
        b,
        out,
        RemainderFn{},
        UseDefaultBinaryOp(),
        UseDefaultBinaryOp(),
        [](const auto* a, const auto* b, auto* o, auto n) {
          int num_el = n;
          vvremainderf((float*)o, (const float*)a, (const float*)b, &num_el);
        });
  } else {
    binary(a, b, out, RemainderFn{});
  }
}

void Exp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvexpf(out.data<float>(), in.data<float>(), reinterpret_cast<int*>(&size));
  } else if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::exp(x); });
  } else {
    throw std::invalid_argument(
        "[exp] Cannot exponentiate elements in array"
        " with non floating point type.");
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

void Log::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    switch (base_) {
      case Base::e:
        vvlogf(
            out.data<float>(), in.data<float>(), reinterpret_cast<int*>(&size));
        break;
      case Base::two:
        vvlog2f(
            out.data<float>(), in.data<float>(), reinterpret_cast<int*>(&size));
        break;
      case Base::ten:
        vvlog10f(
            out.data<float>(), in.data<float>(), reinterpret_cast<int*>(&size));
        break;
    }
  } else {
    eval(inputs, out);
  }
}

void Log1p::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvlog1pf(
        out.data<float>(), in.data<float>(), reinterpret_cast<int*>(&size));
  } else if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::log1p(x); });
  } else {
    throw std::invalid_argument(
        "[log1p] Cannot compute log of elements in array with"
        " non floating point type.");
  }
}

void Maximum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (out.dtype() == float32) {
    binary(
        a,
        b,
        out,
        [](auto x, auto y) { return (x > y) ? x : y; },
        UseDefaultBinaryOp(),
        UseDefaultBinaryOp(),
        [](const auto* a, const auto* b, auto* out, int n) {
          vDSP_vmax((const float*)a, 1, (const float*)b, 1, (float*)out, 1, n);
        });
  } else {
    binary(a, b, out, [](auto x, auto y) { return (x > y) ? x : y; });
  }
}

void Minimum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (out.dtype() == float32) {
    binary(
        a,
        b,
        out,
        [](auto x, auto y) { return (x < y) ? x : y; },
        UseDefaultBinaryOp(),
        UseDefaultBinaryOp(),
        [](const auto* a, const auto* b, auto* out, int n) {
          vDSP_vmin((const float*)a, 1, (const float*)b, 1, (float*)out, 1, n);
        });
  } else {
    binary(a, b, out, [](auto x, auto y) { return (x < y) ? x : y; });
  }
}

void Multiply::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];

  if (a.dtype() == float32) {
    binary(
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
    binary(a, b, out, [](auto x, auto y) { return x * y; });
  }
}

void Negative::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vDSP_vneg(in.data<float>(), 1, out.data<float>(), 1, size);
  } else {
    unary(in, out, [](auto x) { return -x; });
  }
}

void Power::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (out.dtype() == float32 && a.flags().row_contiguous &&
      b.flags().row_contiguous) {
    int size = a.size();
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
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

void Sin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvsinf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void Sinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvsinhf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void Square::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == float32 && in.flags().contiguous) {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vDSP_vsq(in.data<float>(), 1, out.data<float>(), 1, size);
  } else {
    unary(in, out, [](auto x) { return x * x; });
  }
}

void Sqrt::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    if (recip_) {
      vvrsqrtf(out.data<float>(), in.data<float>(), &size);
    } else {
      vvsqrtf(out.data<float>(), in.data<float>(), &size);
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
    binary(
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
    binary(
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
    binary(a, b, out, [](auto x, auto y) { return x - y; });
  }
}

void Tan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvtanf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

void Tanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == float32 && in.flags().contiguous) {
    int size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
    vvtanhf(out.data<float>(), in.data<float>(), &size);
  } else {
    eval(inputs, out);
  }
}

} // namespace mlx::core
