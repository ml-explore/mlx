// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/backend/cpu/unary.h"
#include "mlx/backend/cpu/unary_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Abs::eval_cpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), unsignedinteger) || in.dtype() == bool_) {
    // No-op for unsigned types
    out.copy_shared_buffer(in);
  } else {
    unary_signed(in, out, detail::Abs(), stream());
  }
}

void ArcCos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcCos(), stream());
}

void ArcCosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcCosh(), stream());
}

void ArcSin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcSin(), stream());
}

void ArcSinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcSinh(), stream());
}

void ArcTan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcTan(), stream());
}

void ArcTanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcTanh(), stream());
}

void BitwiseInvert::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_int(in, out, detail::BitwiseInvert(), stream());
}

void Ceil::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Ceil(), stream());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Conjugate::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  unary_complex(inputs[0], out, detail::Conjugate(), stream());
}

void Cos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Cos(), stream());
}

void Cosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Cosh(), stream());
}

void Erf::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_real_fp(in, out, detail::Erf(), stream());
}

void ErfInv::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_real_fp(in, out, detail::ErfInv(), stream());
}

void Exp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Exp(), stream());
}

void Expm1::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Expm1(), stream());
}

void Floor::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Floor(), stream());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Imag::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_complex_to_float(inputs[0], out, detail::Imag(), stream());
}

void Log::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (base_) {
    case Base::e:
      unary_fp(in, out, detail::Log(), stream());
      break;
    case Base::two:
      unary_fp(in, out, detail::Log2(), stream());
      break;
    case Base::ten:
      unary_fp(in, out, detail::Log10(), stream());
      break;
  }
}

void Log1p::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Log1p(), stream());
}

void LogicalNot::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::LogicalNot(), stream());
}

void Negative::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Negative(), stream());
}

void Real::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_complex_to_float(inputs[0], out, detail::Real(), stream());
}

void Round::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Round(), stream());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sigmoid::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sigmoid(), stream());
}

void Sign::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == bool_) {
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, detail::Sign(), stream());
  }
}

void Sin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sin(), stream());
}

void Sinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sinh(), stream());
}

void Square::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Square(), stream());
}

void Sqrt::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (recip_) {
    unary_fp(in, out, detail::Rsqrt(), stream());
  } else {
    unary_fp(in, out, detail::Sqrt(), stream());
  }
}

void Tan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Tan(), stream());
}

void Tanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Tanh(), stream());
}

} // namespace mlx::core
