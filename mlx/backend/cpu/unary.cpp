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
    auto op = detail::Abs{};
    switch (out.dtype()) {
      case int8:
        unary_op<int8_t>(in, out, op);
        break;
      case int16:
        unary_op<int16_t>(in, out, op);
        break;
      case int32:
        unary_op<int32_t>(in, out, op);
        break;
      case int64:
        unary_op<int64_t>(in, out, op);
        break;
      case float16:
        unary_op<float16_t>(in, out, op);
        break;
      case float32:
        unary_op<float>(in, out, op);
        break;
      case float64:
        unary_op<double>(in, out, op);
        break;
      case bfloat16:
        unary_op<bfloat16_t>(in, out, op);
        break;
      case complex64:
        unary_op<complex64_t>(in, out, op);
        break;
      default:
        throw std::runtime_error("[Abs] Called on unsigned type");
    }
  }
}

void ArcCos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcCos());
}

void ArcCosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcCosh());
}

void ArcSin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcSin());
}

void ArcSinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcSinh());
}

void ArcTan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcTan());
}

void ArcTanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::ArcTanh());
}

void BitwiseInverse::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_int(in, out, detail::BitwiseInverse());
}

void Ceil::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Ceil());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Conjugate::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  unary_op<complex64_t>(inputs[0], out, detail::Conjugate());
}

void Cos::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Cos());
}

void Cosh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Cosh());
}

void Erf::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (out.dtype()) {
    case float32:
      unary_op<float>(in, out, detail::Erf());
      break;
    case float16:
      unary_op<float16_t>(in, out, detail::Erf());
      break;
    case float64:
      unary_op<double>(in, out, detail::Erf());
      break;
    case bfloat16:
      unary_op<bfloat16_t>(in, out, detail::Erf());
      break;
    default:
      throw std::invalid_argument(
          "[erf] Error function only defined for arrays"
          " with real floating point type.");
  }
}

void ErfInv::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (out.dtype()) {
    case float32:
      unary_op<float>(in, out, detail::ErfInv());
      break;
    case float16:
      unary_op<float16_t>(in, out, detail::ErfInv());
      break;
    case float64:
      unary_op<double>(in, out, detail::ErfInv());
      break;
    case bfloat16:
      unary_op<bfloat16_t>(in, out, detail::ErfInv());
      break;
    default:
      throw std::invalid_argument(
          "[erf_inv] Inverse error function only defined for arrays"
          " with real floating point type.");
  }
}

void Exp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Exp());
}

void Expm1::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Expm1());
}

void Floor::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Floor());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Imag::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_op<complex64_t, float>(inputs[0], out, detail::Imag());
}

void Log::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (base_) {
    case Base::e:
      unary_fp(in, out, detail::Log());
      break;
    case Base::two:
      unary_fp(in, out, detail::Log2());
      break;
    case Base::ten:
      unary_fp(in, out, detail::Log10());
      break;
  }
}

void Log1p::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Log1p());
}

void LogicalNot::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::LogicalNot());
}

void Negative::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Negative());
}

void Real::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_op<complex64_t, float>(inputs[0], out, detail::Real());
}

void Round::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Round());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sigmoid::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sigmoid());
}

void Sign::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == bool_) {
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, detail::Sign());
  }
}

void Sin::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sin());
}

void Sinh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Sinh());
}

void Square::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Square());
}

void Sqrt::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (recip_) {
    unary_fp(in, out, detail::Rsqrt());
  } else {
    unary_fp(in, out, detail::Sqrt());
  }
}

void Tan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Tan());
}

void Tanh::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  unary_fp(in, out, detail::Tanh());
}

} // namespace mlx::core
