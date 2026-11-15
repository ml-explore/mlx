// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/ternary.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// where(cond, x, y) on contiguous inputs, as a bitwise blend of same-size
// unsigned integers. GCC doesn't vectorize `cond[i] ? x[i] : y[i]` on x86
// (it can't widen a bool load to a 32-bit mask), so the plain loop ran one
// element at a time. This form vectorizes with GCC and clang, and copies the
// selected bits exactly, like the ternary does.
template <typename T>
MLX_CPU_NOINLINE void select_contiguous(
    const bool* cond,
    const T* x,
    const T* y,
    T* out,
    int64_t size) {
  using U = std::conditional_t<
      sizeof(T) == 1,
      uint8_t,
      std::conditional_t<
          sizeof(T) == 2,
          uint16_t,
          std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
  static_assert(sizeof(U) == sizeof(T));
  const auto* c = reinterpret_cast<const uint8_t*>(cond);
  for (int64_t i = 0; i < size; ++i) {
    U xi, yi;
    std::memcpy(&xi, x + i, sizeof(U));
    std::memcpy(&yi, y + i, sizeof(U));
    U mask = U(0) - U(c[i] != 0);
    U r = U((xi & mask) | (yi & U(~mask)));
    std::memcpy(out + i, &r, sizeof(U));
  }
}

template <typename T, typename Op>
void select_dispatch(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op,
    TernaryOpType topt) {
  constexpr bool bitwise = std::is_trivially_copyable_v<T> &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  if constexpr (bitwise) {
    if (topt == TernaryOpType::VectorVectorVector) {
      const bool* cond = a.data<bool>();
      const T* x = b.data<T>();
      const T* y = c.data<T>();
      T* dst = out.data<T>();
      cpu::parallel_for(
          out.size(),
          cpu::elementwise_grain<T>(),
          [&](int64_t begin, int64_t end) {
            select_contiguous<T>(
                cond + begin, x + begin, y + begin, dst + begin, end - begin);
          });
      return;
    }
  }
  ternary_op<bool, T, T, T>(a, b, c, out, op, topt);
}

template <typename Op>
void select_op(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op,
    Stream stream) {
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    c = array::unsafe_weak_copy(c),
                    out = array::unsafe_weak_copy(out),
                    op,
                    topt]() mutable {
    switch (out.dtype()) {
      case bool_:
        select_dispatch<bool>(a, b, c, out, op, topt);
        break;
      case uint8:
        select_dispatch<uint8_t>(a, b, c, out, op, topt);
        break;
      case uint16:
        select_dispatch<uint16_t>(a, b, c, out, op, topt);
        break;
      case uint32:
        select_dispatch<uint32_t>(a, b, c, out, op, topt);
        break;
      case uint64:
        select_dispatch<uint64_t>(a, b, c, out, op, topt);
        break;
      case int8:
        select_dispatch<int8_t>(a, b, c, out, op, topt);
        break;
      case int16:
        select_dispatch<int16_t>(a, b, c, out, op, topt);
        break;
      case int32:
        select_dispatch<int32_t>(a, b, c, out, op, topt);
        break;
      case int64:
        select_dispatch<int64_t>(a, b, c, out, op, topt);
        break;
      case float16:
        select_dispatch<float16_t>(a, b, c, out, op, topt);
        break;
      case float32:
        select_dispatch<float>(a, b, c, out, op, topt);
        break;
      case float64:
        select_dispatch<double>(a, b, c, out, op, topt);
        break;
      case bfloat16:
        select_dispatch<bfloat16_t>(a, b, c, out, op, topt);
        break;
      case complex64:
        select_dispatch<complex64_t>(a, b, c, out, op, topt);
        break;
    }
  });
}

} // namespace

void Select::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  const auto& condition = inputs[0];
  const auto& a = inputs[1];
  const auto& b = inputs[2];
  select_op(condition, a, b, out, detail::Select(), stream());
}

} // namespace mlx::core
