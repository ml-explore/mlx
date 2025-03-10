// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/ternary.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

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
        ternary_op<bool, bool, bool, bool>(a, b, c, out, op, topt);
        break;
      case uint8:
        ternary_op<bool, uint8_t, uint8_t, uint8_t>(a, b, c, out, op, topt);
        break;
      case uint16:
        ternary_op<bool, uint16_t, uint16_t, uint16_t>(a, b, c, out, op, topt);
        break;
      case uint32:
        ternary_op<bool, uint32_t, uint32_t, uint32_t>(a, b, c, out, op, topt);
        break;
      case uint64:
        ternary_op<bool, uint64_t, uint64_t, uint64_t>(a, b, c, out, op, topt);
        break;
      case int8:
        ternary_op<bool, int8_t, int8_t, int8_t>(a, b, c, out, op, topt);
        break;
      case int16:
        ternary_op<bool, int16_t, int16_t, int16_t>(a, b, c, out, op, topt);
        break;
      case int32:
        ternary_op<bool, int32_t, int32_t, int32_t>(a, b, c, out, op, topt);
        break;
      case int64:
        ternary_op<bool, int64_t, int64_t, int64_t>(a, b, c, out, op, topt);
        break;
      case float16:
        ternary_op<bool, float16_t, float16_t, float16_t>(
            a, b, c, out, op, topt);
        break;
      case float32:
        ternary_op<bool, float, float, float>(a, b, c, out, op, topt);
        break;
      case float64:
        ternary_op<bool, double, double, double>(a, b, c, out, op, topt);
        break;
      case bfloat16:
        ternary_op<bool, bfloat16_t, bfloat16_t, bfloat16_t>(
            a, b, c, out, op, topt);
        break;
      case complex64:
        ternary_op<bool, complex64_t, complex64_t, complex64_t>(
            a, b, c, out, op, topt);
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
