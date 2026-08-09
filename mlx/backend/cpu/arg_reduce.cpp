// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename InT, typename OpT>
void arg_reduce(const array& in, array& out, const OpT& op, int axis) {
  auto axis_size = in.shape()[axis];
  auto axis_stride = in.strides()[axis];
  Strides strides = remove_index(in.strides(), axis);
  Shape shape = remove_index(in.shape(), axis);
  auto in_ptr = in.data<InT>();
  auto out_ptr = out.data<uint32_t>();

  for (uint32_t i = 0; i < out.size(); ++i) {
    auto loc = elem_to_loc(i, shape, strides);
    auto local_in_ptr = in_ptr + loc;
    uint32_t ind_v = 0;
    InT v = (*local_in_ptr);
    for (uint32_t j = 0; j < axis_size; ++j, local_in_ptr += axis_stride) {
      op(j, (*local_in_ptr), &ind_v, &v);
    }
    out_ptr[i] = ind_v;
  }
}

template <typename InT>
void arg_reduce_dispatch(
    const array& in,
    array& out,
    ArgReduce::ReduceType rtype,
    int axis) {
  switch (rtype) {
    case ArgReduce::ArgMin: {
      auto op = [](auto ind_x, auto x, auto ind_y, auto y) {
        if (x < (*y)) {
          (*y) = x;
          (*ind_y) = ind_x;
        }
      };
      arg_reduce<InT>(in, out, op, axis);
      break;
    }
    case ArgReduce::ArgMax: {
      auto op = [](auto ind_x, auto x, auto ind_y, auto y) {
        if (x > (*y)) {
          (*y) = x;
          (*ind_y) = ind_x;
        }
      };
      arg_reduce<InT>(in, out, op, axis);
      break;
    }
  }
}

} // namespace

void ArgReduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    reduce_type_ = reduce_type_,
                    axis_ = axis_]() mutable {
    dispatch_all_types(in.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      arg_reduce_dispatch<T>(in, out, reduce_type_, axis_);
    });
  });
}

} // namespace mlx::core
