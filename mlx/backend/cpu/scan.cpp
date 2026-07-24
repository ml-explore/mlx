// Copyright © 2023 Apple Inc.

#include <cassert>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename AccT, typename Op>
void contiguous_scan(
    const T* input,
    U* output,
    int count,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    AccT init) {
  // The running value is kept in AccT (which is wider than U for low-precision
  // floats) and only narrowed to U on store, so that a long scan does not
  // accumulate in a type whose ULP eventually swamps the increment (e.g. a
  // float16 cumsum stalling at 2048). For every dtype where AccT == U this is
  // bit-identical to a plain in-place accumulation.
  if (!reverse) {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        AccT acc = static_cast<AccT>(*input);
        *output = static_cast<U>(acc);
        for (int j = 1; j < stride; j++) {
          input++;
          output++;
          acc = op(acc, *input);
          *output = static_cast<U>(acc);
        }
        output++;
        input++;
      }
    } else {
      for (int i = 0; i < count; i++) {
        AccT acc = init;
        *output = static_cast<U>(acc);
        for (int j = 1; j < stride; j++) {
          acc = op(acc, *input);
          *(output + 1) = static_cast<U>(acc);
          input++;
          output++;
        }
        output++;
        input++;
      }
    }
  } else {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        output += stride - 1;
        input += stride - 1;
        AccT acc = static_cast<AccT>(*input);
        *output = static_cast<U>(acc);
        for (int j = 1; j < stride; j++) {
          input--;
          output--;
          acc = op(acc, *input);
          *output = static_cast<U>(acc);
        }
        output += stride;
        input += stride;
      }
    } else {
      for (int i = 0; i < count; i++) {
        output += stride - 1;
        input += stride - 1;
        AccT acc = init;
        *output = static_cast<U>(acc);
        for (int j = 1; j < stride; j++) {
          acc = op(acc, *input);
          *(output - 1) = static_cast<U>(acc);
          input--;
          output--;
        }
        output += stride;
        input += stride;
      }
    }
  }
};

template <typename T, typename U, typename AccT, typename Op>
void strided_scan(
    const T* input,
    U* output,
    int count,
    int size,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    AccT init) {
  // TODO: Vectorize the following naive implementation
  // One running accumulator per lane, kept in AccT and narrowed to U on store
  // (see the note in contiguous_scan); bit-identical when AccT == U.
  std::vector<AccT> acc(stride);
  if (!reverse) {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        for (int k = 0; k < stride; k++) {
          acc[k] = static_cast<AccT>(input[k]);
          output[k] = static_cast<U>(acc[k]);
        }
        output += stride;
        input += stride;
        for (int j = 1; j < size; j++) {
          for (int k = 0; k < stride; k++) {
            acc[k] = op(acc[k], *input);
            *output = static_cast<U>(acc[k]);
            output++;
            input++;
          }
        }
      }
    } else {
      for (int i = 0; i < count; i++) {
        for (int k = 0; k < stride; k++) {
          acc[k] = init;
          output[k] = static_cast<U>(acc[k]);
        }
        output += stride;
        input += stride;
        for (int j = 1; j < size; j++) {
          for (int k = 0; k < stride; k++) {
            acc[k] = op(acc[k], *(input - stride));
            *output = static_cast<U>(acc[k]);
            output++;
            input++;
          }
        }
      }
    }
  } else {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        output += (size - 1) * stride;
        input += (size - 1) * stride;
        for (int k = 0; k < stride; k++) {
          acc[k] = static_cast<AccT>(input[k]);
          output[k] = static_cast<U>(acc[k]);
        }
        for (int j = 1; j < size; j++) {
          for (int k = stride - 1; k >= 0; k--) {
            output--;
            input--;
            acc[k] = op(acc[k], *input);
            *output = static_cast<U>(acc[k]);
          }
        }
        output += size * stride;
        input += size * stride;
      }
    } else {
      for (int i = 0; i < count; i++) {
        output += (size - 1) * stride;
        input += (size - 1) * stride;
        for (int k = 0; k < stride; k++) {
          acc[k] = init;
          output[k] = static_cast<U>(acc[k]);
        }
        for (int j = 1; j < size; j++) {
          for (int k = stride - 1; k >= 0; k--) {
            output--;
            input--;
            acc[k] = op(acc[k], *(input + stride));
            *output = static_cast<U>(acc[k]);
          }
        }
        output += size * stride;
        input += size * stride;
      }
    }
  }
};

template <typename T, typename U, typename AccT, typename Op>
void scan_op(
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive,
    const Op& op,
    AccT init) {
  if (in.flags().row_contiguous) {
    if (in.strides()[axis] == 1) {
      contiguous_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis),
          in.shape(axis),
          reverse,
          inclusive,
          op,
          init);
    } else {
      strided_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis) / in.strides()[axis],
          in.shape(axis),
          in.strides()[axis],
          reverse,
          inclusive,
          op,
          init);
    }
  } else {
    throw std::runtime_error("Scan op supports only contiguous inputs");
  }
}

template <typename T, typename U, typename AccT = U>
void scan_dispatch(
    Scan::ReduceType rtype,
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive) {
  switch (rtype) {
    case Scan::Sum: {
      auto op = [](AccT y, T x) { return y + static_cast<AccT>(x); };
      auto init = static_cast<AccT>(0);
      scan_op<T, U, AccT>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Prod: {
      auto op = [](AccT y, T x) { return y * static_cast<AccT>(x); };
      auto init = static_cast<AccT>(1);
      scan_op<T, U, AccT>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Min: {
      auto op = [](AccT y, T x) {
        auto xa = static_cast<AccT>(x);
        return xa < y ? xa : y;
      };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<AccT>(std::numeric_limits<float>::infinity())
          : std::numeric_limits<AccT>::max();
      scan_op<T, U, AccT>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Max: {
      auto op = [](AccT y, T x) {
        auto xa = static_cast<AccT>(x);
        return xa < y ? y : xa;
      };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<AccT>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<AccT>::min();
      scan_op<T, U, AccT>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::LogAddExp: {
      auto op = [](AccT a, T b) {
        return detail::LogAddExp{}(a, static_cast<AccT>(b));
      };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<AccT>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<AccT>::min();
      scan_op<T, U, AccT>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
  }
}

} // namespace

void Scan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& encoder = cpu::get_command_encoder(stream());

  // Ensure contiguity
  auto in = inputs[0];
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_cpu(in, stream());
    encoder.add_temporary(in);
  }
  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    reduce_type_ = reduce_type_,
                    reverse_ = reverse_,
                    inclusive_ = inclusive_]() mutable {
    switch (in.dtype()) {
      case bool_: {
        // We could do a full dtype x dtype switch but this is the only case
        // where we accumulate in a different type, for now.
        //
        // TODO: If we add the option to accumulate floats in higher precision
        //       floats perhaps we should add the full all-to-all dispatch.
        if (reduce_type_ == Scan::Sum && out.dtype() == int32) {
          scan_dispatch<bool, int32_t>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        } else {
          scan_dispatch<bool, bool>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        }
        break;
      }
      case uint8:
        scan_dispatch<uint8_t, uint8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint16:
        scan_dispatch<uint16_t, uint16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint32:
        scan_dispatch<uint32_t, uint32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint64:
        scan_dispatch<uint64_t, uint64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int8:
        scan_dispatch<int8_t, int8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int16:
        scan_dispatch<int16_t, int16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int32:
        scan_dispatch<int32_t, int32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int64:
        scan_dispatch<int64_t, int64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float16:
        // Accumulate low-precision floats in float32 so a long scan does not
        // stall once the running value's ULP exceeds the increment (a float16
        // cumsum of ones otherwise saturates at 2048, disagreeing with the
        // GPU).
        scan_dispatch<float16_t, float16_t, float>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float32:
        scan_dispatch<float, float>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float64:
        scan_dispatch<double, double>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case bfloat16:
        scan_dispatch<bfloat16_t, bfloat16_t, float>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case complex64:
        scan_dispatch<complex64_t, complex64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
    }
  });
}

} // namespace mlx::core
