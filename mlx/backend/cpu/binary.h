// Copyright © 2023 Apple Inc.

#pragma once
#include <cassert>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"

#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core {

template <typename Op>
struct VectorScalar {
  Op op;

  VectorScalar(Op op_) : op(op_) {}

  template <typename T, typename U>
  void operator()(const T* a, const T* b, U* dst, int size) {
    T scalar = *b;
    constexpr int N = simd::max_size<T>;
    while (size >= N) {
      simd::store(dst, op(simd::load<T, N>(a), simd::Simd<T, N>(scalar)));
      dst += N;
      a += N;
      size -= N;
    }
    while (size-- > 0) {
      *dst = op(*a, scalar);
      dst++;
      a++;
    }
  }
};

template <typename Op>
struct ScalarVector {
  Op op;

  ScalarVector(Op op_) : op(op_) {}

  template <typename T, typename U>
  void operator()(const T* a, const T* b, U* dst, int size) {
    T scalar = *a;
    constexpr int N = simd::max_size<T>;
    while (size >= N) {
      simd::store(dst, op(simd::Simd<T, N>(scalar), simd::load<T, N>(b)));
      dst += N;
      b += N;
      size -= N;
    }
    while (size-- > 0) {
      *dst = op(scalar, *b);
      dst++;
      b++;
    }
  }
};

template <typename Op>
struct VectorVector {
  Op op;

  VectorVector(Op op_) : op(op_) {}

  template <typename T, typename U>
  void operator()(const T* a, const T* b, U* dst, int size) {
    constexpr int N = simd::max_size<T>;
    while (size >= N) {
      simd::store(dst, op(simd::load<T, N>(a), simd::load<T, N>(b)));
      dst += N;
      a += N;
      b += N;
      size -= N;
    }
    while (size-- > 0) {
      *dst = op(*a, *b);
      dst++;
      a++;
      b++;
    }
  }
};

template <typename T, typename U, typename Op, int D, bool Strided>
void binary_op_dims(
    const T* a,
    const T* b,
    U* out,
    Op op,
    const Shape& shape,
    const Strides& a_strides,
    const Strides& b_strides,
    const Strides& out_strides,
    int axis) {
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      binary_op_dims<T, U, Op, D - 1, Strided>(
          a, b, out, op, shape, a_strides, b_strides, out_strides, axis + 1);
    } else {
      if constexpr (Strided) {
        op(a, b, out, stride_out);
      } else {
        *out = op(*a, *b);
      }
    }
    out += stride_out;
    a += stride_a;
    b += stride_b;
  }
}

template <typename T, typename U, bool Strided, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out,
    Op op,
    int dim,
    const Shape& shape,
    const Strides& a_strides,
    const Strides& b_strides,
    const Strides& out_strides) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* out_ptr = out.data<U>();
  switch (dim) {
    case 1:
      binary_op_dims<T, U, Op, 1, Strided>(
          a_ptr,
          b_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          0);
      return;
    case 2:
      binary_op_dims<T, U, Op, 2, Strided>(
          a_ptr,
          b_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          0);
      return;
    case 3:
      binary_op_dims<T, U, Op, 3, Strided>(
          a_ptr,
          b_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          0);
      return;
  }

  ContiguousIterator a_it(shape, a_strides, dim - 3);
  ContiguousIterator b_it(shape, b_strides, dim - 3);
  auto stride = out_strides[dim - 4];
  for (int64_t elem = 0; elem < a.size(); elem += stride) {
    binary_op_dims<T, U, Op, 3, Strided>(
        a_ptr + a_it.loc,
        b_ptr + b_it.loc,
        out_ptr + elem,
        op,
        shape,
        a_strides,
        b_strides,
        out_strides,
        dim - 3);
    a_it.step();
    b_it.step();
  }
}

template <typename T, typename U, typename Op>
void binary_op(const array& a, const array& b, array& out, Op op) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  // The full computation is scalar scalar so call the base op once
  if (bopt == BinaryOpType::ScalarScalar) {
    *(out.data<U>()) = op(*a.data<T>(), *b.data<T>());
    return;
  }

  // The full computation is scalar vector so delegate to the op
  if (bopt == BinaryOpType::ScalarVector) {
    ScalarVector{op}(a.data<T>(), b.data<T>(), out.data<U>(), b.data_size());
    return;
  }

  // The full computation is vector scalar so delegate to the op
  if (bopt == BinaryOpType::VectorScalar) {
    VectorScalar{op}(a.data<T>(), b.data<T>(), out.data<U>(), a.data_size());
    return;
  }

  // The full computation is vector vector so delegate to the op
  if (bopt == BinaryOpType::VectorVector) {
    VectorVector{op}(a.data<T>(), b.data<T>(), out.data<U>(), out.size());
    return;
  }

  // General computation so let's try to optimize
  auto [new_shape, new_strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), out.strides()});
  const auto& a_strides = new_strides[0];
  const auto& b_strides = new_strides[1];
  const auto& strides = new_strides[2];

  // Get the left-most dim such that the array is row contiguous after
  auto leftmost_rc_dim = [&strides](const auto& arr_strides) {
    int d = arr_strides.size() - 1;
    for (; d >= 0 && arr_strides[d] == strides[d]; d--) {
    }
    return d + 1;
  };
  auto a_rc_dim = leftmost_rc_dim(a_strides);
  auto b_rc_dim = leftmost_rc_dim(b_strides);

  // Get the left-most dim such that the array is a broadcasted "scalar" after
  auto leftmost_s_dim = [](const auto& arr_strides) {
    int d = arr_strides.size() - 1;
    for (; d >= 0 && arr_strides[d] == 0; d--) {
    }
    return d + 1;
  };
  auto a_s_dim = leftmost_s_dim(a_strides);
  auto b_s_dim = leftmost_s_dim(b_strides);

  auto ndim = new_shape.size();

  // Case 1: LxM and FxM where L and F are broadcastable and M is row contiguous
  int dim = ndim;
  if (int d = std::max(a_rc_dim, b_rc_dim); d < ndim) {
    bopt = BinaryOpType::VectorVector;
    dim = d;
    // Case 2: LxM and Fx1 where L and F are broadcastable and M is row
    // contiguous
  } else if (int d = std::max(a_rc_dim, b_s_dim); d < ndim) {
    bopt = BinaryOpType::VectorScalar;
    dim = d;
    // Case 3: Lx1 and FxM where L and F are broadcastable and M is row
    // contiguous
  } else if (int d = std::max(a_s_dim, b_rc_dim); d < ndim) {
    bopt = BinaryOpType::ScalarVector;
    dim = d;
  }

  // Can be sure dim > 0 since otherwise we would have used one of the fully
  // contiguous methods above. Except for the case that the flags do not
  // correspond to the underlying contiguity.
  if (dim == 0 || strides[dim - 1] < 16) {
    bopt = BinaryOpType::General;
    dim = ndim;
  }

  switch (bopt) {
    case BinaryOpType::VectorVector:
      binary_op_dispatch_dims<T, U, true>(
          a,
          b,
          out,
          VectorVector{op},
          dim,
          new_shape,
          a_strides,
          b_strides,
          strides);
      break;
    case BinaryOpType::VectorScalar:
      binary_op_dispatch_dims<T, U, true>(
          a,
          b,
          out,
          VectorScalar{op},
          dim,
          new_shape,
          a_strides,
          b_strides,
          strides);
      break;
    case BinaryOpType::ScalarVector:
      binary_op_dispatch_dims<T, U, true>(
          a,
          b,
          out,
          ScalarVector{op},
          dim,
          new_shape,
          a_strides,
          b_strides,
          strides);
      break;
    default:
      binary_op_dispatch_dims<T, U, false>(
          a, b, out, op, dim, new_shape, a_strides, b_strides, strides);
      break;
  }
}

template <typename T, typename Op>
void binary_op(const array& a, const array& b, array& out, Op op) {
  binary_op<T, T>(a, b, out, op);
}

template <typename Op>
void binary(const array& a, const array& b, array& out, Op op) {
  switch (out.dtype()) {
    case bool_:
      binary_op<bool>(a, b, out, op);
      break;
    case uint8:
      binary_op<uint8_t>(a, b, out, op);
      break;
    case uint16:
      binary_op<uint16_t>(a, b, out, op);
      break;
    case uint32:
      binary_op<uint32_t>(a, b, out, op);
      break;
    case uint64:
      binary_op<uint64_t>(a, b, out, op);
      break;
    case int8:
      binary_op<int8_t>(a, b, out, op);
      break;
    case int16:
      binary_op<int16_t>(a, b, out, op);
      break;
    case int32:
      binary_op<int32_t>(a, b, out, op);
      break;
    case int64:
      binary_op<int64_t>(a, b, out, op);
      break;
    case float16:
      binary_op<float16_t>(a, b, out, op);
      break;
    case float32:
      binary_op<float>(a, b, out, op);
      break;
    case float64:
      binary_op<double>(a, b, out, op);
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, out, op);
      break;
    case complex64:
      binary_op<complex64_t>(a, b, out, op);
      break;
  }
}

} // namespace mlx::core
