// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename Op>
void binary_op_dims1(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  for (size_t i = 0; i < out_a.size(); ++i) {
    auto dst = op(a_ptr[a_idx], b_ptr[b_idx]);
    dst_a[i] = dst.first;
    dst_b[i] = dst.second;
    a_idx += a.strides()[0];
    b_idx += b.strides()[0];
  }
}

template <typename T, typename U, typename Op>
void binary_op_dims1(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    int stride) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; i++) {
    op(a_ptr + a_idx, b_ptr + b_idx, dst_a, dst_b, stride);
    a_idx += a.strides()[0];
    b_idx += b.strides()[0];
    dst_a += stride;
    dst_b += stride;
  }
}

template <typename T, typename U, typename Op>
void binary_op_dims2(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      auto dst = op(a_ptr[a_idx], b_ptr[b_idx]);
      dst_a[out_idx] = dst.first;
      dst_b[out_idx++] = dst.second;
      a_idx += a.strides()[1];
      b_idx += b.strides()[1];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
  }
}

template <typename T, typename U, typename Op>
void binary_op_dims2(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    int stride) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      op(a_ptr + a_idx, b_ptr + b_idx, dst_a, dst_b, stride);
      a_idx += a.strides()[1];
      b_idx += b.strides()[1];
      dst_a += stride;
      dst_b += stride;
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
  }
}

template <typename T, typename U, typename Op>
void binary_op_dims3(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      for (size_t k = 0; k < a.shape()[2]; ++k) {
        auto dst = op(a_ptr[a_idx], b_ptr[b_idx]);
        dst_a[out_idx] = dst.first;
        dst_b[out_idx++] = dst.second;
        a_idx += a.strides()[2];
        b_idx += b.strides()[2];
      }
      a_idx += a.strides()[1] - a.strides()[2] * a.shape()[2];
      b_idx += b.strides()[1] - b.strides()[2] * b.shape()[2];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
  }
}

template <typename T, typename U, typename Op>
void binary_op_dims4(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      for (size_t k = 0; k < a.shape()[2]; ++k) {
        for (size_t ii = 0; ii < a.shape()[3]; ++ii) {
          auto dst = op(a_ptr[a_idx], b_ptr[b_idx]);
          dst_a[out_idx] = dst.first;
          dst_b[out_idx++] = dst.second;
          a_idx += a.strides()[3];
          b_idx += b.strides()[3];
        }
        a_idx += a.strides()[2] - a.strides()[3] * a.shape()[3];
        b_idx += b.strides()[2] - b.strides()[3] * b.shape()[3];
      }
      a_idx += a.strides()[1] - a.strides()[2] * a.shape()[2];
      b_idx += b.strides()[1] - b.strides()[2] * b.shape()[2];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
  }
}

template <typename T, typename U, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  switch (out_a.ndim()) {
    case 1:
      binary_op_dims1<T, U, Op>(a, b, out_a, out_b, op);
      return;
    case 2:
      binary_op_dims2<T, U, Op>(a, b, out_a, out_b, op);
      return;
    case 3:
      binary_op_dims3<T, U, Op>(a, b, out_a, out_b, op);
      return;
    case 4:
      binary_op_dims4<T, U, Op>(a, b, out_a, out_b, op);
      return;
  }

  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  for (size_t i = 0; i < out_a.size(); i++) {
    int a_idx = elem_to_loc(i, a.shape(), a.strides());
    int b_idx = elem_to_loc(i, b.shape(), b.strides());
    std::tie(dst_a[i], dst_b[i]) = op(a_ptr[a_idx], b_ptr[b_idx]);
  }
}

template <typename T, typename U, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    int dim,
    int stride) {
  // Number of dimensions to loop over for vectorized ops
  switch (dim) {
    case 1:
      binary_op_dims1<T, U, Op>(a, b, out_a, out_b, op, stride);
      return;
    case 2:
      binary_op_dims2<T, U, Op>(a, b, out_a, out_b, op, stride);
      return;
  }

  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  for (size_t i = 0; i < out_a.size(); i += stride) {
    int a_idx = elem_to_loc(i, a.shape(), a.strides());
    int b_idx = elem_to_loc(i, b.shape(), b.strides());
    op(a_ptr + a_idx, b_ptr + b_idx, dst_a, dst_b, stride);
    dst_a += stride;
    dst_b += stride;
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename OpSV,
    typename OpVS,
    typename OpVV>
void binary_op(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    OpSV opsv,
    OpVS opvs,
    OpVV opvv) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_a, bopt);
  set_binary_op_output_data(a, b, out_b, bopt);

  // The full computation is scalar scalar so call the base op once
  if (bopt == BinaryOpType::ScalarScalar) {
    std::tie(*(out_a.data<U>()), *(out_b.data<U>())) =
        op(*a.data<T>(), *b.data<T>());
    return;
  }

  // The full computation is scalar vector so delegate to the op
  if (bopt == BinaryOpType::ScalarVector) {
    opsv(
        a.data<T>(),
        b.data<T>(),
        out_a.data<U>(),
        out_b.data<U>(),
        b.data_size());
    return;
  }

  // The full computation is vector scalar so delegate to the op
  if (bopt == BinaryOpType::VectorScalar) {
    opvs(
        a.data<T>(),
        b.data<T>(),
        out_a.data<U>(),
        out_b.data<U>(),
        a.data_size());
    return;
  }

  // The full computation is vector vector so delegate to the op
  if (bopt == BinaryOpType::VectorVector) {
    opvv(
        a.data<T>(),
        b.data<T>(),
        out_a.data<U>(),
        out_b.data<U>(),
        out_a.size());
    return;
  }

  // General computation so let's try to optimize

  // Get the left-most dim such that the array is row contiguous after
  auto& strides = out_a.strides();
  auto leftmost_rc_dim = [&strides](const array& arr) {
    int d = arr.ndim() - 1;
    for (; d >= 0 && arr.strides()[d] == strides[d]; d--) {
    }
    return d + 1;
  };
  auto a_rc_dim = leftmost_rc_dim(a);
  auto b_rc_dim = leftmost_rc_dim(b);

  // Get the left-most dim such that the array is a broadcasted "scalar" after
  auto leftmost_s_dim = [](const array& arr) {
    int d = arr.ndim() - 1;
    for (; d >= 0 && arr.strides()[d] == 0; d--) {
    }
    return d + 1;
  };
  auto a_s_dim = leftmost_s_dim(a);
  auto b_s_dim = leftmost_s_dim(b);

  auto ndim = out_a.ndim();

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
  size_t stride;
  if (dim == 0 || strides[dim - 1] < 16) {
    stride = 1;
    bopt = BinaryOpType::General;
    dim = ndim;
  } else {
    stride = strides[dim - 1];
  }

  switch (bopt) {
    case BinaryOpType::VectorVector:
      binary_op_dispatch_dims<T, U>(a, b, out_a, out_b, opvv, dim, stride);
      break;
    case BinaryOpType::VectorScalar:
      binary_op_dispatch_dims<T, U>(a, b, out_a, out_b, opvs, dim, stride);
      break;
    case BinaryOpType::ScalarVector:
      binary_op_dispatch_dims<T, U>(a, b, out_a, out_b, opsv, dim, stride);
      break;
    default:
      binary_op_dispatch_dims<T, U>(a, b, out_a, out_b, op);
      break;
  }
}

template <typename T, typename Op, typename OpSV, typename OpVS, typename OpVV>
void binary_op(
    const array& a,
    const array& b,
    std::vector<array>& outputs,
    Op op,
    OpSV opsv,
    OpVS opvs,
    OpVV opvv) {
  // TODO: The following mess of constexpr evaluations can probably be achieved
  //       with template specializations and overloading. Would it be simpler?

  if (std::is_same<decltype(opsv), UseDefaultBinaryOp>::value) {
    if (std::is_same<decltype(opvs), UseDefaultBinaryOp>::value) {
      if (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
        // All ops are UseDefaultBinaryOp (why oh why would someone call that?)
        binary_op<T, T>(
            a,
            b,
            outputs[0],
            outputs[1],
            op,
            DefaultScalarVector<T, T, Op>(op),
            DefaultVectorScalar<T, T, Op>(op),
            DefaultVectorVector<T, T, Op>(op));
      } else {
        // opsv and opvs were UseDefaultBinaryOp
        binary_op<T, T>(
            a,
            b,
            outputs[0],
            outputs[1],
            op,
            DefaultScalarVector<T, T, Op>(op),
            DefaultVectorScalar<T, T, Op>(op),
            opvv);
      }
    } else if (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
      // opsv and opvv were UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          outputs[0],
          outputs[1],
          op,
          DefaultScalarVector<T, T, Op>(op),
          opvs,
          DefaultVectorVector<T, T, Op>(op));
    } else {
      // opsv was UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          outputs[0],
          outputs[1],
          op,
          DefaultScalarVector<T, T, Op>(op),
          opvs,
          opvv);
    }
  } else if (std::is_same<decltype(opvs), UseDefaultBinaryOp>::value) {
    if (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
      // opvs and opvv were UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          outputs[0],
          outputs[1],
          op,
          opsv,
          DefaultVectorScalar<T, T, Op>(op),
          DefaultVectorVector<T, T, Op>(op));
    } else {
      // opvs was UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          outputs[0],
          outputs[1],
          op,
          opsv,
          DefaultVectorScalar<T, T, Op>(op),
          opvv);
    }
  } else if (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
    // opvv was UseDefaultBinaryOp
    binary_op<T, T>(
        a,
        b,
        outputs[0],
        outputs[1],
        op,
        opsv,
        opvs,
        DefaultVectorVector<T, T, Op>(op));
  } else {
    // All ops provided
    binary_op<T, T>(a, b, outputs[0], outputs[1], op, opsv, opvs, opvv);
  }
}

template <typename T, typename Op>
void binary_op(
    const array& a,
    const array& b,
    std::vector<array>& outputs,
    Op op) {
  DefaultScalarVector<T, T, Op> opsv(op);
  DefaultVectorScalar<T, T, Op> opvs(op);
  DefaultVectorVector<T, T, Op> opvv(op);
  binary_op<T, T>(a, b, outputs[0], outputs[1], op, opsv, opvs, opvv);
}

template <typename... Ops>
void binary(
    const array& a,
    const array& b,
    std::vector<array>& outputs,
    Ops... ops) {
  switch (outputs[0].dtype()) {
    case bool_:
      binary_op<bool>(a, b, outputs, ops...);
      break;
    case uint8:
      binary_op<uint8_t>(a, b, outputs, ops...);
      break;
    case uint16:
      binary_op<uint16_t>(a, b, outputs, ops...);
      break;
    case uint32:
      binary_op<uint32_t>(a, b, outputs, ops...);
      break;
    case uint64:
      binary_op<uint64_t>(a, b, outputs, ops...);
      break;
    case int8:
      binary_op<int8_t>(a, b, outputs, ops...);
      break;
    case int16:
      binary_op<int16_t>(a, b, outputs, ops...);
      break;
    case int32:
      binary_op<int32_t>(a, b, outputs, ops...);
      break;
    case int64:
      binary_op<int64_t>(a, b, outputs, ops...);
      break;
    case float16:
      binary_op<float16_t>(a, b, outputs, ops...);
      break;
    case float32:
      binary_op<float>(a, b, outputs, ops...);
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, outputs, ops...);
      break;
    case complex64:
      binary_op<complex64_t>(a, b, outputs, ops...);
      break;
  }
}

} // namespace

} // namespace mlx::core
