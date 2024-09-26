// Copyright © 2023 Apple Inc.

#pragma once
#include <cassert>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

enum class BinaryOpType {
  ScalarScalar,
  ScalarVector,
  VectorScalar,
  VectorVector,
  General,
};

BinaryOpType get_binary_op_type(const array& a, const array& b) {
  BinaryOpType bopt;
  if (a.data_size() == 1 && b.data_size() == 1) {
    bopt = BinaryOpType::ScalarScalar;
  } else if (a.data_size() == 1 && b.flags().contiguous) {
    bopt = BinaryOpType::ScalarVector;
  } else if (b.data_size() == 1 && a.flags().contiguous) {
    bopt = BinaryOpType::VectorScalar;
  } else if (
      a.flags().row_contiguous && b.flags().row_contiguous ||
      a.flags().col_contiguous && b.flags().col_contiguous) {
    bopt = BinaryOpType::VectorVector;
  } else {
    bopt = BinaryOpType::General;
  }
  return bopt;
}

void set_binary_op_output_data(
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    bool donate_with_move = false) {
  bool b_donatable = is_donatable(b, out);
  bool a_donatable = is_donatable(a, out);
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      out.set_data(
          allocator::malloc_or_wait(out.itemsize()), 1, a.strides(), a.flags());
      break;
    case BinaryOpType::ScalarVector:
      if (b_donatable) {
        if (donate_with_move) {
          out.move_shared_buffer(b);
        } else {
          out.copy_shared_buffer(b);
        }
      } else {
        out.set_data(
            allocator::malloc_or_wait(b.data_size() * out.itemsize()),
            b.data_size(),
            b.strides(),
            b.flags());
      }
      break;
    case BinaryOpType::VectorScalar:
      if (a_donatable) {
        if (donate_with_move) {
          out.move_shared_buffer(a);
        } else {
          out.copy_shared_buffer(a);
        }
      } else {
        out.set_data(
            allocator::malloc_or_wait(a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::VectorVector:
      if (a_donatable) {
        if (donate_with_move) {
          out.move_shared_buffer(a);
        } else {
          out.copy_shared_buffer(a);
        }
      } else if (b_donatable) {
        if (donate_with_move) {
          out.move_shared_buffer(b);
        } else {
          out.copy_shared_buffer(b);
        }
      } else {
        out.set_data(
            allocator::malloc_or_wait(a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::General:
      if (a_donatable && a.flags().row_contiguous && a.size() == out.size()) {
        if (donate_with_move) {
          out.move_shared_buffer(a);
        } else {
          out.copy_shared_buffer(a);
        }
      } else if (
          b_donatable && b.flags().row_contiguous && b.size() == out.size()) {
        if (donate_with_move) {
          out.move_shared_buffer(b);
        } else {
          out.copy_shared_buffer(b);
        }
      } else {
        out.set_data(allocator::malloc_or_wait(out.nbytes()));
      }
      break;
  }
}

struct UseDefaultBinaryOp {};

template <typename T, typename U, typename Op>
struct DefaultVectorScalar {
  Op op;

  DefaultVectorScalar(Op op_) : op(op_) {}

  void operator()(const T* a, const T* b, U* dst, int size) {
    T scalar = *b;
    while (size-- > 0) {
      *dst = op(*a, scalar);
      dst++;
      a++;
    }
  }
};

template <typename T, typename U, typename Op>
struct DefaultScalarVector {
  Op op;

  DefaultScalarVector(Op op_) : op(op_) {}

  void operator()(const T* a, const T* b, U* dst, int size) {
    T scalar = *a;
    while (size-- > 0) {
      *dst = op(scalar, *b);
      dst++;
      b++;
    }
  }
};

template <typename T, typename U, typename Op>
struct DefaultVectorVector {
  Op op;

  DefaultVectorVector(Op op_) : op(op_) {}

  void operator()(const T* a, const T* b, U* dst, int size) {
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
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& out_strides,
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
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& out_strides) {
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

  ContiguousIterator<size_t> a_it(shape, a_strides, dim - 3);
  ContiguousIterator<size_t> b_it(shape, b_strides, dim - 3);
  size_t stride = out_strides[dim - 4];
  for (size_t elem = 0; elem < a.size(); elem += stride) {
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
    array& out,
    Op op,
    OpSV opsv,
    OpVS opvs,
    OpVV opvv) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  // The full computation is scalar scalar so call the base op once
  if (bopt == BinaryOpType::ScalarScalar) {
    *(out.data<U>()) = op(*a.data<T>(), *b.data<T>());
    return;
  }

  // The full computation is scalar vector so delegate to the op
  if (bopt == BinaryOpType::ScalarVector) {
    opsv(a.data<T>(), b.data<T>(), out.data<U>(), b.data_size());
    return;
  }

  // The full computation is vector scalar so delegate to the op
  if (bopt == BinaryOpType::VectorScalar) {
    opvs(a.data<T>(), b.data<T>(), out.data<U>(), a.data_size());
    return;
  }

  // The full computation is vector vector so delegate to the op
  if (bopt == BinaryOpType::VectorVector) {
    opvv(a.data<T>(), b.data<T>(), out.data<U>(), out.size());
    return;
  }

  // General computation so let's try to optimize
  auto [new_shape, new_strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), out.strides()});
  const auto& a_strides = new_strides[0];
  const auto& b_strides = new_strides[1];
  const auto& strides = new_strides[2];

  // Get the left-most dim such that the array is row contiguous after
  auto leftmost_rc_dim = [&strides](const std::vector<size_t>& arr_strides) {
    int d = arr_strides.size() - 1;
    for (; d >= 0 && arr_strides[d] == strides[d]; d--) {
    }
    return d + 1;
  };
  auto a_rc_dim = leftmost_rc_dim(a_strides);
  auto b_rc_dim = leftmost_rc_dim(b_strides);

  // Get the left-most dim such that the array is a broadcasted "scalar" after
  auto leftmost_s_dim = [](const std::vector<size_t>& arr_strides) {
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
          a, b, out, opvv, dim, new_shape, a_strides, b_strides, strides);
      break;
    case BinaryOpType::VectorScalar:
      binary_op_dispatch_dims<T, U, true>(
          a, b, out, opvs, dim, new_shape, a_strides, b_strides, strides);
      break;
    case BinaryOpType::ScalarVector:
      binary_op_dispatch_dims<T, U, true>(
          a, b, out, opsv, dim, new_shape, a_strides, b_strides, strides);
      break;
    default:
      binary_op_dispatch_dims<T, U, false>(
          a, b, out, op, dim, new_shape, a_strides, b_strides, strides);
      break;
  }
}

template <typename T, typename Op, typename OpSV, typename OpVS, typename OpVV>
void binary_op(
    const array& a,
    const array& b,
    array& out,
    Op op,
    OpSV opsv,
    OpVS opvs,
    OpVV opvv) {
  // TODO: The following mess of constexpr evaluations can probably be achieved
  //       with template specializations and overloading. Would it be simpler?

  if constexpr (std::is_same<decltype(opsv), UseDefaultBinaryOp>::value) {
    if constexpr (std::is_same<decltype(opvs), UseDefaultBinaryOp>::value) {
      if constexpr (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
        // All ops are UseDefaultBinaryOp (why oh why would someone call that?)
        binary_op<T, T>(
            a,
            b,
            out,
            op,
            DefaultScalarVector<T, T, Op>(op),
            DefaultVectorScalar<T, T, Op>(op),
            DefaultVectorVector<T, T, Op>(op));
      } else {
        // opsv and opvs were UseDefaultBinaryOp
        binary_op<T, T>(
            a,
            b,
            out,
            op,
            DefaultScalarVector<T, T, Op>(op),
            DefaultVectorScalar<T, T, Op>(op),
            opvv);
      }
    } else if constexpr (std::is_same<decltype(opvv), UseDefaultBinaryOp>::
                             value) {
      // opsv and opvv were UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          out,
          op,
          DefaultScalarVector<T, T, Op>(op),
          opvs,
          DefaultVectorVector<T, T, Op>(op));
    } else {
      // opsv was UseDefaultBinaryOp
      binary_op<T, T>(
          a, b, out, op, DefaultScalarVector<T, T, Op>(op), opvs, opvv);
    }
  } else if constexpr (std::is_same<decltype(opvs), UseDefaultBinaryOp>::
                           value) {
    if (std::is_same<decltype(opvv), UseDefaultBinaryOp>::value) {
      // opvs and opvv were UseDefaultBinaryOp
      binary_op<T, T>(
          a,
          b,
          out,
          op,
          opsv,
          DefaultVectorScalar<T, T, Op>(op),
          DefaultVectorVector<T, T, Op>(op));
    } else {
      // opvs was UseDefaultBinaryOp
      binary_op<T, T>(
          a, b, out, op, opsv, DefaultVectorScalar<T, T, Op>(op), opvv);
    }
  } else if constexpr (std::is_same<decltype(opvv), UseDefaultBinaryOp>::
                           value) {
    // opvv was UseDefaultBinaryOp
    binary_op<T, T>(
        a, b, out, op, opsv, opvs, DefaultVectorVector<T, T, Op>(op));
  } else {
    // All ops provided
    binary_op<T, T>(a, b, out, op, opsv, opvs, opvv);
  }
}

template <typename T, typename Op>
void binary_op(const array& a, const array& b, array& out, Op op) {
  DefaultScalarVector<T, T, Op> opsv(op);
  DefaultVectorScalar<T, T, Op> opvs(op);
  DefaultVectorVector<T, T, Op> opvv(op);
  binary_op<T, T>(a, b, out, op, opsv, opvs, opvv);
}

template <typename... Ops>
void binary(const array& a, const array& b, array& out, Ops... ops) {
  switch (out.dtype()) {
    case bool_:
      binary_op<bool>(a, b, out, ops...);
      break;
    case uint8:
      binary_op<uint8_t>(a, b, out, ops...);
      break;
    case uint16:
      binary_op<uint16_t>(a, b, out, ops...);
      break;
    case uint32:
      binary_op<uint32_t>(a, b, out, ops...);
      break;
    case uint64:
      binary_op<uint64_t>(a, b, out, ops...);
      break;
    case int8:
      binary_op<int8_t>(a, b, out, ops...);
      break;
    case int16:
      binary_op<int16_t>(a, b, out, ops...);
      break;
    case int32:
      binary_op<int32_t>(a, b, out, ops...);
      break;
    case int64:
      binary_op<int64_t>(a, b, out, ops...);
      break;
    case float16:
      binary_op<float16_t>(a, b, out, ops...);
      break;
    case float32:
      binary_op<float>(a, b, out, ops...);
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, out, ops...);
      break;
    case complex64:
      binary_op<complex64_t>(a, b, out, ops...);
      break;
  }
}

} // namespace

} // namespace mlx::core
