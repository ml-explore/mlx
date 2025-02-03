// Copyright © 2023 Apple Inc.

#include <cassert>
#include <functional>
#include <limits>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename U>
struct Limits {
  static const U max;
  static const U min;
};

#define instantiate_default_limit(type)                           \
  template <>                                                     \
  struct Limits<type> {                                           \
    static constexpr type max = std::numeric_limits<type>::max(); \
    static constexpr type min = std::numeric_limits<type>::min(); \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type) \
  template <>                         \
  struct Limits<type> {               \
    static const type max;            \
    static const type min;            \
  };

instantiate_float_limit(float16_t);
instantiate_float_limit(bfloat16_t);
instantiate_float_limit(float);
instantiate_float_limit(complex64_t);

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

const float Limits<float>::max = std::numeric_limits<float>::infinity();
const float Limits<float>::min = -std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::max =
    std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::min =
    -std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::max = std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::min =
    -std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::max =
    std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::min =
    -std::numeric_limits<float>::infinity();

template <typename T, typename U, typename Op>
void strided_reduce(
    const T* x,
    U* accumulator,
    int size,
    size_t stride,
    Op op) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  for (int i = 0; i < size; i++) {
    U* moving_accumulator = accumulator;
    auto s = stride;
    while (s >= N) {
      auto acc = simd::load<U, N>(moving_accumulator);
      auto v = simd::Simd<U, N>(simd::load<T, N>(x));
      simd::store<U, N>(moving_accumulator, op(acc, v));
      moving_accumulator += N;
      x += N;
      s -= N;
    }
    while (s-- > 0) {
      *moving_accumulator = op(*moving_accumulator, *x);
      moving_accumulator++;
      x++;
    }
  }
};

template <typename T, typename U, typename Op>
void contiguous_reduce(const T* x, U* accumulator, int size, Op op, U init) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  simd::Simd<U, N> accumulator_v(init);
  while (size >= N) {
    accumulator_v = op(accumulator_v, simd::Simd<U, N>(simd::load<T, N>(x)));
    x += N;
    size -= N;
  }
  *accumulator = op(*accumulator, op(accumulator_v));
  while (size-- > 0) {
    *accumulator = op(*accumulator, *x);
    x++;
  }
}

// Helper for the ndimensional strided loop
void nd_loop(
    std::function<void(int)> callback,
    const Shape& shape,
    const Strides& strides) {
  std::function<void(int, int)> loop_inner;
  loop_inner = [&](int dim, int offset) {
    if (dim < shape.size() - 1) {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        loop_inner(dim + 1, offset + i * stride);
      }
    } else {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        callback(offset + i * stride);
      }
    }
  };
  loop_inner(0, 0);
}

template <typename T, typename U, typename Op>
void reduction_op(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init,
    Op op) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  ReductionPlan plan = get_reduction_plan(x, axes);

  if (plan.type == ContiguousAllReduce) {
    U* out_ptr = out.data<U>();
    *out_ptr = init;
    contiguous_reduce(x.data<T>(), out_ptr, x.size(), op, init);
    return;
  }

  if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape[0];
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    for (int i = 0; i < out.size(); i++, out_ptr++, x_ptr += reduction_size) {
      *out_ptr = init;
      contiguous_reduce(x_ptr, out_ptr, reduction_size, op, init);
    }
    return;
  }

  if (plan.type == GeneralContiguousReduce || plan.type == ContiguousReduce) {
    int reduction_size = plan.shape.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    // Unrolling the following loop (and implementing it in order for
    // ContiguousReduce) should hold extra performance boost.
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        contiguous_reduce(x_ptr + offset, out_ptr, reduction_size, op, init);
      }
    } else {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        nd_loop(
            [&](int extra_offset) {
              contiguous_reduce(
                  x_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  op,
                  init);
            },
            plan.shape,
            plan.strides);
      }
    }
    return;
  }

  if (plan.type == ContiguousStridedReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    for (int i = 0; i < out.size(); i += reduction_stride) {
      std::fill_n(out_ptr, reduction_stride, init);
      strided_reduce(x_ptr, out_ptr, reduction_size, reduction_stride, op);
      x_ptr += reduction_stride * reduction_size;
      out_ptr += reduction_stride;
    }
    return;
  }

  if (plan.type == GeneralStridedReduce ||
      plan.type == ContiguousStridedReduce) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        strided_reduce(
            x_ptr + offset, out_ptr, reduction_size, reduction_stride, op);
        out_ptr += reduction_stride;
      }
    } else {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        nd_loop(
            [&](int extra_offset) {
              strided_reduce(
                  x_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  reduction_stride,
                  op);
            },
            plan.shape,
            plan.strides);
        out_ptr += reduction_stride;
      }
    }
    return;
  }

  if (plan.type == GeneralReduce) {
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);
    for (int i = 0; i < out.size(); i++, out_ptr++) {
      int offset = elem_to_loc(i, shape, strides);
      U val = init;
      nd_loop(
          [&](int extra_offset) {
            val = op(val, *(x_ptr + offset + extra_offset));
          },
          plan.shape,
          plan.strides);
      *out_ptr = val;
    }
  }
}

struct AndReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x & (y != 0);
  }

  bool operator()(bool x, bool y) {
    return x & y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x & (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x & y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::all(x);
  };
};

struct OrReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x | (y != 0);
  }

  bool operator()(bool x, bool y) {
    return x | y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x | (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x | y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::any(x);
  };
};

struct MaxReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::maximum(x, y);
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::max(x);
  };
};

struct MinReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::minimum(x, y);
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::min(x);
  };
};

struct SumReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return x + y;
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return y + x;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::sum(x);
  };
};

struct ProdReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return x * y;
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return x * y;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::prod(x);
  };
};

template <typename InT>
void reduce_dispatch_and_or(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::And) {
    reduction_op<InT, bool>(in, out, axes, true, AndReduce());
  } else {
    reduction_op<InT, bool>(in, out, axes, false, OrReduce());
  }
}

template <typename InT>
void reduce_dispatch_sum_prod(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Sum) {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t>(in, out, axes, 0, SumReduce());
    } else {
      reduction_op<InT, InT>(in, out, axes, 0, SumReduce());
    }
  } else {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t>(in, out, axes, 1, ProdReduce());
    } else {
      reduction_op<InT, InT>(in, out, axes, 1, ProdReduce());
    }
  }
}

template <typename InT>
void reduce_dispatch_min_max(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Max) {
    auto init = Limits<InT>::min;
    reduction_op<InT, InT>(in, out, axes, init, MaxReduce());
  } else {
    auto init = Limits<InT>::max;
    reduction_op<InT, InT>(in, out, axes, init, MinReduce());
  }
}

void Reduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  switch (reduce_type_) {
    case Reduce::And:
    case Reduce::Or: {
      switch (in.dtype()) {
        case bool_:
        case uint8:
        case int8:
          reduce_dispatch_and_or<int8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
        case uint16:
        case float16:
        case bfloat16:
          reduce_dispatch_and_or<int16_t>(in, out, reduce_type_, axes_);
          break;
        case uint32:
        case int32:
        case float32:
          reduce_dispatch_and_or<int32_t>(in, out, reduce_type_, axes_);
          break;
        case uint64:
        case int64:
        case complex64:
          reduce_dispatch_and_or<int64_t>(in, out, reduce_type_, axes_);
          break;
      }
      break;
    }
    case Reduce::Sum:
    case Reduce::Prod: {
      switch (in.dtype()) {
        case bool_:
        case uint8:
        case int8:
          reduce_dispatch_sum_prod<int8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
        case uint16:
          reduce_dispatch_sum_prod<int16_t>(in, out, reduce_type_, axes_);
          break;
        case int32:
        case uint32:
          reduce_dispatch_sum_prod<int32_t>(in, out, reduce_type_, axes_);
          break;
        case int64:
        case uint64:
          reduce_dispatch_sum_prod<int64_t>(in, out, reduce_type_, axes_);
          break;
        case float16:
          reduce_dispatch_sum_prod<float16_t>(in, out, reduce_type_, axes_);
          break;
        case bfloat16:
          reduce_dispatch_sum_prod<bfloat16_t>(in, out, reduce_type_, axes_);
          break;
        case float32:
          reduce_dispatch_sum_prod<float>(in, out, reduce_type_, axes_);
          break;
        case complex64:
          reduce_dispatch_sum_prod<complex64_t>(in, out, reduce_type_, axes_);
          break;
      }
      break;
    }
    case Reduce::Max:
    case Reduce::Min: {
      switch (in.dtype()) {
        case bool_:
          reduce_dispatch_min_max<bool>(in, out, reduce_type_, axes_);
          break;
        case uint8:
          reduce_dispatch_min_max<uint8_t>(in, out, reduce_type_, axes_);
          break;
        case uint16:
          reduce_dispatch_min_max<uint16_t>(in, out, reduce_type_, axes_);
          break;
        case uint32:
          reduce_dispatch_min_max<uint32_t>(in, out, reduce_type_, axes_);
          break;
        case uint64:
          reduce_dispatch_min_max<uint64_t>(in, out, reduce_type_, axes_);
          break;
        case int8:
          reduce_dispatch_min_max<uint8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
          reduce_dispatch_min_max<uint16_t>(in, out, reduce_type_, axes_);
          break;
        case int32:
          reduce_dispatch_min_max<int32_t>(in, out, reduce_type_, axes_);
          break;
        case int64:
          reduce_dispatch_min_max<int64_t>(in, out, reduce_type_, axes_);
          break;
        case float16:
          reduce_dispatch_min_max<float16_t>(in, out, reduce_type_, axes_);
          break;
        case float32:
          reduce_dispatch_min_max<float>(in, out, reduce_type_, axes_);
          break;
        case bfloat16:
          reduce_dispatch_min_max<bfloat16_t>(in, out, reduce_type_, axes_);
          break;
        case complex64:
          reduce_dispatch_min_max<complex64_t>(in, out, reduce_type_, axes_);
          break;
      }
      break;
    }
  }
}

} // namespace mlx::core
