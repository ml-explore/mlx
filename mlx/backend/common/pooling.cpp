#include "mlx/primitives.h"

namespace mlx::core {
namespace {

template <typename T>
void avg_pool_1d(
    const T* data_p,
    T* out_p,
    const std::vector<int>& in_shape,
    const std::vector<size_t>& in_strides,
    const std::vector<int>& out_shape,
    const std::vector<size_t>& out_strides,
    int kernel_size,
    int stride,
    int padding) {
  auto out_b = out_shape.at(0);
  auto out_h = out_shape.at(1);
  auto out_c = out_shape.at(2);

  auto in_h = in_shape.at(1);
  auto out_stride_b = out_strides.at(0);
  auto out_stride_h = out_strides.at(1);
  auto out_stride_c = out_strides.at(2);

  auto in_stride_b = in_strides.at(0);
  auto in_stride_h = in_strides.at(1);
  auto in_stride_c = in_strides.at(2);

  T avg = T(1) / kernel_size;
  T last_avg = avg;
  bool last_col = false;
  int start = (out_h - 1) * stride - padding;
  int end = std::min(start + kernel_size, in_h);
  int loop_h = out_h;

  // Might be unnecessary, but instead of dividing per k, we can multiply which
  // is a faster op. So before the loop, figure out the averages needed and
  // whether or not the last col will be smaller. This only becomes an issue on
  // the last col as it may be a smaller window in which case the avg is not
  // 1/ks. So only iterate until the second to last element and then handle the
  // last element separately if needed.
  if (end + padding < start + kernel_size) {
    last_avg = T(1) / (padding + (end - start));
    last_col = true;
    loop_h = out_h - 1;
  }

  T val = T(0);
  for (int b = 0; b < out_b; b++) {
    for (int c = 0; c < out_c; c++) {
      for (int h = 0; h < loop_h; h++) {
        start = h * stride - padding;
        end = std::min(start + kernel_size, in_h);
        start = std::max(start, 0);
        val = 0;
        for (int i = start; i < end; i++) {
          val += data_p[b * in_stride_b + i * in_stride_h + c * in_stride_c];
        }
        val *= avg;
        out_p[b * out_stride_b + h * out_stride_h + c * out_stride_c] = val;
      }
      if (last_col) {
        start = (out_h - 1) * stride - padding;
        end = std::min(start + kernel_size, in_h);
        start = std::max(start, 0);
        val = 0;
        for (int i = start; i < end; i++) {
          val += data_p[b * in_stride_b + i * in_stride_h + c * in_stride_c];
        }
        val *= last_avg;
        out_p
            [b * out_stride_b + (out_h - 1) * out_stride_h + c * out_stride_c] =
                val;
      }
    }
  }
}

template <typename T>
void max_pool_1d(
    const T* data_p,
    T* out_p,
    const std::vector<int>& in_shape,
    const std::vector<size_t>& in_strides,
    const std::vector<int>& out_shape,
    const std::vector<size_t>& out_strides,
    int kernel_size,
    int stride,
    int padding) {
  auto out_b = out_shape.at(0);
  auto out_h = out_shape.at(1);
  auto out_c = out_shape.at(2);

  auto in_h = in_shape.at(1);

  auto out_stride_b = out_strides.at(0);
  auto out_stride_h = out_strides.at(1);
  auto out_stride_c = out_strides.at(2);

  auto in_stride_b = in_strides.at(0);
  auto in_stride_h = in_strides.at(1);
  auto in_stride_c = in_strides.at(2);

  for (int b = 0; b < out_b; b++) {
    for (int c = 0; c < out_c; c++) {
      for (int h = 0; h < out_h; h++) {
        int start = h * stride - padding;
        int end = std::min(start + kernel_size, in_h);
        start = std::max(start, 0);
        T val = 0;
        val = -std::numeric_limits<T>::infinity();
        for (int i = start; i < end; i++) {
          val = std::max(
              val, data_p[b * in_stride_b + i * in_stride_h + c * in_stride_c]);
        }
        out_p[b * out_stride_b + h * out_stride_h + c * out_stride_c] = val;
      }
    }
  }
}

void wrap_pool_1d(
    const array& in,
    array& out,
    int kernel_size,
    int stride,
    int padding,
    Pooling::PoolType type) {
  // TODO: extract this out into a template function based on dtype, for now its
  // float32 only
  switch (in.dtype()) {
    case float32: {
      if (type == Pooling::PoolType::Max) {
        max_pool_1d<float>(
            in.data<float>(),
            out.data<float>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      } else {
        avg_pool_1d<float>(
            in.data<float>(),
            out.data<float>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      }
      return;
    }
    case bfloat16: {
      if (type == Pooling::PoolType::Max) {
        max_pool_1d<bfloat16_t>(
            in.data<bfloat16_t>(),
            out.data<bfloat16_t>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      } else {
        avg_pool_1d<bfloat16_t>(
            in.data<bfloat16_t>(),
            out.data<bfloat16_t>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      }
      return;
    }
    case float16: {
      if (type == Pooling::PoolType::Max) {
        max_pool_1d<float16_t>(
            in.data<float16_t>(),
            out.data<float16_t>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      } else {
        avg_pool_1d<float16_t>(
            in.data<float16_t>(),
            out.data<float16_t>(),
            in.shape(),
            in.strides(),
            out.shape(),
            out.strides(),
            kernel_size,
            stride,
            padding);
      }
      return;
    }
    default: {
      throw std::runtime_error("[Pooling] Unsupported data type for pooling");
    }
  }
}
} // namespace

void Pooling::eval(const std::vector<array>& inputs, array& output) {
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  if (inputs[0].ndim() == 3) {
    wrap_pool_1d(
        inputs[0], output, kernel_size_[0], strides_[0], padding_[0], type_);
    return;
  }

  throw std::runtime_error("[Pooling] only 1d ops supported for now.");
}

} // namespace mlx::core