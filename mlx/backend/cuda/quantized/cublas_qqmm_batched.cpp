// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"

namespace mlx::core {

void CublasQQMM::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    float alpha) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(a_scale);
  encoder.set_input_array(b_scale);
  encoder.set_output_array(out);

  auto nbatch = out.size() / (M_ * N_ * batch_shape.back());

  ContiguousIterator a_it(batch_shape, a_batch_strides, batch_shape.size() - 1);
  ContiguousIterator b_it(batch_shape, b_batch_strides, batch_shape.size() - 1);

  // Scales are contiguous, so their batch stride is just the size of one scale
  // matrix (?)
  size_t a_scale_batch_stride = a_scale.shape(-2) * a_scale.shape(-1);
  size_t b_scale_batch_stride = b_scale.shape(-2) * b_scale.shape(-1);

  auto concurrent = encoder.concurrent_context();
  for (size_t i = 0; i < nbatch; ++i) {
    execute(
        encoder,
        gpu_ptr<uint8_t>(out) +
            out.itemsize() * i * batch_shape.back() * M_ * N_,
        gpu_ptr<uint8_t>(a) + a.itemsize() * a_it.loc,
        gpu_ptr<uint8_t>(b) + b.itemsize() * b_it.loc,
        gpu_ptr<uint8_t>(a_scale) + i * a_scale_batch_stride,
        gpu_ptr<uint8_t>(b_scale) + i * b_scale_batch_stride,
        nullptr,
        alpha);
    a_it.step();
    b_it.step();
  }
}

} // namespace mlx::core
