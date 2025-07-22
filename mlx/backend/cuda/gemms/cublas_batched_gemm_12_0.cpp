// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"

namespace mlx::core::cu {

void Matmul::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  auto nbatch = out.size() / (M_ * N_ * batch_shape.back());
  ContiguousIterator a_it(batch_shape, a_batch_strides, batch_shape.size() - 1);
  ContiguousIterator b_it(batch_shape, b_batch_strides, batch_shape.size() - 1);
  auto concurrent = encoder.concurrent_context();
  for (size_t i = 0; i < nbatch; ++i) {
    run_impl(
        encoder,
        out.data<int8_t>() + out.itemsize() * i * batch_shape.back() * M_ * N_,
        a.data<int8_t>() + a.itemsize() * a_it.loc,
        b.data<int8_t>() + b.itemsize() * b_it.loc,
        nullptr);
    a_it.step();
    b_it.step();
  }
}

void Matmul::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& c,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides,
    const mlx::core::Strides& c_batch_strides,
    float alpha,
    float beta) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  auto nbatch = out.size() / (M_ * N_ * batch_shape.back());
  ContiguousIterator a_it(batch_shape, a_batch_strides, batch_shape.size() - 1);
  ContiguousIterator b_it(batch_shape, b_batch_strides, batch_shape.size() - 1);
  ContiguousIterator c_it(batch_shape, c_batch_strides, batch_shape.size() - 1);
  auto concurrent = encoder.concurrent_context();
  for (size_t i = 0; i < nbatch; ++i) {
    run_impl(
        encoder,
        out.data<int8_t>() + out.itemsize() * i * batch_shape.back() * M_ * N_,
        a.data<int8_t>() + a.itemsize() * a_it.loc,
        b.data<int8_t>() + b.itemsize() * b_it.loc,
        c.data<int8_t>() + c.itemsize() * c_it.loc,
        alpha,
        beta);
    a_it.step();
    b_it.step();
    c_it.step();
  }
}

} // namespace mlx::core::cu
