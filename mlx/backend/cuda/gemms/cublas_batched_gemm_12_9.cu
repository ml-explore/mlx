// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

__global__ void set_mm_device_pointers(
    int8_t** pointers,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    int item_size,
    const __grid_constant__ Shape batch_shape,
    const __grid_constant__ Strides a_batch_strides,
    const __grid_constant__ Strides b_batch_strides,
    int64_t batch_stride,
    int batch_ndim,
    int batch_count) {
  auto index = cg::this_grid().thread_rank();
  if (index >= batch_count) {
    return;
  }
  auto [a_offset, b_offset] = elem_to_loc(
      index,
      batch_shape.data(),
      a_batch_strides.data(),
      b_batch_strides.data(),
      batch_ndim);
  pointers[index] = a_start + item_size * a_offset;
  pointers[index + batch_count] = b_start + item_size * b_offset;
  pointers[index + 2 * batch_count] =
      out_start + item_size * index * batch_stride;
}

__global__ void set_addmm_device_pointers(
    int8_t** pointers,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* c_start,
    int8_t* out_start,
    int item_size,
    const __grid_constant__ Shape batch_shape,
    const __grid_constant__ Strides a_batch_strides,
    const __grid_constant__ Strides b_batch_strides,
    const __grid_constant__ Strides c_batch_strides,
    int64_t batch_stride,
    int batch_ndim,
    int batch_count) {
  auto index = cg::this_grid().thread_rank();
  if (index >= batch_count) {
    return;
  }
  auto [a_offset, b_offset, c_offset] = elem_to_loc(
      index,
      batch_shape.data(),
      a_batch_strides.data(),
      b_batch_strides.data(),
      c_batch_strides.data(),
      batch_ndim);
  pointers[index] = a_start + item_size * a_offset;
  pointers[index + batch_count] = b_start + item_size * b_offset;
  pointers[index + 2 * batch_count] = c_start + item_size * c_offset;
  pointers[index + 3 * batch_count] =
      out_start + item_size * index * batch_stride;
}

void set_pointer_mode(cublasLtMatrixLayout_t desc, int batch_count) {
  auto batch_mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY;
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
      desc,
      CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
      &batch_mode,
      sizeof(batch_mode)));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
      desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(int32_t)));
}

void Matmul::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides) {
  auto batch_count = out.size() / (M_ * N_);
  set_pointer_mode(a_desc_, batch_count);
  set_pointer_mode(b_desc_, batch_count);
  set_pointer_mode(out_desc_, batch_count);

  // Launch kernel to set device offsets
  auto pointers = array(
      allocator::malloc(batch_count * sizeof(uint64_t) * 3),
      {static_cast<int>(batch_count * 3)},
      uint64);

  encoder.add_temporary(pointers);
  int block_size = 512;
  encoder.set_output_array(pointers);

  encoder.add_kernel_node(
      cu::set_mm_device_pointers,
      cuda::ceil_div(pointers.size(), block_size),
      block_size,
      0,
      pointers.data<int8_t*>(),
      a.data<int8_t>(),
      b.data<int8_t>(),
      out.data<int8_t>(),
      static_cast<int>(out.dtype().size()),
      const_param(batch_shape),
      const_param(a_batch_strides),
      const_param(b_batch_strides),
      static_cast<int64_t>(M_) * N_,
      static_cast<int>(batch_shape.size()),
      batch_count);

  // Run matmul
  encoder.set_input_array(pointers);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  auto a_pointers = pointers.data<int8_t*>();
  auto b_pointers = a_pointers + batch_count;
  auto out_pointers = b_pointers + batch_count;
  run_impl(
      encoder,
      reinterpret_cast<void*>(out_pointers),
      reinterpret_cast<void*>(a_pointers),
      reinterpret_cast<void*>(b_pointers),
      nullptr);
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
  auto batch_count = out.size() / (M_ * N_);
  set_pointer_mode(a_desc_, batch_count);
  set_pointer_mode(b_desc_, batch_count);
  set_pointer_mode(c_desc_, batch_count);
  set_pointer_mode(out_desc_, batch_count);

  // Launch kernel to set device offsets
  auto pointers = array(
      allocator::malloc(batch_count * sizeof(uint64_t) * 4),
      {static_cast<int>(batch_count * 4)},
      uint64);

  encoder.add_temporary(pointers);
  int block_size = 512;
  encoder.set_output_array(pointers);
  encoder.add_kernel_node(
      cu::set_addmm_device_pointers,
      cuda::ceil_div(pointers.size(), block_size),
      block_size,
      0,
      pointers.data<int8_t*>(),
      a.data<int8_t>(),
      b.data<int8_t>(),
      c.data<int8_t>(),
      out.data<int8_t>(),
      static_cast<int>(out.dtype().size()),
      const_param(batch_shape),
      const_param(a_batch_strides),
      const_param(b_batch_strides),
      const_param(c_batch_strides),
      static_cast<int64_t>(M_) * N_,
      static_cast<int>(batch_shape.size()),
      batch_count);

  // Run matmul
  encoder.set_input_array(pointers);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  auto a_pointers = pointers.data<int8_t*>();
  auto b_pointers = a_pointers + batch_count;
  auto c_pointers = b_pointers + batch_count;
  auto out_pointers = c_pointers + batch_count;
  run_impl(
      encoder,
      reinterpret_cast<void*>(out_pointers),
      reinterpret_cast<void*>(a_pointers),
      reinterpret_cast<void*>(b_pointers),
      reinterpret_cast<void*>(c_pointers),
      alpha,
      beta);
}

} // namespace mlx::core::cu
