// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <int NDIM>
__global__ void set_mm_device_pointers_nd(
    int8_t** pointers,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    int item_size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> batch_shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_batch_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_batch_strides,
    int64_t batch_stride,
    int batch_count) {
  auto index = cg::this_grid().thread_rank();
  if (index >= batch_count) {
    return;
  }
  auto [a_offset, b_offset] = elem_to_loc_nd<NDIM>(
      index,
      batch_shape.data(),
      a_batch_strides.data(),
      b_batch_strides.data());
  pointers[index] = a_start + item_size * a_offset;
  pointers[index + batch_count] = b_start + item_size * b_offset;
  pointers[index + 2 * batch_count] =
      out_start + item_size * index * batch_stride;
}

__global__ void set_mm_device_pointers_g(
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

template <int NDIM>
__global__ void set_addmm_device_pointers_nd(
    int8_t** pointers,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* c_start,
    int8_t* out_start,
    int item_size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> batch_shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_batch_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_batch_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> c_batch_strides,
    int64_t batch_stride,
    int batch_count) {
  auto index = cg::this_grid().thread_rank();
  if (index >= batch_count) {
    return;
  }
  auto [a_offset, b_offset, c_offset] = elem_to_loc_nd<NDIM>(
      index,
      batch_shape.data(),
      a_batch_strides.data(),
      b_batch_strides.data(),
      c_batch_strides.data());
  pointers[index] = a_start + item_size * a_offset;
  pointers[index + batch_count] = b_start + item_size * b_offset;
  pointers[index + 2 * batch_count] = c_start + item_size * c_offset;
  pointers[index + 3 * batch_count] =
      out_start + item_size * index * batch_stride;
}

__global__ void set_addmm_device_pointers_g(
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

} // namespace cu

namespace {

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

} // namespace

void CublasGemm::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    float alpha) {
  int batch_count = out.size() / (M_ * N_);
  set_pointer_mode(a_desc_, batch_count);
  set_pointer_mode(b_desc_, batch_count);
  set_pointer_mode(out_desc_, batch_count);

  // Launch kernel to set device offsets
  auto pointers = array(
      allocator::malloc(batch_count * sizeof(void*) * 3),
      {batch_count * 3},
      uint64);

  encoder.add_temporary(pointers);
  encoder.set_output_array(pointers);

  int block_dims = std::min(batch_count, 256);
  int num_blocks = cuda::ceil_div(batch_count, block_dims);
  int64_t batch_stride = M_ * N_;
  int item_size = out.itemsize();

  int ndim = batch_shape.size();
  if (ndim <= 3) {
    dispatch_1_2_3(ndim, [&](auto ndim_constant) {
      encoder.add_kernel_node(
          cu::set_mm_device_pointers_nd<ndim_constant()>,
          num_blocks,
          block_dims,
          0,
          pointers.data<int8_t*>(),
          a.data<int8_t>(),
          b.data<int8_t>(),
          out.data<int8_t>(),
          item_size,
          const_param<ndim_constant()>(batch_shape),
          const_param<ndim_constant()>(a_batch_strides),
          const_param<ndim_constant()>(b_batch_strides),
          batch_stride,
          batch_count);
    });
  } else {
    encoder.add_kernel_node(
        cu::set_mm_device_pointers_g,
        num_blocks,
        block_dims,
        0,
        pointers.data<int8_t*>(),
        a.data<int8_t>(),
        b.data<int8_t>(),
        out.data<int8_t>(),
        item_size,
        const_param(batch_shape),
        const_param(a_batch_strides),
        const_param(b_batch_strides),
        batch_stride,
        ndim,
        batch_count);
  }

  // Run matmul
  encoder.set_input_array(pointers);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  auto a_pointers = pointers.data<int8_t*>();
  auto b_pointers = a_pointers + batch_count;
  auto out_pointers = b_pointers + batch_count;
  execute(
      encoder,
      reinterpret_cast<void*>(out_pointers),
      reinterpret_cast<void*>(a_pointers),
      reinterpret_cast<void*>(b_pointers),
      nullptr,
      alpha);
}

void CublasGemm::run_batched(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& c,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    const Strides& c_batch_strides,
    float alpha,
    float beta) {
  int batch_count = out.size() / (M_ * N_);
  set_pointer_mode(a_desc_, batch_count);
  set_pointer_mode(b_desc_, batch_count);
  set_pointer_mode(c_desc_, batch_count);
  set_pointer_mode(out_desc_, batch_count);

  // Launch kernel to set device offsets
  auto pointers = array(
      allocator::malloc(batch_count * sizeof(uint64_t) * 4),
      {batch_count * 4},
      uint64);

  encoder.add_temporary(pointers);
  encoder.set_output_array(pointers);

  int block_dims = std::min(batch_count, 256);
  int num_blocks = cuda::ceil_div(batch_count, block_dims);
  int64_t batch_stride = M_ * N_;
  int item_size = out.itemsize();

  int ndim = batch_shape.size();
  if (ndim <= 3) {
    dispatch_1_2_3(ndim, [&](auto ndim_constant) {
      encoder.add_kernel_node(
          cu::set_addmm_device_pointers_nd<ndim_constant()>,
          num_blocks,
          block_dims,
          0,
          pointers.data<int8_t*>(),
          a.data<int8_t>(),
          b.data<int8_t>(),
          c.data<int8_t>(),
          out.data<int8_t>(),
          item_size,
          const_param<ndim_constant()>(batch_shape),
          const_param<ndim_constant()>(a_batch_strides),
          const_param<ndim_constant()>(b_batch_strides),
          const_param<ndim_constant()>(c_batch_strides),
          batch_stride,
          batch_count);
    });
  } else {
    encoder.add_kernel_node(
        cu::set_addmm_device_pointers_g,
        num_blocks,
        block_dims,
        0,
        pointers.data<int8_t*>(),
        a.data<int8_t>(),
        b.data<int8_t>(),
        c.data<int8_t>(),
        out.data<int8_t>(),
        item_size,
        const_param(batch_shape),
        const_param(a_batch_strides),
        const_param(b_batch_strides),
        const_param(c_batch_strides),
        batch_stride,
        ndim,
        batch_count);
  }

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
  execute(
      encoder,
      reinterpret_cast<void*>(out_pointers),
      reinterpret_cast<void*>(a_pointers),
      reinterpret_cast<void*>(b_pointers),
      reinterpret_cast<void*>(c_pointers),
      alpha,
      beta);
}

} // namespace mlx::core
