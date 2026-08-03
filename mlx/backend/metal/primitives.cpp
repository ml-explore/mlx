// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core {

template <typename T>
void arange_set_scalars(T start, T next, metal::CommandEncoder& enc) {
  enc.set_bytes(start, 0);
  T step = next - start;
  enc.set_bytes(step, 1);
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = get_arange_kernel(d, "arange" + type_to_name(out), out);
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(
      std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  switch (out.dtype()) {
    case bool_: // unsupported
      throw std::runtime_error("[Arange::eval_gpu] Does not support bool");
    case uint8:
      arange_set_scalars<uint8_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint16:
      arange_set_scalars<uint16_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint32:
      arange_set_scalars<uint32_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint64:
      arange_set_scalars<uint64_t>(start_, start_ + step_, compute_encoder);
      break;
    case int8:
      arange_set_scalars<int8_t>(start_, start_ + step_, compute_encoder);
      break;
    case int16:
      arange_set_scalars<int16_t>(start_, start_ + step_, compute_encoder);
      break;
    case int32:
      arange_set_scalars<int32_t>(start_, start_ + step_, compute_encoder);
      break;
    case int64:
      arange_set_scalars<int64_t>(start_, start_ + step_, compute_encoder);
      break;
    case float16:
      arange_set_scalars<float16_t>(start_, start_ + step_, compute_encoder);
      break;
    case float32:
      arange_set_scalars<float>(start_, start_ + step_, compute_encoder);
      break;
    case bfloat16:
      arange_set_scalars<bfloat16_t>(start_, start_ + step_, compute_encoder);
      break;
    default:
      throw std::runtime_error("[Arange::eval_gpu] Does not support type.");
  }

  compute_encoder.set_output_array(out, 2);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string op_name;
  switch (reduce_type_) {
    case ArgReduce::ArgMin:
      op_name = "argmin_";
      break;
    case ArgReduce::ArgMax:
      op_name = "argmax_";
      break;
  }

  // Prepare the shapes, strides and axis arguments.
  auto in_strides = in.strides();
  auto shape = in.shape();
  auto out_strides = out.strides();
  auto axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];
  if (out_strides.size() == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  // ArgReduce
  int simd_size = 32;
  int n_reads = 4;
  auto& compute_encoder = metal::get_command_encoder(s);
  {
    auto kernel = d.get_kernel(op_name + type_to_name(in));
    NS::UInteger thread_group_size = std::min(
        (axis_size + n_reads - 1) / n_reads,
        kernel->maxTotalThreadsPerThreadgroup());
    // round up to the closest number divisible by simd_size
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    auto gd = get_2d_grid_dims(out.shape(), out.strides());
    MTL::Size grid_dims = MTL::Size(thread_group_size, gd.width, gd.height);
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    if (ndim == 0) {
      // Pass place holders so metal doesn't complain
      int shape_ = 0;
      int64_t stride_ = 0;
      compute_encoder.set_bytes(shape_, 2);
      compute_encoder.set_bytes(stride_, 3);
      compute_encoder.set_bytes(stride_, 4);
    } else {
      compute_encoder.set_vector_bytes(shape, 2);
      compute_encoder.set_vector_bytes(in_strides, 3);
      compute_encoder.set_vector_bytes(out_strides, 4);
    }
    compute_encoder.set_bytes(ndim, 5);
    compute_encoder.set_bytes(axis_stride, 6);
    compute_encoder.set_bytes(axis_size, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Load::eval_gpu] Not implemented.");
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  size_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_per_key / 2;
  bool odd = out_per_key % 2;

  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string kname = keys.flags().row_contiguous ? "rbitsc" : "rbits";
  auto kernel = d.get_kernel(kname);

  // organize into grid nkeys x elem_per_key
  MTL::Size grid_dims = MTL::Size(num_keys, half_size + odd, 1);
  auto group_dims = get_block_dims(num_keys, half_size + odd, 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(keys, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(odd, 2);
  compute_encoder.set_bytes(bytes_per_key, 3);

  if (!keys.flags().row_contiguous) {
    int ndim = keys.ndim();
    compute_encoder.set_bytes(ndim, 4);
    compute_encoder.set_vector_bytes(keys.shape(), 5);
    compute_encoder.set_vector_bytes(keys.strides(), 6);
  }

  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void RandomInt::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  auto& keys = inputs[0];
  auto& low = inputs[1];
  auto& high = inputs[2];
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  size_t n = out.size();
  size_t num_keys = keys.size() / 2;
  size_t elems_per_key = n / num_keys;
  auto& s = stream();
  auto& d = metal::device(s.device);
  bool is_signed = out.dtype().val() == Dtype::Val::int8 ||
      out.dtype().val() == Dtype::Val::int16 ||
      out.dtype().val() == Dtype::Val::int32 ||
      out.dtype().val() == Dtype::Val::int64;
  std::string kernel_name = is_signed ? "randint_signed" : "randint_unsigned";
  auto kernel = d.get_kernel(kernel_name);
  int dtype = 0;
  if (is_signed) {
    switch (out.dtype().val()) {
      case Dtype::Val::int8:
        dtype = 0;
        break;
      case Dtype::Val::int16:
        dtype = 1;
        break;
      case Dtype::Val::int32:
        dtype = 2;
        break;
      case Dtype::Val::int64:
        dtype = 3;
        break;
      default:
        throw std::runtime_error("Unsupported signed randint dtype.");
    }
  } else {
    switch (out.dtype().val()) {
      case Dtype::Val::bool_:
        dtype = 0;
        break;
      case Dtype::Val::uint8:
        dtype = 1;
        break;
      case Dtype::Val::uint16:
        dtype = 2;
        break;
      case Dtype::Val::uint32:
        dtype = 3;
        break;
      case Dtype::Val::uint64:
        dtype = 4;
        break;
      default:
        throw std::runtime_error("Unsupported unsigned randint dtype.");
    }
  }

  MTL::Size grid_dims = MTL::Size(n, 1, 1);
  auto group_dims = get_block_dims(n, 1, 1);
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(keys, 0);
  enc.set_input_array(low, 1);
  enc.set_input_array(high, 2);
  enc.set_output_array(out, 3);
  enc.set_bytes(n, 4);
  enc.set_bytes(elems_per_key, 5);
  int bounds_ndim = low.ndim();
  enc.set_bytes(bounds_ndim, 6);
  enc.set_vector_bytes(low.shape(), 7);
  enc.set_vector_bytes(low.strides(), 8);
  enc.set_vector_bytes(high.strides(), 9);
  int key_ndim = keys.ndim();
  enc.set_bytes(key_ndim, 10);
  enc.set_vector_bytes(keys.shape(), 11);
  enc.set_vector_bytes(keys.strides(), 12);
  enc.set_bytes(dtype, 13);
  enc.dispatch_threads(grid_dims, group_dims);
}

void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[QRF::eval_gpu] Metal QR factorization NYI.");
}

void SVD::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[SVD::eval_gpu] Metal SVD NYI.");
}

void Inverse::eval_gpu(const std::vector<array>& inputs, array& output) {
  throw std::runtime_error("[Inverse::eval_gpu] Metal inversion NYI.");
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[Cholesky::eval_gpu] Metal Cholesky decomposition NYI.");
}

void Eig::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Eig::eval_gpu] Metal Eig NYI.");
}

void Eigh::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Eigh::eval_gpu] Metal Eigh NYI.");
}

void LUF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[LUF::eval_gpu] Metal LU factorization NYI.");
}

} // namespace mlx::core
