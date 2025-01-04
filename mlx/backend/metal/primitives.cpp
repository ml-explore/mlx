// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/load.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/slicing.h"
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

void reshape(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
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
  auto& compute_encoder = d.get_command_encoder(s.index);
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
    case complex64:
      throw std::runtime_error("[Arange::eval_gpu] Does not support complex64");
  }

  compute_encoder.set_output_array(out, 2);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
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
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name + type_to_name(in));
    NS::UInteger thread_group_size = std::min(
        (axis_size + n_reads - 1) / n_reads,
        kernel->maxTotalThreadsPerThreadgroup());
    // round up to the closest number divisible by simd_size
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    size_t n_threads = out.size() * thread_group_size;
    MTL::Size grid_dims = MTL::Size(n_threads, 1, 1);
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

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  CopyType ctype =
      inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void BroadcastAxes::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  concatenate_gpu(inputs, out, axis_, stream());
}

void Contiguous::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.flags().row_contiguous ||
      (allow_col_major_ && in.flags().col_contiguous)) {
    move_or_copy(in, out);
  } else {
    copy_gpu(in, out, CopyType::General);
  }
}

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void CustomTransforms::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Depends::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto in = inputs[0];
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_gpu(in, out, ctype);
}

void ExpandDims::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Flatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void Unflatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto read_task = [out = out,
                    offset = offset_,
                    reader = reader_,
                    swap_endianness = swap_endianness_]() mutable {
    load(out, offset, reader, swap_endianness);
  };

  // Limit the size that the command buffer will wait on to avoid timing out
  // on the event (<4 seconds).
  if (out.nbytes() > (1 << 28)) {
    read_task();
    return;
  }
  auto fut = io::thread_pool().enqueue(std::move(read_task)).share();
  auto signal_task = [out = out, fut = std::move(fut)]() {
    fut.wait();
    out.event().signal();
  };
  scheduler::enqueue(io_stream(), std::move(signal_task));
  auto& d = metal::device(stream().device);
  d.end_encoding(stream().index);
  auto command_buffer = d.get_command_buffer(stream().index);
  command_buffer->encodeWait(
      static_cast<MTL::Event*>(out.event().raw_event().get()),
      out.event().value());
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Inputs must be base input array and scalar val array
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  // Padding value must be a scalar
  assert(val.size() == 1);

  // Padding value, input and output must be of the same type
  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  pad_gpu(in, val, out, axes_, low_pad_size_, stream());
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
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
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  auto group_dims = get_block_dims(num_keys, half_size + odd, 1);
  auto& compute_encoder = d.get_command_encoder(s.index);
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

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void Split::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  slice_gpu(in, out, start_indices_, strides_, stream());
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    move_or_copy(in, out);
    return;
  }

  // Check if materialization is needed
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  // Calculate out strides, initial offset and if copy needs to be made
  auto [data_offset, out_strides] = prepare_slice(in, start_indices_, strides_);

  // Do copy
  copy_gpu_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ upd.shape(),
      /* const Strides& i_strides = */ upd.strides(),
      /* const Strides& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      /* const Stream& s = */ stream());
}

void Squeeze::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
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

void Eigh::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Eigvalsh::eval_gpu] Metal Eigh NYI.");
}

void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto ibytes = size_of(in.dtype());
  auto obytes = size_of(out.dtype());
  // Conditions for buffer copying (disjunction):
  // - type size is the same
  // - type size is smaller and the last axis is contiguous
  // - the entire array is row contiguous
  if (ibytes == obytes || (obytes < ibytes && in.strides().back() == 1) ||
      in.flags().row_contiguous) {
    auto strides = in.strides();
    for (int i = 0; i < static_cast<int>(strides.size()) - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    move_or_copy(
        in, out, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(in.shape(), in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc_or_wait(tmp.nbytes()));
    copy_gpu_inplace(in, tmp, CopyType::General, stream());

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.move_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

} // namespace mlx::core
