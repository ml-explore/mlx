// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/arange.cuh"
#include "mlx/backend/cuda/kernels/arg_reduce.cuh"
#include "mlx/backend/cuda/kernels/random.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/slicing.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <assert.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

namespace mlx::core {

namespace {

void reshape(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
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

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto& s = stream();
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_output_array(out);
  encoder.launch_thrust([&, this](auto policy) {
    MLX_SWITCH_INT_FLOAT_TYPES_CHECKED(out.dtype(), "Arange", CTYPE, [&]() {
      using OutType = cuda_type_t<CTYPE>;
      CTYPE step =
          static_cast<CTYPE>(start_ + step_) - static_cast<CTYPE>(start_);
      thrust::transform(
          policy,
          thrust::counting_iterator<uint32_t>(0),
          thrust::counting_iterator<uint32_t>(out.data_size()),
          thrust::device_pointer_cast(out.data<OutType>()),
          mxcuda::Arange<OutType>{
              static_cast<OutType>(start_), static_cast<OutType>(step)});
    });
  });
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& s = stream();

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
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_REAL_TYPES_CHECKED(in.dtype(), "ArgReduce", CTYPE, [&]() {
      using InType = cuda_type_t<CTYPE>;
      constexpr uint32_t N_READS = 4;
      MLX_GET_BLOCK_DIM(ceil_div(axis_size, N_READS), BLOCK_DIM, {
        auto kernel = &mxcuda::arg_reduce_general<
            InType,
            mxcuda::ArgMax<InType>,
            BLOCK_DIM,
            N_READS>;
        if (reduce_type_ == ArgReduce::ArgMin) {
          kernel = &mxcuda::arg_reduce_general<
              InType,
              mxcuda::ArgMin<InType>,
              BLOCK_DIM,
              N_READS>;
        }
        kernel<<<out.data_size(), BLOCK_DIM, 0, stream>>>(
            in.data<InType>(),
            out.data<uint32_t>(),
            mxcuda::const_param(shape),
            mxcuda::const_param(in_strides),
            mxcuda::const_param(out_strides),
            ndim,
            axis_stride,
            axis_size);
      });
    });
  });
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
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    dim3 total_threads{
        static_cast<uint32_t>(num_keys),
        static_cast<uint32_t>(half_size + odd)};
    dim3 block_dim = get_block_dim(total_threads);
    dim3 num_blocks = ceil_div(total_threads, block_dim);
    if (keys.flags().row_contiguous) {
      mxcuda::rbitsc<<<num_blocks, block_dim, 0, stream>>>(
          keys.data<uint32_t>(), out.data<uint8_t>(), odd, bytes_per_key);
    } else {
      mxcuda::rbits<<<num_blocks, block_dim, 0, stream>>>(
          keys.data<uint32_t>(),
          out.data<uint8_t>(),
          odd,
          bytes_per_key,
          keys.ndim(),
          mxcuda::const_param(keys.shape()),
          mxcuda::const_param(keys.strides()));
    }
  });
}

// TODO: Code below are identical to backend/metal/primitives.cpp
void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  CopyType ctype =
      inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void BroadcastAxes::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Contiguous::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  constexpr size_t extra_bytes = 16384;
  if (in.buffer_size() <= out.nbytes() + extra_bytes &&
      (in.flags().row_contiguous ||
       (allow_col_major_ && in.flags().col_contiguous))) {
    out.copy_shared_buffer(in);
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

void ExpandDims::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
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

void Flatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
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

void Squeeze::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Unflatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
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
    out.copy_shared_buffer(
        in, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(in.shape(), in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    copy_gpu_inplace(in, tmp, CopyType::General, stream());

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.copy_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

NO_GPU(AddMM)
NO_GPU(ArgPartition)
NO_GPU(ArgSort)
NO_GPU(BlockMaskedMM)
NO_GPU_MULTI(Compiled)
NO_GPU(Concatenate)
NO_GPU(Convolution)
NO_GPU_MULTI(DivMod)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(FFT)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Load)
NO_GPU_MULTI(LUF)
NO_GPU(Matmul)
NO_GPU(Pad)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(SliceUpdate)
NO_GPU(Softmax)
NO_GPU(Sort)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)

namespace fast {
NO_GPU_MULTI(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_MULTI(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_MULTI(RoPE)
NO_GPU(ScaledDotProductAttention)
NO_GPU_MULTI(AffineQuantize)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace mlx::core
