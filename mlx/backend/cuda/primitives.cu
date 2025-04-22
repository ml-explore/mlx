// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/kernels/arange.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"
#include "mlx/backend/cuda/kernels/random.cuh"
#include "mlx/distributed/primitives.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <cassert>

namespace mlx::core {

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Arange::eval_gpu");
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_output_array(out);
  encoder.launch_kernel([&, this](cudaStream_t stream) {
    MLX_SWITCH_INT_FLOAT_TYPES_CHECKED(out.dtype(), "Arange", CTYPE, {
      using OutType = cuda_type_t<CTYPE>;
      CTYPE step =
          static_cast<CTYPE>(start_ + step_) - static_cast<CTYPE>(start_);
      thrust::transform(
          cu::thrust_policy(stream),
          thrust::counting_iterator<uint32_t>(0),
          thrust::counting_iterator<uint32_t>(out.data_size()),
          thrust::device_pointer_cast(out.data<OutType>()),
          cu::Arange<OutType>{
              static_cast<OutType>(start_), static_cast<OutType>(step)});
    });
  });
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomBits::eval_gpu");
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
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    dim3 grid_dim{
        static_cast<uint32_t>(num_keys),
        static_cast<uint32_t>(half_size + odd)};
    dim3 block_dim = get_block_dims(grid_dim.x, grid_dim.y, 1);
    dim3 num_blocks{
        cuda::ceil_div(grid_dim.x, block_dim.x),
        cuda::ceil_div(grid_dim.y, block_dim.y)};
    if (keys.flags().row_contiguous) {
      cu::rbitsc<<<num_blocks, block_dim, 0, stream>>>(
          keys.data<uint32_t>(),
          out.data<uint8_t>(),
          grid_dim,
          odd,
          bytes_per_key);
    } else {
      cu::rbits<<<num_blocks, block_dim, 0, stream>>>(
          keys.data<uint32_t>(),
          out.data<uint8_t>(),
          grid_dim,
          odd,
          bytes_per_key,
          keys.ndim(),
          const_param(keys.shape()),
          const_param(keys.strides()));
    }
  });
}

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

NO_GPU(ArgPartition)
NO_GPU(BlockMaskedMM)
NO_GPU_MULTI(Compiled)
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
NO_GPU(LogSumExp)
NO_GPU_MULTI(LUF)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(SliceUpdate)
NO_GPU(Softmax)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eig)
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
