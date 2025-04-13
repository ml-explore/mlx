// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/arange.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"
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

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

NO_GPU(AddMM)
NO_GPU(ArgPartition)
NO_GPU(ArgReduce)
NO_GPU(ArgSort)
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
NO_GPU(Matmul)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(RandomBits)
NO_GPU(Reduce)
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
