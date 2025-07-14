// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/arange.cuh"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
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
  auto& encoder = cu::get_command_encoder(stream());
  encoder.set_output_array(out);
  auto capture = encoder.capture_context();
  dispatch_int_float_types(out.dtype(), "Arange", [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using OutType = cuda_type_t<CTYPE>;
    CTYPE step =
        static_cast<CTYPE>(start_ + step_) - static_cast<CTYPE>(start_);
    thrust::transform(
        cu::thrust_policy(encoder.stream()),
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(out.data_size()),
        thrust::device_pointer_cast(out.data<OutType>()),
        cu::Arange<OutType>{
            static_cast<OutType>(start_), static_cast<OutType>(step)});
  });
}

bool fast::ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  return true;
}

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

NO_GPU(BlockMaskedMM)
NO_GPU(Convolution)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(FFT)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Load)
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(SegmentedMM)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)

namespace fast {
NO_GPU(ScaledDotProductAttention)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace mlx::core
