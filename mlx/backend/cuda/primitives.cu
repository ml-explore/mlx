// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
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

NO_GPU(Abs)
NO_GPU(Add)
NO_GPU(ArcCos)
NO_GPU(ArcCosh)
NO_GPU(ArcSin)
NO_GPU(ArcSinh)
NO_GPU(ArcTan)
NO_GPU(ArcTan2)
NO_GPU(ArcTanh)
NO_GPU(ArgPartition)
NO_GPU(ArgReduce)
NO_GPU(ArgSort)
NO_GPU(BitwiseBinary)
NO_GPU(BitwiseInvert)
NO_GPU(BlockMaskedMM)
NO_GPU(Ceil)
NO_GPU_MULTI(Compiled)
NO_GPU(Conjugate)
NO_GPU(Convolution)
NO_GPU(Cos)
NO_GPU(Cosh)
NO_GPU(Divide)
NO_GPU_MULTI(DivMod)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(Remainder)
NO_GPU(Equal)
NO_GPU(Erf)
NO_GPU(ErfInv)
NO_GPU(Exp)
NO_GPU(Expm1)
NO_GPU(FFT)
NO_GPU(Floor)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Greater)
NO_GPU(GreaterEqual)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Less)
NO_GPU(LessEqual)
NO_GPU(Load)
NO_GPU(Log)
NO_GPU(Log1p)
NO_GPU(LogicalNot)
NO_GPU(LogicalAnd)
NO_GPU(LogicalOr)
NO_GPU(LogAddExp)
NO_GPU(LogSumExp)
NO_GPU_MULTI(LUF)
NO_GPU(Maximum)
NO_GPU(Minimum)
NO_GPU(Multiply)
NO_GPU(Negative)
NO_GPU(NotEqual)
NO_GPU(Partition)
NO_GPU(Power)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(RandomBits)
NO_GPU(Real)
NO_GPU(Reduce)
NO_GPU(Round)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(Sigmoid)
NO_GPU(Sign)
NO_GPU(Sin)
NO_GPU(Sinh)
NO_GPU(SliceUpdate)
NO_GPU(Softmax)
NO_GPU(Sort)
NO_GPU(Square)
NO_GPU(Sqrt)
NO_GPU(Subtract)
NO_GPU_MULTI(SVD)
NO_GPU(Tan)
NO_GPU(Tanh)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)

namespace fast {
NO_GPU_USE_FALLBACK(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_USE_FALLBACK(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_USE_FALLBACK(RoPE)
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
