// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace mlx::core {

namespace cu {

template <typename T>
struct Arange {
  const T start;
  const T step;

  __device__ T operator()(uint32_t i) const {
    return start + i * step;
  }
};

} // namespace cu

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Arange::eval_gpu");
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));

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

} // namespace mlx::core
