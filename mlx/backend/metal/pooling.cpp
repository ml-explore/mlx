#include <cassert>
#include <cmath>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#include "mlx/primitives.h"

namespace mlx::core {
namespace {
void pool_1d(
    const array& in,
    array& out,
    const Stream& s,
    metal::Device& d,
    int kernel_size,
    int padding,
    int stride,
    int dilation,
    Pooling::PoolType type) {
  // TODO: add type support and AVG
  if (type == Pooling::PoolType::Max) {
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel("max_pool_1d_float");
    compute_encoder->setComputePipelineState(kernel);
    auto in_height = in.shape(1);
    auto out_height = out.shape(1);
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(&kernel_size, sizeof(int), 2);
    compute_encoder->setBytes(&stride, sizeof(int), 3);
    compute_encoder->setBytes(&padding, sizeof(int), 4);
    compute_encoder->setBytes(&in_height, sizeof(int), 5);
    compute_encoder->setBytes(&out_height, sizeof(int), 6);
    compute_encoder->setBytes(in.strides().data(), 3 * sizeof(size_t), 7);
    compute_encoder->setBytes(out.strides().data(), 3 * sizeof(size_t), 8);

    int dim0 = out.shape(0);
    int dim2 = out.shape(2);

    MTL::Size group_dims = get_block_dims(1, out_height, 1);
    MTL::Size grid_dims =
        MTL::Size(dim0, std::ceil(float(out_height) / group_dims.height), dim2);
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }
}

} // namespace

void Pooling::eval_gpu(const std::vector<array>& inputs, array& output) {
  assert(inputs.size() == 1);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  switch (inputs[0].ndim()) {
    case 3: {
      pool_1d(
          inputs[0],
          output,
          s,
          d,
          kernel_size_[0],
          padding_[0],
          stride_[0],
          dilation_[0],
          type_);
      return;
    }
    default: {
      throw std::runtime_error(
          "[Pooling] only 1D pooling only supported for now.");
    }
  }
}
} // namespace mlx::core