// Copyright © 2026 Apple Inc.

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

id<MTLBuffer> metal_buffer(const array& a) {
  return (__bridge id<MTLBuffer>)(void*)a.buffer().ptr();
}

id<MTLDevice> metal_device(metal::Device& device) {
  return (__bridge id<MTLDevice>)(void*)device.mtl_device();
}

id<MTLCommandBuffer> metal_command_buffer(metal::CommandEncoder& encoder) {
  return (__bridge id<MTLCommandBuffer>)(void*)encoder.get_command_buffer();
}

MPSMatrixDescriptor*
matrix_descriptor(int rows, int columns, MPSDataType dtype) {
  const auto row_bytes = static_cast<NSUInteger>(columns) *
      (dtype == MPSDataTypeUInt32 ? sizeof(uint32_t) : sizeof(float));
  return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                               columns:columns
                                              rowBytes:row_bytes
                                             dataType:dtype];
}

} // namespace

void Inverse::eval_gpu(const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::invalid_argument(
        "[Inverse::eval_gpu] Metal inversion supports float32 arrays only.");
  }

  output.set_data(allocator::malloc(output.nbytes()));
  if (output.size() == 0) {
    return;
  }

  auto& encoder = metal::get_command_encoder(stream());
  const auto& input = inputs[0];
  const auto& rhs = inputs[1];

  const int order = input.shape(-1);
  const size_t batch_size = input.size() / (order * order);
  array lu(input.shape(), float32, nullptr, {});
  array pivots({static_cast<int>(batch_size), order}, uint32, nullptr, {});
  lu.set_data(allocator::malloc(lu.nbytes()));
  pivots.set_data(allocator::malloc(pivots.nbytes()));

  // MPS needs a separate command buffer after MLX compute work.
  encoder.end_encoding();
  auto retained_inputs = std::make_shared<std::vector<array>>(
      std::initializer_list<array>{input, rhs});
  auto input_command_buffer = metal_command_buffer(encoder);
  [input_command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
    (void)retained_inputs;
  }];
  encoder.commit();

  auto matrix = matrix_descriptor(order, order, MPSDataTypeFloat32);
  auto pivot_matrix = matrix_descriptor(1, order, MPSDataTypeUInt32);
  auto decomposition = [[MPSMatrixDecompositionLU alloc]
      initWithDevice:metal_device(metal::device(stream().device))
                rows:order
             columns:order];
  auto solve = [[MPSMatrixSolveLU alloc]
      initWithDevice:metal_device(metal::device(stream().device))
           transpose:NO
               order:order
numberOfRightHandSides:order];
  auto command_buffer = metal_command_buffer(encoder);
  auto retained_arrays = std::make_shared<std::vector<array>>(
      std::initializer_list<array>{input, rhs, lu, pivots});
  auto resources = [[NSMutableArray alloc] init];
  [resources addObject:matrix];
  [resources addObject:pivot_matrix];
  [resources addObject:decomposition];
  [resources addObject:solve];
  [decomposition release];
  [solve release];

  const auto matrix_bytes = static_cast<size_t>(order) * order * sizeof(float);
  const auto pivot_bytes = static_cast<size_t>(order) * sizeof(uint32_t);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto source = [[MPSMatrix alloc] initWithBuffer:metal_buffer(input)
                                             offset:input.offset() + batch * matrix_bytes
                                         descriptor:matrix];
    auto factor = [[MPSMatrix alloc] initWithBuffer:metal_buffer(lu)
                                             offset:lu.offset() + batch * matrix_bytes
                                         descriptor:matrix];
    auto right_hand_side = [[MPSMatrix alloc]
        initWithBuffer:metal_buffer(rhs)
                 offset:rhs.offset() + batch * matrix_bytes
             descriptor:matrix];
    auto solution = [[MPSMatrix alloc] initWithBuffer:metal_buffer(output)
                                               offset:output.offset() + batch * matrix_bytes
                                           descriptor:matrix];
    auto pivot_indices = [[MPSMatrix alloc]
        initWithBuffer:metal_buffer(pivots)
                 offset:pivots.offset() + batch * pivot_bytes
             descriptor:pivot_matrix];
    [resources addObject:source];
    [resources addObject:factor];
    [resources addObject:right_hand_side];
    [resources addObject:solution];
    [resources addObject:pivot_indices];
    [source release];
    [factor release];
    [right_hand_side release];
    [solution release];
    [pivot_indices release];

    [decomposition encodeToCommandBuffer:command_buffer
                             sourceMatrix:source
                             resultMatrix:factor
                             pivotIndices:pivot_indices
                                   status:nil];
    [solve encodeToCommandBuffer:command_buffer
                    sourceMatrix:factor
             rightHandSideMatrix:right_hand_side
                    pivotIndices:pivot_indices
                  solutionMatrix:solution];
  }
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
    (void)retained_arrays;
    [resources release];
  }];

  encoder.register_output_array(output);
  // Register the MPS result with MLX's command-encoder dependency tracking.
  encoder.set_input_array(output, 0);
  encoder.end_encoding();
}

} // namespace mlx::core
