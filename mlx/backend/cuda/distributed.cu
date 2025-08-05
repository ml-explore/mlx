// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"
#include "mlx/backend/cuda/kernel_utils.cuh"


#include <cassert>

namespace mlx::core {
    namespace distributed {
    void AllReduce::eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) {
      // Here I assume for now that in is donatable and contiguous.
      // TODO
      assert(inputs.size() == 1);
      assert(outputs.size() == 1);

      auto& input = inputs[0];
      auto& output = outputs[0];

      auto& encoder = cu::get_command_encoder(stream());
      output.set_data(allocator::malloc(output.nbytes()));

      encoder.set_input_array(input);
      encoder.set_output_array(output);

      auto capture = encoder.capture_context();
      auto& s = stream();
      
      switch (reduce_type_) {
        case Sum:
          distributed::detail::all_sum(group(), input, output, s);
          break;
        case Max:
          distributed::detail::all_max(group(), input, output, s);
          break;
        case Min:
          distributed::detail::all_min(group(), input, output, s);
          break;
        default:
          throw std::runtime_error(
              "Only all reduce sum, max, and min are supported.");
      }
    }
    
    void Send::eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) {
      // Here FOR NOW I assume that it is always row_contigious
      // because not sure how to copy correctly
      // TODO
      assert(inputs.size() == 1);
      assert(outputs.size() == 1);
    
      distributed::detail::send(group(), inputs[0], dst_, stream());
      outputs[0].copy_shared_buffer(inputs[0]);
    }
    
    void Recv::eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) {
      assert(inputs.size() == 0);
      assert(outputs.size() == 1);
      outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
      distributed::detail::recv(group(), outputs[0], src_, stream());
    }
    
    void AllGather::eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) {
      // Here FOR NOW I assume that it is always row_contigious
      // because not sure how to copy correctly
      // TODO
      assert(inputs.size() == 1);
      assert(outputs.size() == 1);
    
      auto& input = inputs[0];
      auto& output = outputs[0];
    
      output.copy_shared_buffer(input);
      distributed::detail::all_gather(group(), input, output, stream());
    } 
    }// namespace distributed
}