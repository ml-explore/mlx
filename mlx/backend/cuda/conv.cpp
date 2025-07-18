// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

// cudnn_frontend.h redefines this macro.
#undef CHECK_CUDA_ERROR

#include <cudnn_frontend.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace cu {

using namespace cudnn_frontend;

// The populate_cuda_graph API is supposed to integrate cuDNN with CUDA graph
// but it does not seems to support convolution yet.
#define CUDNN_CONV_SUPPORTS_CUDA_GRAPH_NATIVE_API 0

#define CHECK_CUDNN_FE_ERROR(cmd)                              \
  if (cmd.is_bad()) {                                          \
    throw std::runtime_error(fmt::format("{} failed.", #cmd)); \
  }

class Convolution {
 public:
  Convolution(
      Device& device,
      Dtype dtype,
      const std::vector<int64_t>& input_shape,
      const std::vector<int64_t>& input_strides,
      const std::vector<int64_t>& filter_shape,
      const std::vector<int64_t>& filter_strides,
      const std::vector<int64_t>& output_shape,
      const std::vector<int64_t>& output_strides,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding_lo,
      const std::vector<int64_t>& padding_hi,
      const std::vector<int64_t>& dilation)
      : handle_(device.cudnn_handle()) {
    auto cudnn_type = dtype_to_cudnn_type(dtype);
    bool is_half = dtype == float16 || dtype == bfloat16;

    graph_.set_io_data_type(cudnn_type)
        .set_intermediate_data_type(cudnn_type)
        .set_compute_data_type(is_half ? DataType_t::FLOAT : cudnn_type);
#if CUDNN_CONV_SUPPORTS_CUDA_GRAPH_NATIVE_API
    graph_.select_behavior_notes(
        {BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
#endif
    input_attr_ = graph_.tensor(graph::Tensor_attributes()
                                    .set_data_type(cudnn_type)
                                    .set_dim(input_shape)
                                    .set_stride(input_strides));
    filter_attr_ = graph_.tensor(graph::Tensor_attributes()
                                     .set_data_type(cudnn_type)
                                     .set_dim(filter_shape)
                                     .set_stride(filter_strides));

    auto conv_options = graph::Conv_fprop_attributes()
                            .set_pre_padding(padding_lo)
                            .set_post_padding(padding_hi)
                            .set_stride(stride)
                            .set_dilation(dilation);
    output_attr_ = graph_.conv_fprop(input_attr_, filter_attr_, conv_options);
    output_attr_->set_output(true)
        .set_data_type(cudnn_type)
        .set_dim(output_shape)
        .set_stride(output_strides);

    CHECK_CUDNN_FE_ERROR(graph_.validate());
    CHECK_CUDNN_FE_ERROR(graph_.build_operation_graph(handle_));
    CHECK_CUDNN_FE_ERROR(graph_.create_execution_plans({HeurMode_t::A}));
    CHECK_CUDNN_FE_ERROR(graph_.check_support(handle_));
    CHECK_CUDNN_FE_ERROR(graph_.build_plans(handle_));
    CHECK_CUDNN_FE_ERROR(graph_.get_workspace_size(workspace_size_));
  }

  void run(
      CommandEncoder& encoder,
      const void* input,
      const void* filter,
      void* output) {
    array workspace(
        allocator::malloc(workspace_size_),
        {static_cast<int>(workspace_size_)},
        int8);
    encoder.add_temporary(workspace);

    std::unordered_map<int64_t, void*> variant_pack{
        {input_attr_->get_uid(), const_cast<void*>(input)},
        {filter_attr_->get_uid(), const_cast<void*>(filter)},
        {output_attr_->get_uid(), output}};

#if CUDNN_CONV_SUPPORTS_CUDA_GRAPH_NATIVE_API
    cudaGraph_t cudnn_cuda_graph;
    cudaGraphCreate(&cudnn_cuda_graph, 0);
    CHECK_CUDNN_FE_ERROR(graph_.populate_cuda_graph(
        handle_, variant_pack, workspace.data<void>(), cudnn_cuda_graph));
    encoder.add_graph_node(cudnn_cuda_graph);
    cudaGraphDestroy(cudnn_cuda_graph);
#else
    auto capture = encoder.capture_context();
    CHECK_CUDNN_FE_ERROR(
        graph_.execute(handle_, variant_pack, workspace.data<void>()));
#endif
  }

 private:
  DataType_t dtype_to_cudnn_type(Dtype dtype) {
    switch (dtype) {
      case int8:
        return DataType_t::INT8;
      case int32:
        return DataType_t::INT32;
      case uint8:
        return DataType_t::UINT8;
      case float16:
        return DataType_t::HALF;
      case bfloat16:
        return DataType_t::BFLOAT16;
      case float32:
        return DataType_t::FLOAT;
      case float64:
        return DataType_t::DOUBLE;
      default:
        throw std::runtime_error(fmt::format(
            "Unsupported dtype in Convolution: {}.", dtype_to_string(dtype)));
    }
  }

  cudnnHandle_t handle_;
  graph::Graph graph_;
  std::shared_ptr<graph::Tensor_attributes> input_attr_;
  std::shared_ptr<graph::Tensor_attributes> filter_attr_;
  std::shared_ptr<graph::Tensor_attributes> output_attr_;
  int64_t workspace_size_{0};
};

} // namespace cu

namespace {

template <typename T, typename U>
inline std::vector<T> convert_vector(const std::vector<U>& vec) {
  return std::vector<T>(vec.begin(), vec.end());
}

auto nhwc_to_nchw(const array& in) {
  auto shape = convert_vector<int64_t>(in.shape());
  shape.insert(shape.begin() + 1, shape.back());
  shape.erase(shape.end() - 1);
  auto strides = convert_vector<int64_t>(in.strides());
  strides.insert(strides.begin() + 1, strides.back());
  strides.erase(strides.end() - 1);
  return std::make_tuple(shape, strides);
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  assert(inputs.size() == 2);
  const array& in = inputs[0];
  const array& wt = inputs[1];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_input_array(wt);
  encoder.set_output_array(out);

  // cuDNN requires dims to be passed as NCHW.
  auto [input_shape, input_strides] = nhwc_to_nchw(in);
  auto [filter_shape, filter_strides] = nhwc_to_nchw(wt);
  auto [output_shape, output_strides] = nhwc_to_nchw(out);

  cu::Convolution conv(
      cu::device(s.device),
      in.dtype(),
      input_shape,
      input_strides,
      filter_shape,
      filter_strides,
      output_shape,
      output_strides,
      convert_vector<int64_t>(kernel_strides_),
      convert_vector<int64_t>(padding_lo_),
      convert_vector<int64_t>(padding_hi_),
      convert_vector<int64_t>(kernel_dilation_));
  conv.run(encoder, in.data<void>(), wt.data<void>(), out.data<void>());
}

} // namespace mlx::core
