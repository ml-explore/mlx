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

// cudnn_frontend.h redefines this macro.
#undef CHECK_CUDNN_ERROR
#undef CHECK_CUDNN_FRONTEND_ERROR

namespace mlx::core {

namespace cu {

using namespace cudnn_frontend;

#define CHECK_CUDNN_FRONTEND_ERROR(cmd)                        \
  if (cmd.is_bad()) {                                          \
    throw std::runtime_error(fmt::format("{} failed.", #cmd)); \
  }

#define CHECK_CUDNN_ERROR(cmd) check_cudnn_error(#cmd, (cmd))

void check_cudnn_error(const char* name, cudnnStatus_t err) {
  if (err != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}.", name, cudnnGetErrorString(err)));
  }
}

auto swapaxes(const array& in, int axis1, int axis2) {
  std::vector<int> axes(in.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[axis1], axes[axis2]);
  std::vector<int64_t> shape(axes.size());
  std::vector<int64_t> strides(in.ndim());
  for (size_t ax = 0; ax < axes.size(); ++ax) {
    shape[ax] = in.shape()[axes[ax]];
    strides[ax] = in.strides()[axes[ax]];
  }
  return std::make_tuple(shape, strides);
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
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int groups)
      : handle_(device.cudnn_handle()) {
    auto cudnn_type = dtype_to_cudnn_type(dtype);
    bool is_half = dtype == float16 || dtype == bfloat16;

    graph_.set_io_data_type(cudnn_type)
        .set_compute_data_type(is_half ? DataType_t::FLOAT : cudnn_type);
    input_attr_ = graph_.tensor(graph::Tensor_attributes()
                                    .set_dim(input_shape)
                                    .set_stride(input_strides));
    filter_attr_ = graph_.tensor(graph::Tensor_attributes()
                                     .set_dim(filter_shape)
                                     .set_stride(filter_strides));

    auto conv_options = graph::Conv_fprop_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation);
    output_attr_ = graph_.conv_fprop(input_attr_, filter_attr_, conv_options);
    output_attr_->set_output(true);

    CHECK_CUDNN_FRONTEND_ERROR(graph_.validate());
    CHECK_CUDNN_FRONTEND_ERROR(graph_.build_operation_graph(handle_));
    CHECK_CUDNN_FRONTEND_ERROR(graph_.create_execution_plans({HeurMode_t::A}));
    CHECK_CUDNN_FRONTEND_ERROR(graph_.check_support(handle_));
    CHECK_CUDNN_FRONTEND_ERROR(graph_.build_plans(handle_));

#if 0
    int ndim = input_shape.size();
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc_));
    CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
        input_desc_,
        cudnn_type,
        ndim,
        input_shape.data(),
        input_strides.data()));

    CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&filter_desc_));
    CHECK_CUDNN_ERROR(cudnnSetFilterNdDescriptor(
        filter_desc_,
        cudnn_type,
        CUDNN_TENSOR_NCHW,
        ndim,
        filter_shape.data()));

    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc_));
    CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
        output_desc_,
        cudnn_type,
        ndim,
        output_shape.data(),
        output_strides.data()));

    CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CHECK_CUDNN_ERROR(cudnnSetConvolutionGroupCount(conv_desc_, groups));
    CHECK_CUDNN_ERROR(cudnnSetConvolutionNdDescriptor(
        conv_desc_,
        ndim - 2,
        padding.data(),
        stride.data(),
        dilation.data(),
        CUDNN_CROSS_CORRELATION,
        is_half ? CUDNN_DATA_FLOAT : cudnn_type));
    if (is_half) {
      CHECK_CUDNN_ERROR(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    } else if (dtype == float32) {
      CHECK_CUDNN_ERROR(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_FMA_MATH));
    } else {
      CHECK_CUDNN_ERROR(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));
    }

    std::vector<int> expected_output_shape(ndim);
    CHECK_CUDNN_ERROR(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc_,
        input_desc_,
        filter_desc_,
        ndim,
        expected_output_shape.data()));
    std::cout << "expected_output_shape: " << expected_output_shape
              << std::endl;

    cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int count;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
        handle_,
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        std::size(results),
        &count,
        results));
    for (int i = 0; i < count; ++i) {
      if (results[i].status == CUDNN_STATUS_SUCCESS) {
        algo_ = results[i].algo;
        std::cout << "Found algorithm" << std::endl;
        break;
      }
    }

    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
        handle_,
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        algo_,
        &workspace_size_));
#endif
  }

  ~Convolution() {
#if 0
    cudnnDestroyTensorDescriptor(input_desc_);
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyTensorDescriptor(output_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);
#endif
  }

  void run(
      cu::CommandEncoder& encoder,
      const void* input,
      const void* filter,
      void* output) {
    float alpha = 1;
    float beta = 0;

    array workspace(
        allocator::malloc(workspace_size_),
        {static_cast<int>(workspace_size_)},
        int8);
    encoder.add_temporary(workspace);

    std::unordered_map<int64_t, void*> ptr_map{
        {input_attr_->get_uid(), const_cast<void*>(input)},
        {filter_attr_->get_uid(), const_cast<void*>(filter)},
        {output_attr_->get_uid(), output}};

    auto capture = encoder.capture_context();
    CHECK_CUDNN_ERROR(cudnnSetStream(handle_, encoder.stream()));
    CHECK_CUDNN_FRONTEND_ERROR(
        graph_.execute(handle_, ptr_map, workspace.data<void>()));
#if 0
    CHECK_CUDNN_ERROR(cudnnConvolutionForward(
        handle_,
        &alpha,
        input_desc_,
        input,
        filter_desc_,
        filter,
        conv_desc_,
        algo_,
        workspace.data<void>(),
        workspace_size_,
        &beta,
        output_desc_,
        output));
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
  cudnnTensorDescriptor_t input_desc_{nullptr};
  cudnnFilterDescriptor_t filter_desc_{nullptr};
  cudnnTensorDescriptor_t output_desc_{nullptr};
  cudnnConvolutionDescriptor_t conv_desc_{nullptr};
  cudnnConvolutionFwdAlgo_t algo_{
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM};

  graph::Graph graph_;
  std::shared_ptr<graph::Tensor_attributes> input_attr_;
  std::shared_ptr<graph::Tensor_attributes> filter_attr_;
  std::shared_ptr<graph::Tensor_attributes> output_attr_;
  size_t workspace_size_{0};
};

} // namespace cu

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  const auto& in = inputs[0];
  const auto& wt = inputs[1];
  out.set_data(allocator::malloc(out.nbytes()));

  int ndim = in.ndim();
  auto [input_shape, input_strides] = cu::swapaxes(in, 1, ndim - 1);
  auto [filter_shape, filter_strides] = cu::swapaxes(wt, 1, ndim - 1);
  auto [output_shape, output_strides] = cu::swapaxes(out, 1, ndim - 1);

  cu::Convolution conv(
      cu::device(s.device),
      in.dtype(),
      input_shape,
      input_strides,
      filter_shape,
      filter_strides,
      output_shape,
      output_strides,
      std::vector<int64_t>(kernel_strides_.begin(), kernel_strides_.end()),
      std::vector<int64_t>(padding_lo_.begin(), padding_lo_.end()),
      std::vector<int64_t>(kernel_dilation_.begin(), kernel_dilation_.end()),
      groups_);
  conv.run(encoder, in.data<void>(), wt.data<void>(), out.data<void>());
}

} // namespace mlx::core
