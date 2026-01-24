// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/conv/conv.h"
#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace {

// Alias for better readability.
#define CONV_FORWARD CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
#define CONV_BACKWARD_INPUT \
  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
#define CONV_BACKWARD_WEIGHT \
  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR

// Custom placeholder representing fallback kernel.
#define CONV_FALLBACK static_cast<cudnnBackendDescriptorType_t>(-1)

struct ConvCacheKey {
  int device_id;
  cudnnDataType_t cudnn_dtype;
  std::array<int, MAX_NDIM> input_shape;
  std::array<int, MAX_NDIM> weight_shape;
  std::array<int, MAX_NDIM> stride;
  std::array<int, MAX_NDIM> padding_lo;
  std::array<int, MAX_NDIM> padding_hi;
  std::array<int, MAX_NDIM> dilation;
  int groups;
  bool flip;
  uint8_t input_alignment;
  uint8_t weight_alignment;
  uint8_t output_alignment;
};

auto& conv_cache() {
  static LRUBytesKeyCache<
      ConvCacheKey,
      std::pair<
          cudnnBackendDescriptorType_t,
          std::optional<cudnn_frontend::ExecutionPlan>>>
      cache("MLX_CUDA_CONV_CACHE_SIZE", /* default_capacity */ 128);
  return cache;
}

auto get_conv_op_settings(
    cudnnBackendDescriptorType_t backend_type,
    array& x,
    array& w,
    array& y,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo_,
    const std::vector<int>& padding_hi_,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation) {
  auto padding_lo = convert_vector<int64_t>(padding_lo_);
  auto padding_hi = convert_vector<int64_t>(padding_hi_);

  if (backend_type == CONV_BACKWARD_INPUT) {
    for (int i = 0; i < padding_lo.size(); ++i) {
      int wt_size = 1 + kernel_dilation[i] * (w.shape(1 + i) - 1);
      padding_lo[i] = wt_size - padding_lo[i] - 1;
      int in_size = 1 + kernel_strides[i] * (x.shape(1 + i) - 1);
      int out_size = 1 + input_dilation[i] * (y.shape(1 + i) - 1);
      padding_hi[i] = out_size - in_size + padding_hi[i];
    }
    return std::make_tuple(
        convert_vector<int64_t>(input_dilation),
        std::move(padding_lo),
        std::move(padding_hi),
        convert_vector<int64_t>(kernel_dilation));

  } else if (backend_type == CONV_BACKWARD_WEIGHT) {
    padding_hi = padding_lo;
    return std::make_tuple(
        convert_vector<int64_t>(kernel_dilation),
        std::move(padding_lo),
        std::move(padding_hi),
        convert_vector<int64_t>(kernel_strides));

  } else {
    return std::make_tuple(
        convert_vector<int64_t>(kernel_strides),
        std::move(padding_lo),
        std::move(padding_hi),
        convert_vector<int64_t>(kernel_dilation));
  }
}

std::optional<cudnn_frontend::OperationGraph> build_conv_op_graph(
    cu::CommandEncoder& encoder,
    cudnnBackendDescriptorType_t backend_type,
    Dtype dtype,
    array& x,
    array& w,
    array& y,
    const SmallVector<int64_t>& stride,
    const SmallVector<int64_t>& padding_lo,
    const SmallVector<int64_t>& padding_hi,
    const SmallVector<int64_t>& dilation) {
  try {
    auto compute_dtype = (dtype == float16 || dtype == bfloat16)
        ? CUDNN_DATA_FLOAT
        : dtype_to_cudnn_type(dtype);
    auto conv_desc = cudnn_frontend::ConvDescBuilder()
                         .setDataType(compute_dtype)
                         .setMathMode(CUDNN_CROSS_CORRELATION)
                         .setNDims(stride.size())
                         .setStrides(stride.size(), stride.data())
                         .setPrePadding(padding_lo.size(), padding_lo.data())
                         .setPostPadding(padding_hi.size(), padding_hi.data())
                         .setDilation(dilation.size(), dilation.data())
                         .build();

    auto op = cudnn_frontend::OperationBuilder(backend_type)
                  .setxDesc(build_cudnn_tensor_nchw('x', x))
                  .setwDesc(build_cudnn_tensor_nchw('w', w))
                  .setyDesc(build_cudnn_tensor_nchw('y', y))
                  .setcDesc(conv_desc)
                  .build();

    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
    return cudnn_frontend::OperationGraphBuilder()
        .setHandle(encoder.device().cudnn_handle())
        .setOperationGraph(ops.size(), ops.data())
        .build();
  } catch (cudnn_frontend::cudnnException& error) {
    if (error.getCudnnStatus() != CUDNN_STATUS_BAD_PARAM) {
      throw;
    }
    return std::nullopt;
  }
}

// Transpose from (C_out, H, W, C_in / groups) to (C_in, H, W, C_out / groups).
array group_transpose(
    const array& x,
    int groups,
    int group_dim,
    int axis1,
    int axis2,
    Stream s) {
  if (groups == 1) {
    return swapaxes_in_eval(x, axis1, axis2);
  }
  int ndim = x.ndim();
  if (group_dim < 0) {
    group_dim += ndim;
  }
  if (axis1 < 0) {
    axis1 += ndim;
  }
  if (axis2 < 0) {
    axis2 += ndim;
  }
  if (group_dim <= axis1) {
    axis1 += 1;
  }
  if (group_dim <= axis2) {
    axis2 += 1;
  }
  auto shape = x.shape();
  shape.insert(shape.begin() + group_dim, groups);
  shape[group_dim + 1] = shape[group_dim + 1] / groups;
  array x_trans = reshape_in_eval(x, std::move(shape), s);
  x_trans = swapaxes_in_eval(x_trans, axis1, axis2);
  x_trans = flatten_in_eval(x_trans, group_dim, group_dim + 1, s);
  return x_trans;
}

// Do necessary transposes and copies to prepare the inputs and outputs for
// building the cuDNN conv op. It is safe to be called multiple times in one
// eval_gpu, with cost of possible redundant copies.
std::tuple<array, array, array> prepare_args(
    cu::CommandEncoder& encoder,
    cudnnBackendDescriptorType_t backend_type,
    array in,
    array wt,
    array out,
    int groups,
    Stream s) {
  // Transpose the args depending on the backend type.
  // TODO: Handle groups.
  if (backend_type == CONV_BACKWARD_INPUT) {
    wt = group_transpose(wt, groups, 0, 0, -1, s);
  } else if (backend_type == CONV_BACKWARD_WEIGHT) {
    in = group_transpose(in, groups, -1, 0, -1, s);
    wt = swapaxes_in_eval(wt, 0, -1);
    // Create a contiguous array that shares the data with |out|, but with dim
    // C_in and C_out swapped.
    Shape shape(out.shape());
    std::swap(shape.front(), shape.back());
    Strides strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    array intermediate(std::move(shape), out.dtype(), nullptr, {});
    intermediate.copy_shared_buffer(
        out, std::move(strides), {true, true, false}, out.data_size());
    out = intermediate;
  }

  // cuDNN requires contiguous input.
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in);
  }
  if (!wt.flags().row_contiguous) {
    wt = contiguous_copy_gpu(wt, s);
    encoder.add_temporary(wt);
  }

  return {std::move(in), std::move(wt), std::move(out)};
}

// Get the x/w/y args from the in/wt/out args depending on backend type.
inline std::tuple<array&, array&, array&> dispatch_args(
    cudnnBackendDescriptorType_t backend_type,
    array& in,
    array& wt,
    array& out) {
  switch (backend_type) {
    case CONV_BACKWARD_INPUT:
      return {out, wt, in};
    case CONV_BACKWARD_WEIGHT:
      return {in, out, wt};
    default:
      return {in, wt, out};
  }
}

// Register inputs and outputs before actually running conv op. Can only be
// called once per eval_gpu.
void register_args(
    cu::CommandEncoder& encoder,
    cudnnBackendDescriptorType_t backend_type,
    array& in,
    array& wt,
    array& intermediate_out,
    array& final_out) {
  encoder.set_input_array(in);
  encoder.set_input_array(wt);
  encoder.set_output_array(final_out);

  if (backend_type == CONV_BACKWARD_WEIGHT) {
    // Turn |out| into a strided array, which will have C_in and C_out swapped
    // in vjp and the final |grad_weight| will then be contiguous.
    Strides strides = intermediate_out.strides();
    std::swap(strides.front(), strides.back());
    final_out.copy_shared_buffer(
        intermediate_out,
        std::move(strides),
        {false, false, false},
        intermediate_out.data_size());
  }
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out_) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  if (out_.size() == 0) {
    return;
  }

  assert(inputs.size() == 2);
  array in = inputs[0];
  array wt = inputs[1];
  array out = out_;
  out.set_data(allocator::malloc(out.nbytes()));
  Dtype dtype = out.dtype();

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Search cache.
  ConvCacheKey cache_key{
      encoder.device().cuda_device(),
      dtype_to_cudnn_type(dtype),
      vector_key(in.shape()),
      vector_key(wt.shape()),
      vector_key(kernel_strides_),
      vector_key(padding_lo_),
      vector_key(padding_hi_),
      vector_key(kernel_dilation_),
      groups_,
      flip_,
      get_alignment(in),
      get_alignment(wt),
      get_alignment(out)};
  if (auto it = conv_cache().find(cache_key); it != conv_cache().end()) {
    auto& [backend_type, plan] = it->second;
    if (plan) {
      // Run cached plan.
      std::tie(in, wt, out) =
          prepare_args(encoder, backend_type, in, wt, out, groups_, s);
      register_args(encoder, backend_type, in, wt, out, out_);
      auto [x, w, y] = dispatch_args(backend_type, in, wt, out);
      if (!encode_cudnn_plan(encoder, *plan, {'x', 'w', 'y'}, x, w, y)) {
        throw std::runtime_error("[conv] Cached plan failed to execute.");
      }
    } else {
      // Run fallback kernel.
      gemm_conv(
          encoder,
          in,
          wt,
          out,
          kernel_strides_,
          padding_lo_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          s);
    }
    return;
  }

  // There is no reliable way to deduce the proper cuDNN backend for the
  // convolution, so we make a best guess and then try.
  SmallVector<cudnnBackendDescriptorType_t, 2> try_backends;
  if (flip_) {
    // When weight is flipped, we assume it is backward input convolution.
    try_backends.push_back(CONV_BACKWARD_INPUT);
  } else {
    // Otherwise it could be backward weight convolution or forward convolution,
    // mathematically there is no difference so we have to use heuristics.
    // Empirically backward convolutions have large kernel dimensions, and
    // usually have |in| and |wt| transposed.
    if (!in.flags().row_contiguous && !wt.flags().row_contiguous &&
        wt.shape(2) > out.shape(2)) {
      try_backends = {CONV_BACKWARD_WEIGHT, CONV_FORWARD};
    } else {
      try_backends = {CONV_FORWARD, CONV_BACKWARD_WEIGHT};
    }
  }

  // Try to build op graph.
  cudnnBackendDescriptorType_t backend_type;
  std::optional<cudnn_frontend::OperationGraph> op_graph;
  for (auto try_backend : try_backends) {
    auto [in_copy, wt_copy, out_copy] =
        prepare_args(encoder, try_backend, in, wt, out, groups_, s);
    auto [x, w, y] = dispatch_args(try_backend, in_copy, wt_copy, out_copy);
    auto [stride, padding_lo, padding_hi, dilation] = get_conv_op_settings(
        try_backend,
        x,
        w,
        y,
        kernel_strides_,
        padding_lo_,
        padding_hi_,
        kernel_dilation_,
        input_dilation_);
    op_graph = build_conv_op_graph(
        encoder,
        try_backend,
        dtype,
        x,
        w,
        y,
        stride,
        padding_lo,
        padding_hi,
        dilation);
    if (op_graph) {
      backend_type = try_backend;
      in = std::move(in_copy);
      wt = std::move(wt_copy);
      out = std::move(out_copy);
      break;
    }
  }

  if (op_graph) {
    // Setup inputs and outputs.
    register_args(encoder, backend_type, in, wt, out, out_);

    // Find a plan for the graph and execute it.
    auto plan = find_cudnn_plan_from_op_graph(
        encoder.device().cudnn_handle(), backend_type, dtype, *op_graph);
    if (!plan) {
      throw std::runtime_error("[conv] Unable to find an execution plan.");
    }
    auto [x, w, y] = dispatch_args(backend_type, in, wt, out);
    if (encode_cudnn_plan(encoder, *plan, {'x', 'w', 'y'}, x, w, y)) {
      conv_cache().emplace(
          cache_key, std::make_pair(backend_type, std::move(*plan)));
      return;
    }
  }

  // Use fallback kernel for settings not supported by cuDNN.
  gemm_conv(
      encoder,
      in,
      wt,
      out,
      kernel_strides_,
      padding_lo_,
      kernel_dilation_,
      input_dilation_,
      groups_,
      flip_,
      s);
  conv_cache().emplace(cache_key, std::make_pair(CONV_FALLBACK, std::nullopt));
}

} // namespace mlx::core
