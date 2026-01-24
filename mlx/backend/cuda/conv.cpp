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

enum ConvBackendType {
  CONV_FALLBACK,
  CONV_FORWARD,
  CONV_BACKWARD_INPUT,
  CONV_BACKWARD_WEIGHT,
};

struct ConvCacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
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
      std::pair<ConvBackendType, std::optional<DnnGraph>>>
      cache("MLX_CUDA_CONV_CACHE_SIZE", /* default_capacity */ 128);
  return cache;
}

auto get_conv_settings(
    ConvBackendType backend_type,
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
      int in_size = 1 + kernel_strides[i] * (y.shape(1 + i) - 1);
      int out_size = 1 + input_dilation[i] * (x.shape(1 + i) - 1);
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

std::optional<DnnGraph> build_conv_graph(
    cu::CommandEncoder& encoder,
    ConvBackendType backend_type,
    Dtype dtype,
    array& x,
    array& w,
    array& y,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding_lo,
    const std::vector<int64_t>& padding_hi,
    const std::vector<int64_t>& dilation) {
  auto compute_dtype =
      (dtype == float16 || dtype == bfloat16) ? float32 : dtype;
  DnnGraph graph(encoder.device().cudnn_handle(), dtype, compute_dtype);
  auto x_ = graph.tensor_nchw("X", 'x', x);
  auto w_ = graph.tensor_nchw("W", 'w', w);

  auto set_options = [&](auto& options) {
    options.set_compute_data_type(dtype_to_cudnn_type(compute_dtype))
        .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION)
        .set_stride(stride)
        .set_pre_padding(padding_lo)
        .set_post_padding(padding_hi)
        .set_dilation(dilation);
  };

  std::shared_ptr<fe::graph::Tensor_attributes> y_;
  if (backend_type == CONV_FORWARD) {
    auto options = fe::graph::Conv_fprop_attributes();
    set_options(options);
    y_ = graph.conv_fprop(x_, w_, options);
  } else if (backend_type == CONV_BACKWARD_INPUT) {
    auto options = fe::graph::Conv_dgrad_attributes();
    set_options(options);
    y_ = graph.conv_dgrad(x_, w_, options);
  } else if (backend_type == CONV_BACKWARD_WEIGHT) {
    auto options = fe::graph::Conv_wgrad_attributes();
    set_options(options);
    y_ = graph.conv_wgrad(w_, x_, options);
  }
  graph.tensor_nchw(y_, 'y', y)->set_output(true);

  if (graph.prepare().is_bad()) {
    return std::nullopt;
  }
  graph.deselect_numeric_notes({fe::NumericalNote_t::DOWN_CONVERT_INPUTS});
  if (dtype == float32 && !env::enable_tf32()) {
    graph.deselect_numeric_notes({fe::NumericalNote_t::TENSOR_CORE});
  }
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
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
    ConvBackendType backend_type,
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

// Register inputs and outputs before actually running conv op. Can only be
// called once per eval_gpu.
void register_args(
    cu::CommandEncoder& encoder,
    ConvBackendType backend_type,
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
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  array in = inputs[0];
  array wt = inputs[1];
  array out = out_;
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  Dtype dtype = out.dtype();

  // Search cache.
  BytesKey<ConvCacheKey> cache_key;
  cache_key.pod = {
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
    auto& [backend_type, graph] = it->second;
    if (graph) {
      // Run cached graph.
      std::tie(in, wt, out) =
          prepare_args(encoder, backend_type, in, wt, out, groups_, s);
      register_args(encoder, backend_type, in, wt, out, out_);
      CHECK_CUDNN_FE_ERROR(graph->encode_capturing(
          encoder,
          {
              {'x', gpu_ptr<void>(in)},
              {'w', gpu_ptr<void>(wt)},
              {'y', gpu_ptr<void>(out)},
          }));
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
  SmallVector<ConvBackendType, 2> try_backends;
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
  ConvBackendType backend_type;
  std::optional<DnnGraph> graph;
  for (auto try_backend : try_backends) {
    auto [x, w, y] =
        prepare_args(encoder, try_backend, in, wt, out, groups_, s);
    auto [stride, padding_lo, padding_hi, dilation] = get_conv_settings(
        try_backend,
        x,
        w,
        y,
        kernel_strides_,
        padding_lo_,
        padding_hi_,
        kernel_dilation_,
        input_dilation_);
    graph = build_conv_graph(
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
    if (graph) {
      backend_type = try_backend;
      in = std::move(x);
      wt = std::move(w);
      out = std::move(y);
      break;
    }
  }

  if (graph) {
    register_args(encoder, backend_type, in, wt, out, out_);
    CHECK_CUDNN_FE_ERROR(graph->encode_capturing(
        encoder,
        {
            {'x', gpu_ptr<void>(in)},
            {'w', gpu_ptr<void>(wt)},
            {'y', gpu_ptr<void>(out)},
        }));
    conv_cache().emplace(
        cache_key, std::make_pair(backend_type, std::move(*graph)));
    return;
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
