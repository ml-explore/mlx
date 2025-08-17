// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace {

// Not all engines support it so can not use this API now.
#define MLX_USE_CUDNN_NATIVE_CUDA_GRAPH_API 0

// Alias for better readability.
#define CONV_FORWARD CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
#define CONV_BACKWARD_INPUT \
  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
#define CONV_BACKWARD_WEIGHT \
  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR

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
      std::pair<cudnnBackendDescriptorType_t, cudnn_frontend::ExecutionPlan>>
      cache(/* capacity */ 128);
  return cache;
}

template <typename T, typename Vec>
inline SmallVector<T> convert_vector(const Vec& vec) {
  return SmallVector<T>(vec.begin(), vec.end());
}

template <typename T, template <typename U> class Vec>
inline std::array<T, MAX_NDIM> fixed_vector(const Vec<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error(
        fmt::format("ndim can not be larger than {}.", MAX_NDIM));
  }
  std::array<T, MAX_NDIM> result = {};
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

auto nhwc_to_nchw(const array& x) {
  auto shape = convert_vector<int64_t>(x.shape());
  shape.insert(shape.begin() + 1, shape.back());
  shape.erase(shape.end() - 1);
  auto strides = convert_vector<int64_t>(x.strides());
  strides.insert(strides.begin() + 1, strides.back());
  strides.erase(strides.end() - 1);
  return std::make_tuple(std::move(shape), std::move(strides));
}

inline cudnnDataType_t dtype_to_cudnn_type(Dtype dtype) {
  switch (dtype) {
    case int8:
      return CUDNN_DATA_INT8;
    case int32:
      return CUDNN_DATA_INT32;
    case uint8:
      return CUDNN_DATA_UINT8;
    case float16:
      return CUDNN_DATA_HALF;
    case bfloat16:
      return CUDNN_DATA_BFLOAT16;
    case float32:
      return CUDNN_DATA_FLOAT;
    case float64:
      return CUDNN_DATA_DOUBLE;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in Convolution: {}.", dtype_to_string(dtype)));
  }
}

inline uint8_t get_alignment(const array& x) {
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(x.data<void>());
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

inline cudnn_frontend::Tensor build_tensor(int64_t id, const array& x) {
  auto [shape, strides] = nhwc_to_nchw(x);
  return cudnn_frontend::TensorBuilder()
      .setDim(shape.size(), shape.data())
      .setStrides(strides.size(), strides.data())
      .setId(id)
      .setAlignment(get_alignment(x))
      .setDataType(dtype_to_cudnn_type(x.dtype()))
      .build();
}

cudnn_frontend::EngineConfigList get_engine_configs(
    cudnnBackendDescriptorType_t backend_type,
    Dtype dtype,
    cudnn_frontend::OperationGraph& op_graph,
    bool use_fallback = false) {
  cudnn_frontend::GeneratorSource source;
  if (use_fallback) {
    source = [&backend_type](cudnn_frontend::OperationGraph& op_graph) {
      auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                          .setOperationGraph(op_graph)
                          .setOperation(backend_type)
                          .build();
      return fallback.getFallbackList();
    };
  } else {
    source = [](cudnn_frontend::OperationGraph& op_graph) {
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                            .setOperationGraph(op_graph)
                            .setHeurMode(CUDNN_HEUR_MODE_A)
                            .build();
      return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    };
  }

  cudnn_frontend::EngineConfigGenerator generator(1, &source);
  auto configs = generator.generate_engine_config(op_graph);

  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_frontend::filter(configs, filtered_configs, [dtype](auto c) {
    if (cudnn_frontend::hasNumericalNote<
            CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
      return true;
    }
    if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c) &&
        dtype == float32 && !env::enable_tf32()) {
      return true;
    }
    return false;
  });
  return filtered_configs;
}

bool execute_plan(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    array& x,
    array& w,
    array& y) {
  int workspace_size = plan.getWorkspaceSize();
  array workspace(allocator::malloc(workspace_size), {workspace_size}, uint8);

  int64_t uids[3] = {'x', 'w', 'y'};
  void* data_ptrs[3] = {
      x.data<void>(),
      w.data<void>(),
      y.data<void>(),
  };

  auto variantPack = cudnn_frontend::VariantPackBuilder()
                         .setWorkspacePointer(workspace.data<void>())
                         .setDataPointers(3, data_ptrs)
                         .setUids(3, uids)
                         .build();

  auto handle = encoder.device().cudnn_handle();
  cudnnSetStream(handle, encoder.stream());

#if CUDNN_VERSION >= 90500 && MLX_USE_CUDNN_NATIVE_CUDA_GRAPH_API
  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);
  std::unique_ptr<cudaGraph_t, void (*)(cudaGraph_t*)> graph_freer(
      &graph, [](cudaGraph_t* p) { cudaGraphDestroy(*p); });
  if (cudnnBackendPopulateCudaGraph(
          handle, plan.get_raw_desc(), variantPack.get_raw_desc(), graph) !=
      CUDNN_STATUS_SUCCESS) {
    return false;
  }
  encoder.add_graph_node(graph);
#else
  auto capture = encoder.capture_context();
  if (cudnnBackendExecute(
          handle, plan.get_raw_desc(), variantPack.get_raw_desc()) !=
      CUDNN_STATUS_SUCCESS) {
    // Discard the captured graph when failed.
    capture.discard = true;
    return false;
  }
#endif

  encoder.add_temporary(workspace);
  return true;
}

bool try_engines(
    cu::CommandEncoder& encoder,
    const ConvCacheKey& cache_key,
    cudnnBackendDescriptorType_t backend_type,
    cudnn_frontend::EngineConfigList& configs,
    const std::string& op_graph_tag,
    array& x,
    array& w,
    array& y) {
  for (auto& config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(encoder.device().cudnn_handle())
                      .setEngineConfig(config, op_graph_tag)
                      .build();
      if (execute_plan(encoder, plan, x, w, y)) {
        conv_cache().emplace(
            cache_key, std::make_pair(backend_type, std::move(plan)));
        return true;
      }
    } catch (cudnn_frontend::cudnnException& error) {
      if (error.getCudnnStatus() != CUDNN_STATUS_NOT_SUPPORTED) {
        throw;
      }
    }
  }
  return false;
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

std::optional<cudnn_frontend::OperationGraph> build_op_graph(
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
                  .setxDesc(build_tensor('x', x))
                  .setwDesc(build_tensor('w', w))
                  .setyDesc(build_tensor('y', y))
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
      fixed_vector(in.shape()),
      fixed_vector(wt.shape()),
      fixed_vector(kernel_strides_),
      fixed_vector(padding_lo_),
      fixed_vector(padding_hi_),
      fixed_vector(kernel_dilation_),
      groups_,
      flip_,
      get_alignment(in),
      get_alignment(wt),
      get_alignment(out)};
  if (auto it = conv_cache().find(cache_key); it != conv_cache().end()) {
    auto& [backend_type, plan] = it->second;
    std::tie(in, wt, out) =
        prepare_args(encoder, backend_type, in, wt, out, groups_, s);
    register_args(encoder, backend_type, in, wt, out, out_);
    auto [x, w, y] = dispatch_args(backend_type, in, wt, out);
    if (!execute_plan(encoder, plan, x, w, y)) {
      throw std::runtime_error("[conv] Cached plan failed to execute.");
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
    op_graph = build_op_graph(
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
  if (!op_graph) {
    throw std::runtime_error("[conv] Can not build op graph.");
  }

  // Get ready to execute the graph.
  register_args(encoder, backend_type, in, wt, out, out_);

  // Try to run plans based on heuristics.
  auto configs = get_engine_configs(backend_type, dtype, *op_graph);
  auto tag = op_graph->getTag();
  auto [x, w, y] = dispatch_args(backend_type, in, wt, out);
  if (try_engines(encoder, cache_key, backend_type, configs, tag, x, w, y)) {
    return;
  }
  // Then try fallback plans.
  configs = get_engine_configs(backend_type, dtype, *op_graph);
  if (try_engines(encoder, cache_key, backend_type, configs, tag, x, w, y)) {
    return;
  }
  throw std::runtime_error("[conv] Unable to find a working engine.");
}

} // namespace mlx::core
