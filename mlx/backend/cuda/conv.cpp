// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

// cudnn_frontend.h redefines this macro.
#undef CHECK_CUDA_ERROR

#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace {

// Not all engines support it so can not use this API now.
#define MLX_USE_CUDNN_NATIVE_CUDA_GRAPH_API 0

struct ConvCacheKey {
  int device_id;
  cudnnBackendDescriptorType_t backend_type;
  cudnnDataType_t cudnn_type;
  std::array<int, MAX_NDIM> input_shape;
  std::array<int, MAX_NDIM> filter_shape;
  std::array<int, MAX_NDIM> padding_lo;
  std::array<int, MAX_NDIM> padding_hi;
  std::array<int, MAX_NDIM> stride;
  std::array<int, MAX_NDIM> dilation;
  int groups;
  uint8_t input_alignment;
  uint8_t filter_alignment;
  uint8_t output_alignment;
};

auto& conv_cache() {
  static LRUBytesKeyCache<ConvCacheKey, cudnn_frontend::ExecutionPlan> cache(
      /* capacity */ 128);
  return cache;
}

template <typename T, typename U>
inline std::vector<T> convert_vector(const std::vector<U>& vec) {
  return std::vector<T>(vec.begin(), vec.end());
}

template <typename T>
inline std::array<T, MAX_NDIM> fixed_vector(const std::vector<T>& vec) {
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
  return std::make_tuple(shape, strides);
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
    const array& in,
    const array& wt,
    array& out) {
  int workspace_size = plan.getWorkspaceSize();
  array workspace(allocator::malloc(workspace_size), {workspace_size}, uint8);

  int64_t uids[3] = {'x', 'w', 'y'};
  void* data_ptrs[3] = {
      const_cast<void*>(in.data<void>()),
      const_cast<void*>(wt.data<void>()),
      out.data<void>(),
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
    cudnn_frontend::EngineConfigList& configs,
    const ConvCacheKey& cache_key,
    const std::string& op_graph_tag,
    const array& in,
    const array& wt,
    array& out) {
  for (auto& config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(encoder.device().cudnn_handle())
                      .setEngineConfig(config, op_graph_tag)
                      .build();
      if (execute_plan(encoder, plan, in, wt, out)) {
        conv_cache().emplace(cache_key, std::move(plan));
        return true;
      }
    } catch (cudnn_frontend::cudnnException&) {
    }
  }
  return false;
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  if (out.size() == 0) {
    return;
  }

  assert(inputs.size() == 2);
  array in = inputs[0];
  array wt = inputs[1];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // cuDNN requires contiguous input.
  // TODO: Handle NCHW format specially.
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in);
  }
  if (!wt.flags().row_contiguous) {
    wt = contiguous_copy_gpu(wt, s);
    encoder.add_temporary(wt);
  }

  encoder.set_input_array(in);
  encoder.set_input_array(wt);
  encoder.set_output_array(out);

  auto backend_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
  auto cudnn_type = dtype_to_cudnn_type(in.dtype());

  // Search cache.
  ConvCacheKey cache_key{
      encoder.device().cuda_device(),
      backend_type,
      cudnn_type,
      fixed_vector(in.shape()),
      fixed_vector(wt.shape()),
      fixed_vector(padding_lo_),
      fixed_vector(padding_hi_),
      fixed_vector(kernel_strides_),
      fixed_vector(kernel_dilation_),
      groups_,
      get_alignment(in),
      get_alignment(wt),
      get_alignment(out)};
  if (auto it = conv_cache().find(cache_key); it != conv_cache().end()) {
    if (!execute_plan(encoder, it->second, in, wt, out)) {
      throw std::runtime_error("Cached convolution plan failed to execute.");
    }
    return;
  }

  // Build operation graph.
  auto compute_data_type = (in.dtype() == float16 || in.dtype() == bfloat16)
      ? CUDNN_DATA_FLOAT
      : cudnn_type;

  auto stride = convert_vector<int64_t>(kernel_strides_);
  auto padding_lo = convert_vector<int64_t>(padding_lo_);
  auto padding_hi = convert_vector<int64_t>(padding_hi_);
  auto dilation = convert_vector<int64_t>(kernel_dilation_);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setDataType(compute_data_type)
                       .setMathMode(CUDNN_CROSS_CORRELATION)
                       .setNDims(stride.size())
                       .setStrides(stride.size(), stride.data())
                       .setPrePadding(padding_lo.size(), padding_lo.data())
                       .setPostPadding(padding_hi.size(), padding_hi.data())
                       .setDilation(dilation.size(), dilation.data())
                       .build();

  auto op = cudnn_frontend::OperationBuilder(backend_type)
                .setxDesc(build_tensor('x', in))
                .setwDesc(build_tensor('w', wt))
                .setyDesc(build_tensor('y', out))
                .setcDesc(conv_desc)
                .build();

  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(encoder.device().cudnn_handle())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();

  // Try to run plans based on heuristics.
  auto configs = get_engine_configs(backend_type, in.dtype(), op_graph);
  auto op_graph_tag = op_graph.getTag();
  if (try_engines(encoder, configs, cache_key, op_graph_tag, in, wt, out)) {
    return;
  }
  // Then try fallback plans.
  configs = get_engine_configs(backend_type, in.dtype(), op_graph);
  if (try_engines(encoder, configs, cache_key, op_graph_tag, in, wt, out)) {
    return;
  }
  throw std::runtime_error("Unable to find an engine for convolution.");
}

} // namespace mlx::core
