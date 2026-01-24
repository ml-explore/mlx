// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

namespace {

// Create a cudnn tensor descriptor.
template <typename Vec>
inline cudnn_frontend::Tensor build_cudnn_tensor(
    int64_t id,
    const array& x,
    const Vec& shape,
    const Vec& strides) {
  return cudnn_frontend::TensorBuilder()
      .setDim(shape.size(), shape.data())
      .setStrides(strides.size(), strides.data())
      .setId(id)
      .setAlignment(get_alignment(x))
      .setDataType(dtype_to_cudnn_type(x.dtype()))
      .build();
}

// In MLX a singleton dim (shape[dim] == 1) can have any stride, but in cuDNN
// whether a tensor is contiguous is determined with:
// shape[dim] == shape[dim + 1] * strides[dim + 1]
// So a contiguous array with singleton dims in MLX may be mistakenly treated
// as strided in cuDNN, and we work around it by normalizing the strides.
Strides normalized_strides(const array& x) {
  if (!x.flags().row_contiguous || x.ndim() < 2) {
    return x.strides();
  }
  Strides strides = x.strides();
  for (int i = x.ndim() - 2; i >= 0; --i) {
    if (x.shape(i) == 1) {
      strides[i] = x.shape(i + 1) * strides[i + 1];
    }
  }
  return strides;
}

// Return the shape and strides after transposing from NHWC to NCHW.
auto nhwc_to_nchw(SmallVector<int64_t> shape, SmallVector<int64_t> strides) {
  assert(shape.size() >= 3);
  shape.insert(shape.begin() + 1, shape.back());
  shape.erase(shape.end() - 1);
  strides.insert(strides.begin() + 1, strides.back());
  strides.erase(strides.end() - 1);
  return std::make_tuple(std::move(shape), std::move(strides));
}

inline auto nhwc_to_nchw(const array& x) {
  return nhwc_to_nchw(
      convert_vector<int64_t>(x.shape()), normalized_strides(x));
}

// Return available engines for a |op_graph|.
cudnn_frontend::EngineConfigList get_cudnn_engine_configs(
    cudnnBackendDescriptorType_t backend_type,
    Dtype dtype,
    cudnn_frontend::OperationGraph& op_graph,
    bool use_fallback = true) {
  SmallVector<cudnn_frontend::GeneratorSource, 2> sources;
  sources.push_back([](auto& op_graph) {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(op_graph)
                          .setHeurMode(CUDNN_HEUR_MODE_A)
                          .build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  });
  if (use_fallback) {
    sources.push_back([&backend_type](auto& op_graph) {
      auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                          .setOperationGraph(op_graph)
                          .setOperation(backend_type)
                          .build();
      return fallback.getFallbackList();
    });
  }

  auto configs =
      cudnn_frontend::EngineConfigGenerator(sources.size(), sources.data())
          .generate_engine_config(op_graph);

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

// Take |engine_configs| and |op_graph| and find a working execution plans
// from them.
std::optional<cudnn_frontend::ExecutionPlan>
find_cudnn_plan_from_engine_configs(
    cudnnHandle_t handle,
    const cudnn_frontend::EngineConfigList& engine_configs,
    const cudnn_frontend::OperationGraph& op_graph) {
  auto op_graph_tag = op_graph.getTag();
  for (const auto& config : engine_configs) {
    try {
      return cudnn_frontend::ExecutionPlanBuilder()
          .setHandle(handle)
          .setEngineConfig(config, op_graph_tag)
          .build();
    } catch (cudnn_frontend::cudnnException& error) {
      if (error.getCudnnStatus() != CUDNN_STATUS_NOT_SUPPORTED) {
        throw;
      }
    }
  }
  return std::nullopt;
}

// Prepare workspace and args to execute plan.
template <typename F>
bool prepare_cudnn_plan(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    int num_args,
    const int64_t* uids,
    void** data_ptrs,
    F&& execute) {
  int workspace_size = plan.getWorkspaceSize();
  array workspace(
      workspace_size > 0 ? allocator::malloc(workspace_size)
                         : allocator::Buffer(nullptr),
      {workspace_size},
      uint8);

  auto args = cudnn_frontend::VariantPackBuilder()
                  .setWorkspacePointer(workspace.data<void>())
                  .setDataPointers(num_args, data_ptrs)
                  .setUids(num_args, uids)
                  .build();

  auto handle = encoder.device().cudnn_handle();
  cudnnSetStream(handle, encoder.stream());

  if (!execute(handle, plan.get_raw_desc(), args.get_raw_desc())) {
    return false;
  }

  encoder.add_temporary(workspace);
  return true;
}

} // namespace

cudnn_frontend::Tensor build_cudnn_tensor(int64_t id, const array& x) {
  auto shape = convert_vector<int64_t>(x.shape());
  return build_cudnn_tensor(id, x, shape, normalized_strides(x));
}

cudnn_frontend::Tensor build_cudnn_tensor_nchw(int64_t id, const array& x) {
  auto [shape, strides] = nhwc_to_nchw(x);
  return build_cudnn_tensor(id, x, shape, strides);
}

cudnn_frontend::Tensor build_cudnn_tensor_4d_nchw(int64_t id, const array& x) {
  if (x.ndim() == 0) {
    SmallVector<int64_t, 4> scalar_dims = {1, 1, 1, 1};
    return build_cudnn_tensor(id, x, scalar_dims, scalar_dims);
  }
  if (x.ndim() == 1) {
    int64_t s = x.shape(0);
    SmallVector<int64_t, 4> shape = {1, x.shape(0), 1, 1};
    SmallVector<int64_t, 4> strides = {s, 1, s, s};
    return build_cudnn_tensor(id, x, shape, strides);
  }
  if (x.ndim() == 2) {
    int64_t s =
        x.flags().row_contiguous ? x.shape(1) * x.strides(1) : x.strides(0);
    SmallVector<int64_t, 4> shape = {x.shape(0), x.shape(1), 1, 1};
    SmallVector<int64_t, 4> strides = {s, x.strides(1), s, s};
    return build_cudnn_tensor(id, x, shape, strides);
  }
  if (x.ndim() == 3 || x.ndim() == 4) {
    return build_cudnn_tensor_nchw(id, x);
  }
  throw std::runtime_error(
      fmt::format("Unsupported array with {} dims.", x.ndim()));
}

cudnn_frontend::Tensor build_cudnn_scalar_4d(int64_t id, Dtype dtype) {
  SmallVector<int64_t, 4> scalar_dims = {1, 1, 1, 1};
  return cudnn_frontend::TensorBuilder()
      .setDim(scalar_dims.size(), scalar_dims.data())
      .setStrides(scalar_dims.size(), scalar_dims.data())
      .setId(id)
      .setAlignment(16)
      .setDataType(dtype_to_cudnn_type(dtype))
      .setByValue(true)
      .build();
}

std::optional<cudnn_frontend::ExecutionPlan> find_cudnn_plan_from_op_graph(
    cudnnHandle_t handle,
    cudnnBackendDescriptorType_t backend_type,
    Dtype dtype,
    cudnn_frontend::OperationGraph& op_graph) {
  auto engine_configs = get_cudnn_engine_configs(backend_type, dtype, op_graph);
  return find_cudnn_plan_from_engine_configs(handle, engine_configs, op_graph);
}

bool encode_cudnn_plan_with_capturing(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    int num_args,
    const int64_t* uids,
    void** data_ptrs) {
  return prepare_cudnn_plan(
      encoder,
      plan,
      num_args,
      uids,
      data_ptrs,
      [&](auto handle, auto plan, auto args) {
        auto capture = encoder.capture_context();
        if (cudnnBackendExecute(handle, plan, args) != CUDNN_STATUS_SUCCESS) {
          // Discard the captured graph when failed.
          capture.discard = true;
          return false;
        }
        return true;
      });
}

#if CUDNN_VERSION >= 90500
bool encode_cudnn_plan_with_graph_api(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    CudaGraph& graph,
    int num_args,
    const int64_t* uids,
    void** data_ptrs) {
  return prepare_cudnn_plan(
      encoder,
      plan,
      num_args,
      uids,
      data_ptrs,
      [&](auto handle, auto plan, auto args) {
        if (!graph) {
          graph = CudaGraph(encoder.device());
          if (cudnnBackendPopulateCudaGraph(handle, plan, args, graph) !=
              CUDNN_STATUS_SUCCESS) {
            return false;
          }
        } else {
          if (cudnnBackendUpdateCudaGraph(handle, plan, args, graph) !=
              CUDNN_STATUS_SUCCESS) {
            return false;
          }
        }
        encoder.add_graph_node(graph);
        return true;
      });
}
#endif

} // namespace mlx::core
