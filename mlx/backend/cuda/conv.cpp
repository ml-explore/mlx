// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
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

template <typename T, typename U>
inline std::vector<T> convert_vector(const std::vector<U>& vec) {
  return std::vector<T>(vec.begin(), vec.end());
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

inline int64_t get_alignment(const array& x) {
  int64_t alignment = 1;
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
    // In PyTorch, tf32 seems to be always turned off for convolution even with
    // "torch.backends.cudnn.allow_tf32 = True", so we are disabling tf32 too to
    // keep results same.
    if (dtype == float32 &&
        cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
      return true;
    }
    return false;
  });
  return filtered_configs;
}

bool execute_plan(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ManagedOpaqueDescriptor& config,
    const std::string& op_graph_tag,
    const array& in,
    const array& wt,
    array& out) {
  auto handle = encoder.device().cudnn_handle();
  auto plan = cudnn_frontend::ExecutionPlanBuilder()
                  .setHandle(handle)
                  .setEngineConfig(config, op_graph_tag)
                  .build();

  int64_t workspace_size = plan.getWorkspaceSize();
  array workspace(
      allocator::malloc(workspace_size),
      {static_cast<int>(workspace_size)},
      int8);

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

  auto capture = encoder.capture_context();
  if (cudnnBackendExecute(
          handle, plan.get_raw_desc(), variantPack.get_raw_desc()) !=
      CUDNN_STATUS_SUCCESS) {
    // Discard the captured graph when failed.
    capture.discard = true;
    return false;
  }

  encoder.add_completed_handler([plan = std::move(plan)]() {});
  encoder.add_temporary(workspace);
  return true;
}

bool execute_plans(
    cu::CommandEncoder& encoder,
    cudnn_frontend::EngineConfigList& configs,
    const std::string& op_graph_tag,
    const array& in,
    const array& wt,
    array& out) {
  for (auto& config : configs) {
    try {
      if (execute_plan(encoder, config, op_graph_tag, in, wt, out)) {
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

  // TODO: Searching a working execution plan is expensive, add cache.

  // Build operation graph.
  auto compute_data_type = (in.dtype() == float16 || in.dtype() == bfloat16)
      ? CUDNN_DATA_FLOAT
      : dtype_to_cudnn_type(in.dtype());

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

  auto backend_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
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
  if (execute_plans(encoder, configs, op_graph.getTag(), in, wt, out)) {
    return;
  }
  // Then try fallback plans.
  configs = get_engine_configs(backend_type, in.dtype(), op_graph);
  if (execute_plans(encoder, configs, op_graph.getTag(), in, wt, out)) {
    return;
  }
  throw std::runtime_error("Unable to find an engine for convolution.");
}

} // namespace mlx::core
