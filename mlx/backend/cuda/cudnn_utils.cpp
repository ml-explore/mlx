// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

namespace {

#define RETURN_IF_ERROR(cmd)          \
  if (auto ret = cmd; ret.is_bad()) { \
    return ret;                       \
  }

// In MLX a singleton dim (shape[dim] == 1) can have any stride, but in cuDNN
// whether a tensor is contiguous is determined with:
// shape[dim] == shape[dim + 1] * strides[dim + 1]
// So a contiguous array with singleton dims in MLX may be mistakenly treated
// as strided in cuDNN, and we work around it by normalizing the strides.
std::vector<int64_t> normalized_strides(const array& x) {
  std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
  if (std::all_of(
          strides.begin(), strides.end(), [](int64_t s) { return s == 0; })) {
    strides.back() = 1;
    return strides;
  }
  if (!x.flags().row_contiguous || x.ndim() < 2) {
    return strides;
  }
  for (int i = x.ndim() - 2; i >= 0; --i) {
    if (x.shape(i) == 1) {
      strides[i] = x.shape(i + 1) * strides[i + 1];
    }
  }
  return strides;
}

// Return the shape and strides after transposing from NHWC to NCHW.
inline auto nhwc_to_nchw(const array& x) {
  auto shape = convert_vector<int64_t>(x.shape());
  auto strides = normalized_strides(x);
  assert(shape.size() >= 3);
  shape.insert(shape.begin() + 1, shape.back());
  shape.erase(shape.end() - 1);
  strides.insert(strides.begin() + 1, strides.back());
  strides.erase(strides.end() - 1);
  return std::make_tuple(std::move(shape), std::move(strides));
}

} // namespace

fe::error_t DnnGraph::prepare() {
  RETURN_IF_ERROR(validate());
  try {
    RETURN_IF_ERROR(build_operation_graph(handle_));
  } catch (cudnn_frontend::cudnnException& error) {
    // cuDNN bug: they did not catch all exceptions in the API.
    return {fe::error_code_t::CUDNN_BACKEND_API_FAILED, error.what()};
  }
  RETURN_IF_ERROR(create_execution_plans({fe::HeurMode_t::A}));
  return {};
}

fe::error_t DnnGraph::build() {
  RETURN_IF_ERROR(check_support(handle_));
  RETURN_IF_ERROR(build_plans(handle_));
  return {};
}

fe::error_t DnnGraph::encode_graph(
    cu::CommandEncoder& encoder,
    std::unordered_map<int64_t, void*> variant_pack) {
  cudnnSetStream(handle_, encoder.stream());
  CudaGraph cuda_graph(encoder.device());
  RETURN_IF_ERROR(populate_cuda_graph(
      handle_, variant_pack, prepare_workspace(encoder), cuda_graph));
  encoder.add_graph_node(cuda_graph);
  return {};
}

fe::error_t DnnGraph::encode_capturing(
    cu::CommandEncoder& encoder,
    std::unordered_map<int64_t, void*> variant_pack) {
  auto* workspace_ptr = prepare_workspace(encoder);
  auto capture = encoder.capture_context();
  cudnnSetStream(handle_, encoder.stream());
  auto ret = execute(handle_, variant_pack, workspace_ptr);
  if (ret.is_bad()) {
    capture.discard = true;
  }
  return ret;
}

void* DnnGraph::prepare_workspace(cu::CommandEncoder& encoder) {
  int64_t workspace_size = 0;
  CHECK_CUDNN_FE_ERROR(get_workspace_size(workspace_size));
  if (workspace_size > 0) {
    array workspace(
        cu::malloc_async(workspace_size, encoder),
        {static_cast<int>(workspace_size)},
        uint8);
    encoder.add_temporary(workspace);
    return gpu_ptr<void>(workspace);
  }
  return nullptr;
}

void DnnGraph::set_tensor_attrs(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides) {
  tensor->set_uid(uid)
      .set_alignment(get_alignment(x))
      .set_data_type(dtype_to_cudnn_type(x.dtype()))
      .set_dim(shape)
      .set_stride(strides);
}

void DnnGraph::set_tensor_attrs(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x) {
  set_tensor_attrs(
      tensor,
      uid,
      x,
      convert_vector<int64_t>(x.shape()),
      normalized_strides(x));
}

void DnnGraph::set_tensor_attrs_nchw(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x) {
  auto [shape, strides] = nhwc_to_nchw(x);
  set_tensor_attrs(tensor, uid, x, shape, strides);
}

} // namespace mlx::core
