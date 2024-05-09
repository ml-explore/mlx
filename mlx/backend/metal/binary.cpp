// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <cassert>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr std::string_view binary_kernels = R"(
[[kernel]] void {0}_ss(
      device const {1}* a,
      device const {1}* b,
      device {2}* c,
      uint index [[thread_position_in_grid]]) {{
  c[index] = {3}()(a[0], b[0]);
}}

[[kernel]] void {0}_sv(
      device const {1}* a,
      device const {1}* b,
      device {2}* c,
      uint index [[thread_position_in_grid]]) {{
  c[index] = {3}()(a[0], b[index]);
}}

[[kernel]] void {0}_vs(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     uint index [[thread_position_in_grid]]) {{
  c[index] = {3}()(a[index], b[0]);
}}

[[kernel]] void {0}_vv(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     uint index [[thread_position_in_grid]]) {{
  c[index] = {3}()(a[index], b[index]);
}}

[[kernel]] void {0}_g_1(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     constant const size_t& a_stride,
     constant const size_t& b_stride,
     uint index [[thread_position_in_grid]]) {{
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  c[index] = {3}()(a[a_idx], b[b_idx]);
}}

[[kernel]] void {0}_g_2(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     constant const size_t a_strides[2],
     constant const size_t b_strides[2],
     uint2 index [[thread_position_in_grid]],
     uint2 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  c[out_idx] = {3}()(a[a_idx], b[b_idx]);
}}

[[kernel]] void {0}_g_3(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     constant const size_t a_strides[3],
     constant const size_t b_strides[3],
     uint3 index [[thread_position_in_grid]],
     uint3 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = {3}()(a[a_idx], b[b_idx]);
}}

[[kernel]] void {0}_g_4(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int shape[4],
    constant const size_t a_strides[4],
    constant const size_t b_strides[4],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd<4>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = {3}()(a[idx.x], b[idx.y]);

}}

[[kernel]] void {0}_g_5(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int shape[5],
    constant const size_t a_strides[5],
    constant const size_t b_strides[5],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd<5>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = {3}()(a[idx.x], b[idx.y]);
}}

[[kernel]] void {0}_g(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  c[out_idx] = {3}()(a[idx.x], b[idx.y]);
}}
)";

constexpr std::string_view binary_two_kernels = R"(
[[kernel]] void {0}_ss(
      device const {1}* a,
      device const {1}* b,
      device {2}* c,
      device {2}* d,
      uint index [[thread_position_in_grid]]) {{
  auto out = {3}()(a[0], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}}

[[kernel]] void {0}_sv(
      device const {1}* a,
      device const {1}* b,
      device {2}* c,
      device {2}* d,
      uint index [[thread_position_in_grid]]) {{
  auto out = {3}()(a[0], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}}

[[kernel]] void {0}_vs(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     device {2}* d,
     uint index [[thread_position_in_grid]]) {{
  auto out = {3}()(a[index], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}}

[[kernel]] void {0}_vv(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     device {2}* d,
     uint index [[thread_position_in_grid]]) {{
  auto out = {3}()(a[index], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}}

[[kernel]] void {0}_g_1(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     device {2}* d,
     constant const size_t& a_stride,
     constant const size_t& b_stride,
     uint index [[thread_position_in_grid]]) {{
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  auto out = {3}()(a[a_idx], b[b_idx]);
  c[index] = out[0];
  d[index] = out[1];
}}

[[kernel]] void {0}_g_2(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
     device {2}* d,
     constant const size_t a_strides[2],
     constant const size_t b_strides[2],
     uint2 index [[thread_position_in_grid]],
     uint2 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  auto out = {3}()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}}

[[kernel]] void {0}_g_3(
     device const {1}* a,
     device const {1}* b,
     device {2}* c,
    device {2}* d,
     constant const size_t a_strides[3],
     constant const size_t b_strides[3],
     uint3 index [[thread_position_in_grid]],
     uint3 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = {3}()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}}

[[kernel]] void {0}_g_4(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    device {2}* d,
    constant const int shape[4],
    constant const size_t a_strides[4],
    constant const size_t b_strides[4],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd<4>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = {3}()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}}

[[kernel]] void {0}_g_5(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    device {2}* d,
    constant const int shape[5],
    constant const size_t a_strides[5],
    constant const size_t b_strides[5],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd<5>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = {3}()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}}

[[kernel]] void {0}_g(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    device {2}* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  auto out = {3}()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}}
)";

void binary_op(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::string op) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt, true);
  set_binary_op_output_data(a, b, outputs[1], bopt, true);

  auto& out = outputs[0];
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_out = strides[2];

  std::string lib_name;
  std::string kernel_name;
  {
    std::ostringstream kname;
    kname << op << type_to_name(a);
    lib_name = kname.str();
    switch (bopt) {
      case BinaryOpType::ScalarScalar:
        kname << "_ss";
        break;
      case BinaryOpType::ScalarVector:
        kname << "_sv";
        break;
      case BinaryOpType::VectorScalar:
        kname << "_vs";
        break;
      case BinaryOpType::VectorVector:
        kname << "_vv";
        break;
      case BinaryOpType::General:
        kname << "_g";
        break;
    }
    if (bopt == BinaryOpType::General &&
        shape.size() <= MAX_BINARY_SPECIALIZED_DIMS) {
      kname << "_" << shape.size();
    }
    kernel_name = kname.str();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream op_t;
    out.primitive().print(op_t);
    std::ostringstream kernel_source;
    kernel_source << metal::get_kernel_preamble() << std::endl;
    kernel_source << fmt::format(
        binary_two_kernels,
        op + type_to_name(a),
        get_type_string(a.dtype()),
        get_type_string(out.dtype()),
        op_t.str());
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto kernel = d.get_kernel(kernel_name, lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  // - If a is donated it goes to the first output
  // - If b is donated it goes to the first output if a was not donated
  //   otherwise it goes to the second output
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? outputs[0] : a, 0);
  compute_encoder.set_input_array(
      donate_b ? (donate_a ? outputs[1] : outputs[0]) : b, 1);
  compute_encoder.set_output_array(outputs[0], 2);
  compute_encoder.set_output_array(outputs[1], 3);

  if (bopt == BinaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 4);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 6);
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 7);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D grid of threads
    size_t nthreads = out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

void binary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt, true);
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_out = strides[2];

  std::string lib_name;
  std::string kernel_name;
  {
    std::ostringstream kname;
    kname << op << type_to_name(a);
    lib_name = kname.str();
    switch (bopt) {
      case BinaryOpType::ScalarScalar:
        kname << "_ss";
        break;
      case BinaryOpType::ScalarVector:
        kname << "_sv";
        break;
      case BinaryOpType::VectorScalar:
        kname << "_vs";
        break;
      case BinaryOpType::VectorVector:
        kname << "_vv";
        break;
      case BinaryOpType::General:
        kname << "_g";
        break;
    }
    if (bopt == BinaryOpType::General &&
        shape.size() <= MAX_BINARY_SPECIALIZED_DIMS) {
      kname << "_" << shape.size();
    }
    kernel_name = kname.str();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream op_t;
    out.primitive().print(op_t);
    std::ostringstream kernel_source;
    kernel_source << metal::get_kernel_preamble() << std::endl;
    kernel_source << fmt::format(
        binary_kernels,
        op + type_to_name(a),
        get_type_string(a.dtype()),
        get_type_string(out.dtype()),
        op_t.str());
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto kernel = d.get_kernel(kernel_name, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? out : a, 0);
  compute_encoder.set_input_array(donate_b ? out : b, 1);
  compute_encoder.set_output_array(out, 2);

  if (bopt == BinaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 3);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 3);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 4);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 6);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D grid of threads
    size_t nthreads =
        bopt == BinaryOpType::General ? out.size() : out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "add");
}

void ArcTan2::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "arctan2");
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (op_) {
    case BitwiseBinary::And:
      binary_op(inputs, out, "bitwise_and");
      break;
    case BitwiseBinary::Or:
      binary_op(inputs, out, "bitwise_or");
      break;
    case BitwiseBinary::Xor:
      binary_op(inputs, out, "bitwise_xor");
      break;
    case BitwiseBinary::LeftShift:
      binary_op(inputs, out, "left_shift");
      break;
    case BitwiseBinary::RightShift:
      binary_op(inputs, out, "right_shift");
      break;
  }
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "div");
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  binary_op(inputs, outputs, "divmod");
}

void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "rem");
}

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, equal_nan_ ? "naneq" : "eq");
}

void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "ge");
}

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "geq");
}

void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "le");
}

void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "leq");
}

void LogicalAnd::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "land");
}

void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "lor");
}

void LogAddExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "lae");
}

void Maximum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "max");
}

void Minimum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "min");
}

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "mul");
}

void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "neq");
}

void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "pow");
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "sub");
}

} // namespace mlx::core
