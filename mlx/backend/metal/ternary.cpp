// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <cassert>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/ternary.h"
#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr std::string_view ternary_kernels = R"(
[[kernel]] void {0}_v(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    uint index [[thread_position_in_grid]]) {{
  d[index] = {2}()(a[index], b[index], c[index]);
}}

[[kernel]] void {0}_g_1(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t& a_strides,
    constant const size_t& b_strides,
    constant const size_t& c_strides,
    uint index [[thread_position_in_grid]]) {{
  auto a_idx = elem_to_loc_1(index, a_strides);
  auto b_idx = elem_to_loc_1(index, b_strides);
  auto c_idx = elem_to_loc_1(index, c_strides);
  d[index] = {2}()(a[a_idx], b[b_idx], c[c_idx]);
}}

[[kernel]] void {0}_g_2(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    constant const size_t c_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  auto c_idx = elem_to_loc_2(index, c_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  d[out_idx] = {2}()(a[a_idx], b[b_idx], c[c_idx]);
}}

[[kernel]] void {0}_g_3(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    constant const size_t c_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  auto c_idx = elem_to_loc_3(index, c_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = {2}()(a[a_idx], b[b_idx], c[c_idx]);
}}

[[kernel]] void {0}_g_4(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int shape[4],
    constant const size_t a_strides[4],
    constant const size_t b_strides[4],
    constant const size_t c_strides[4],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx =
      elem_to_loc_3_nd<4>(index, shape, a_strides, b_strides, c_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = {2}()(a[idx.x], b[idx.y], c[idx.z]);
}}

[[kernel]] void {0}_g_5(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int shape[5],
    constant const size_t a_strides[5],
    constant const size_t b_strides[5],
    constant const size_t c_strides[5],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx =
      elem_to_loc_3_nd<5>(index, shape, a_strides, b_strides, c_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = {2}()(a[idx.x], b[idx.y], c[idx.z]);
}}

[[kernel]] void {0}_g(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  auto idx =
      elem_to_loc_3_nd(index, shape, a_strides, b_strides, c_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  d[out_idx] = {2}()(a[idx.x], b[idx.y], c[idx.z]);
}}
)";

void ternary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  assert(inputs.size() == 3);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt, true /* donate_with_move */);

  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, c, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_c = strides[2];
  auto& strides_out = strides[3];

  std::string lib_name;
  std::string kernel_name;
  {
    std::ostringstream kname;
    kname << op << type_to_name(b);
    lib_name = kname.str();
    if (topt == TernaryOpType::General) {
      kname << "_g";
      if (shape.size() <= MAX_BINARY_SPECIALIZED_DIMS) {
        kname << "_" << shape.size();
      }
    } else {
      kname << "_v";
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
        ternary_kernels,
        op + type_to_name(b),
        get_type_string(b.dtype()),
        op_t.str());
    lib = d.get_library(lib_name, kernel_source.str());
  }
  auto kernel = d.get_kernel(kernel_name, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(c, 2);
  compute_encoder.set_output_array(out, 3);

  if (topt == TernaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 4);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 6);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 7);

      if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
        compute_encoder->setBytes(&ndim, sizeof(int), 8);
      }
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 6);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    MTL::Size group_dims = get_block_dims(dim0, dim1, rest);
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

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  ternary_op(inputs, out, "select");
}

} // namespace mlx::core
