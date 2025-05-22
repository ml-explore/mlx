// Copyright © 2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(
        x.data_shared_ptr() == nullptr ? out : x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_output_array(out, 2);
    compute_encoder.set_bytes(eps_, 3);
    compute_encoder.set_bytes(axis_size, 4);
    compute_encoder.set_bytes(w_stride, 5);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&d, &s](const array& x) -> std::pair<array, bool> {
    if (x.flags().row_contiguous) {
      return {x, false};
    }

    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    return {x_copy, true};
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[2].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  auto [g, g_copied] = check_input(inputs[2]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.copy_shared_buffer(x);
  } else if (g.is_donatable()) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    d.add_temporary(g, s.index);
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate the gradient accumulator gw and a temporary to store the
  // gradients before they are accumulated.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      d.add_temporary(gw_temp, s.index);
    }
  }
  gw.set_data(allocator::malloc(gw.nbytes()));

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);

  std::string hash_name = op_name + ((has_w) ? "_w" : "_now");
  metal::MTLFCList func_consts = {
      {&has_w, MTL::DataType::DataTypeBool, 20},
  };

  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name, "mlx", hash_name, func_consts);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g, 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }
}

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    uint32_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(
        x.data_shared_ptr() == nullptr ? out : x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(b, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(eps_, 4);
    compute_encoder.set_bytes(axis_size, 5);
    compute_encoder.set_bytes(w_stride, 6);
    compute_encoder.set_bytes(b_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x) -> std::pair<array, bool> {
    if (x.flags().row_contiguous) {
      return {x, false};
    }
    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    return {x_copy, true};
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[3].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  const array& b = inputs[2];
  auto [g, g_copied] = check_input(inputs[3]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    d.add_temporary(g, s.index);
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate a temporary to store the gradients for w and allocate the output
  // gradient accumulators.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      d.add_temporary(gw_temp, s.index);
    }
  }
  gw.set_data(allocator::malloc(gw.nbytes()));
  gb.set_data(allocator::malloc(gb.nbytes()));

  // Finish with the gradient for b in case we had a b
  auto& compute_encoder = d.get_command_encoder(s.index);
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        g, gb, "sum", plan, {0}, compute_encoder, d, s);
  }

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);

  std::string hash_name = op_name + ((has_w) ? "_w" : "_now");
  metal::MTLFCList func_consts = {
      {&has_w, MTL::DataType::DataTypeBool, 20},
  };

  {
    auto kernel = d.get_kernel(op_name, "mlx", hash_name, func_consts);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g, 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }
}

} // namespace mlx::core::fast
