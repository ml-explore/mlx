// Copyright © 2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) -> const array& {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];

  if (x.is_donatable()) {
    out.move_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

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

    uint32_t w_stride = w.strides()[0];
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

  d.add_temporaries(std::move(copies), s.index);
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) -> const array& {
    if (x.flags().row_contiguous) {
      return x;
    }
    // Make sure we 'll only ever allocate once. The point of that goes beyond
    // the minor optimization. We need to ensure that there will be no
    // reallocation such that the references won't change when we
    // push_back(...). So tl;dr 3 possible copies x, g and gw_temp.
    copies.reserve(3);

    copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
    copy_gpu(x, copies.back(), CopyType::General, s);
    return copies.back();
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& g = check_input(inputs[2]);
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Allocate space for the outputs
  bool x_in_gx = false;
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.move_shared_buffer(x);
    x_in_gx = true;
  } else if (g.is_donatable()) {
    gx.move_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc_or_wait(gx.nbytes()));
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate the gradient accumulator gw and a temporary to store the
  // gradients before they are accumulated.
  array gw_temp({n_rows, x.shape().back()}, gw.dtype(), nullptr, {});
  bool g_in_gw = false;
  if (!g_in_gx && g.is_donatable()) {
    gw_temp.move_shared_buffer(g);
    g_in_gw = true;
  } else {
    gw_temp.set_data(allocator::malloc_or_wait(gw_temp.nbytes()));
  }
  copies.push_back(gw_temp);
  gw.set_data(allocator::malloc_or_wait(gw.nbytes()));

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);
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

    uint32_t w_stride = w.strides()[0];
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x_in_gx ? gx : x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g_in_gx ? gx : (g_in_gw ? gw_temp : g), 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
  strided_reduce_general_dispatch(
      gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);

  d.add_temporaries(std::move(copies), s.index);
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) -> const array& {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  if (x.is_donatable()) {
    out.move_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

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

  d.add_temporaries(std::move(copies), s.index);
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) -> const array& {
    if (x.flags().row_contiguous) {
      return x;
    }
    // Make sure we 'll only ever allocate once. The point of that goes beyond
    // the minor optimization. We need to ensure that there will be no
    // reallocation such that the references won't change when we
    // push_back(...). So tl;dr 3 possible copies x, g and gw_temp.
    copies.reserve(3);

    copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
    copy_gpu(x, copies.back(), CopyType::General, s);
    return copies.back();
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];
  const array& g = check_input(inputs[3]);
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Allocate space for the outputs
  bool x_in_gx = false;
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.move_shared_buffer(x);
    x_in_gx = true;
  } else if (g.is_donatable()) {
    gx.move_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc_or_wait(gx.nbytes()));
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate a temporary to store the gradients for w and allocate the output
  // gradient accumulators.
  array gw_temp({n_rows, x.shape().back()}, gw.dtype(), nullptr, {});
  bool g_in_gw = false;
  if (!g_in_gx && g.is_donatable()) {
    gw_temp.move_shared_buffer(g);
    g_in_gw = true;
  } else {
    gw_temp.set_data(allocator::malloc_or_wait(gw_temp.nbytes()));
  }
  copies.push_back(gw_temp);
  gw.set_data(allocator::malloc_or_wait(gw.nbytes()));
  gb.set_data(allocator::malloc_or_wait(gb.nbytes()));

  // Finish with the gradient for b in case we had a b
  auto& compute_encoder = d.get_command_encoder(s.index);
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        g_in_gx ? gx : (g_in_gw ? gw_temp : g),
        gb,
        "sum",
        plan,
        {0},
        compute_encoder,
        d,
        s);
  }

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);
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
    compute_encoder.set_input_array(x_in_gx ? gx : x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g_in_gx ? gx : (g_in_gw ? gw_temp : g), 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  if (gw.ndim() == 1 && gw.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
