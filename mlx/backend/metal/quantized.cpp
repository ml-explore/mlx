// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

void launch_qmm(
    std::string name,
    const std::vector<array>& inputs,
    array& out,
    int group_size,
    int bits,
    int D,
    int O,
    int B,
    int N,
    MTL::Size& group_dims,
    MTL::Size& grid_dims,
    bool batched,
    bool matrix,
    bool gather,
    bool aligned,
    bool quad,
    const Stream& s) {
  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

  // Ensure that the last two dims are row contiguous.
  // TODO: Check if we really need this for x as well...
  std::vector<array> copies;
  auto ensure_row_contiguous_last_dims = [&copies, &s](const array& arr) {
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      return arr_copy;
    }
  };
  auto x = ensure_row_contiguous_last_dims(x_pre);
  auto w = ensure_row_contiguous_last_dims(w_pre);
  auto scales = ensure_row_contiguous_last_dims(scales_pre);
  auto biases = ensure_row_contiguous_last_dims(biases_pre);

  int x_batch_ndims = x.ndim() - 2;
  auto& x_shape = x.shape();
  auto& x_strides = x.strides();
  int w_batch_ndims = w.ndim() - 2;
  auto& w_shape = w.shape();
  auto& w_strides = w.strides();
  auto& s_strides = scales.strides();
  auto& b_strides = biases.strides();

  std::string aligned_n = (O % 32) == 0 ? "true" : "false";

  std::ostringstream kname;
  auto type_string = get_type_string(x.dtype());
  kname << name << "_" << type_string << "_gs_" << group_size << "_b_" << bits;
  if (quad) {
    kname << "_d_" << D;
  }
  if (aligned) {
    kname << "_alN_" << aligned_n;
  }
  if (!gather) {
    kname << "_batch_" << batched;
  }

  // Encode and dispatch kernel
  std::string template_def;
  if (quad) {
    template_def = get_template_definition(
        kname.str(), name, type_string, group_size, bits, D, batched);
  } else if (aligned && !gather) {
    template_def = get_template_definition(
        kname.str(), name, type_string, group_size, bits, aligned_n, batched);
  } else if (!gather && !aligned) {
    template_def = get_template_definition(
        kname.str(), name, type_string, group_size, bits, batched);
  } else if (aligned && gather) {
    template_def = get_template_definition(
        kname.str(), name, type_string, group_size, bits, aligned_n);
  } else {
    template_def = get_template_definition(
        kname.str(), name, type_string, group_size, bits);
  }
  auto& d = metal::device(s.device);
  auto kernel = get_quantized_kernel(d, kname.str(), template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder->setBytes(&D, sizeof(int), 5);
  compute_encoder->setBytes(&O, sizeof(int), 6);

  int offset = 7;
  if (matrix) {
    compute_encoder->setBytes(&B, sizeof(int), 7);
    offset += 1;
  }

  if (batched || gather) {
    compute_encoder->setBytes(&x_batch_ndims, sizeof(int), offset);
    set_vector_bytes(compute_encoder, x_shape, offset + 1);
    set_vector_bytes(compute_encoder, x_strides, offset + 2);
    compute_encoder->setBytes(&w_batch_ndims, sizeof(int), offset + 3);
    set_vector_bytes(compute_encoder, w_shape, offset + 4);
    set_vector_bytes(compute_encoder, w_strides, offset + 5);
    set_vector_bytes(compute_encoder, s_strides, offset + 6);
    set_vector_bytes(compute_encoder, b_strides, offset + 7);
  }
  if (gather) {
    auto& lhs_indices = inputs[4];
    auto& rhs_indices = inputs[5];

    // TODO: collapse batch dims
    auto& batch_shape = lhs_indices.shape();
    int batch_ndims = batch_shape.size();
    auto& lhs_strides = lhs_indices.strides();
    auto& rhs_strides = rhs_indices.strides();

    compute_encoder->setBytes(&batch_ndims, sizeof(int), offset + 8);
    set_vector_bytes(compute_encoder, batch_shape, offset + 9);
    compute_encoder.set_input_array(lhs_indices, offset + 10);
    compute_encoder.set_input_array(rhs_indices, offset + 11);
    set_vector_bytes(compute_encoder, lhs_strides, offset + 12);
    set_vector_bytes(compute_encoder, rhs_strides, offset + 13);
  }

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
  d.add_temporaries(std::move(copies), s.index);
}

void qvm_split_k(
    const std::vector<array>& inputs,
    array& out,
    int group_size,
    int bits,
    int D,
    int O,
    int B,
    int N,
    const Stream& s) {
  int split_k = D > 8192 ? 32 : 8;
  int split_D = (D + split_k - 1) / split_k;
  N *= split_k;

  int bo = 64;
  int bd = 32;
  MTL::Size group_dims = MTL::Size(bd, 2, 1);
  MTL::Size grid_dims = MTL::Size(O / bo, B, N);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

  // Ensure that the last two dims are row contiguous.
  // TODO: Check if we really need this for x as well...
  std::vector<array> copies;
  auto ensure_row_contiguous_last_dims = [&copies, &s](const array& arr) {
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      return arr_copy;
    }
  };
  auto x = ensure_row_contiguous_last_dims(x_pre);
  auto w = ensure_row_contiguous_last_dims(w_pre);
  auto scales = ensure_row_contiguous_last_dims(scales_pre);
  auto biases = ensure_row_contiguous_last_dims(biases_pre);

  int x_batch_ndims = x.ndim() - 2;
  auto x_shape = x.shape();
  auto x_strides = x.strides();
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();
  auto b_strides = biases.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x.ndim() - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));
  b_strides.insert(b_strides.end() - 2, split_D * biases.shape(-1));

  int final_block_size = D - (split_k - 1) * split_D;

  auto& d = metal::device(s.device);

  auto temp_shape = out.shape();
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
  d.add_temporary(intermediate, s.index);

  std::ostringstream kname;
  auto type_string = get_type_string(x.dtype());
  kname << "qvm_split_k" << "_" << type_string << "_gs_" << group_size << "_b_"
        << bits << "_spk_" << split_k;
  auto template_def = get_template_definition(
      kname.str(), "qvm_split_k", type_string, group_size, bits, split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel(d, kname.str(), template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_input_array(biases, 2);
  compute_encoder.set_input_array(x, 3);
  compute_encoder.set_output_array(intermediate, 4);
  compute_encoder->setBytes(&split_D, sizeof(int), 5);
  compute_encoder->setBytes(&O, sizeof(int), 6);

  compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 7);
  set_vector_bytes(compute_encoder, x_shape, 8);
  set_vector_bytes(compute_encoder, x_strides, 9);
  compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 10);
  set_vector_bytes(compute_encoder, w_shape, 11);
  set_vector_bytes(compute_encoder, w_strides, 12);
  set_vector_bytes(compute_encoder, s_strides, 13);
  set_vector_bytes(compute_encoder, b_strides, 14);
  compute_encoder->setBytes(&final_block_size, sizeof(int), 15);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
  d.add_temporaries(std::move(copies), s.index);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qmm_op(
    const std::vector<array>& inputs,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    bool gather,
    const Stream& s) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  MTL::Size group_dims;
  MTL::Size grid_dims;

  auto& x = inputs[0];
  auto& w = inputs[1];
  bool batched = !gather && (w.ndim() > 2 || !x.flags().row_contiguous);

  int D = x.shape(-1);
  int O = out.shape(-1);
  // For the unbatched W case, avoid `adjust_matrix_offsets`
  // for a small performance gain.
  int B = (batched || gather) ? x.shape(-2) : x.size() / D;
  int N = (batched || gather) ? out.size() / B / O : 1;

  std::string name = gather ? "bs_" : "";
  bool matrix = false;
  bool aligned = false;
  bool quad = false;

  if (transpose) {
    if (B < 6 && (D == 128 || D == 64)) {
      name += "qmv_quad";
      constexpr int quads_per_simd = 8;
      constexpr int results_per_quadgroup = 8;
      int bo = quads_per_simd * results_per_quadgroup;
      int simdgroup_size = 32;
      group_dims = MTL::Size(simdgroup_size, 1, 1);
      grid_dims = MTL::Size((O + bo - 1) / bo, B, N);
      quad = true;
    } else if (B < 6 && O % 8 == 0 && D % 512 == 0 && D >= 512) {
      name += "qmv_fast";
      int bo = 8;
      int bd = 32;
      group_dims = MTL::Size(bd, 2, 1);
      grid_dims = MTL::Size(O / bo, B, N);
    } else if (B < 6) {
      name += "qmv";
      int bo = 8;
      int bd = 32;
      group_dims = MTL::Size(bd, 2, 1);
      grid_dims = MTL::Size((O + bo - 1) / bo, B, N);
    } else {
      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      group_dims = MTL::Size(32, wn, wm);
      grid_dims = MTL::Size((O + bn - 1) / bn, (B + bm - 1) / bm, N);
      name += "qmm_t";
      matrix = true;
      aligned = true;
    }
  } else {
    if (B < 4 && D >= 1024 && !gather) {
      return qvm_split_k(inputs, out, group_size, bits, D, O, B, N, s);
    } else if (B < 4) {
      name += "qvm";
      int bo = 64;
      int bd = 32;
      group_dims = MTL::Size(bd, 2, 1);
      grid_dims = MTL::Size(O / bo, B, N);
    } else {
      name += "qmm_n";
      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      group_dims = MTL::Size(32, wn, wm);
      grid_dims = MTL::Size(O / bn, (B + bm - 1) / bm, N);
      matrix = true;
      if ((O % bn) != 0) {
        std::ostringstream msg;
        msg << "[quantized_matmul] The output size should be divisible by "
            << bn << " but received " << O << ".";
        throw std::runtime_error(msg.str());
      }
    }
  }
  launch_qmm(
      name,
      inputs,
      out,
      group_size,
      bits,
      D,
      O,
      B,
      N,
      group_dims,
      grid_dims,
      batched,
      matrix,
      gather,
      aligned,
      quad,
      s);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);
  qmm_op(
      inputs, out, transpose_, group_size_, bits_, /*gather=*/false, stream());
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 6);
  qmm_op(
      inputs, out, transpose_, group_size_, bits_, /*gather=*/true, stream());
}

void fast::AffineQuantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  bool compute_scale_bias = inputs.size() == 1;

  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);

  std::vector<array> copies;
  auto ensure_row_contiguous = [&copies, &s](const array& arr) {
    if (arr.flags().row_contiguous) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      return arr_copy;
    }
  };
  auto w = ensure_row_contiguous(w_pre);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_input_array(w, 0);
  if (!compute_scale_bias) {
    auto& scales_pre = inputs[1];
    auto& biases_pre = inputs[2];
    auto scales = ensure_row_contiguous(scales_pre);
    auto biases = ensure_row_contiguous(biases_pre);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_input_array(biases, 2);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    auto& biases = outputs[2];
    scales.set_data(allocator::malloc_or_wait(scales.nbytes()));
    biases.set_data(allocator::malloc_or_wait(biases.nbytes()));
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
    compute_encoder.set_output_array(biases, 3);
  }

  std::ostringstream kname;
  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto kernel_func = "affine_quantize_scales_biases";
  if (dequantize_) {
    kernel_func = "affine_dequantize";
  } else if (compute_scale_bias) {
    kernel_func = "affine_quantize";
  }
  kname << kernel_func << "_" << type_string << "_gs_" << group_size_ << "_b_"
        << bits_;
  auto template_def = get_template_definition(
      kname.str(), kernel_func, type_string, group_size_, bits_);
  auto kernel = get_quantized_kernel(d, kname.str(), template_def);
  compute_encoder->setComputePipelineState(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = 8 / bits_;
  int per_thread = compute_scale_bias ? group_size_ / simd_size : packs_per_int;
  size_t nthreads =
      dequantize_ ? w.size() * uint8_per_uint32 : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatchThreads(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
