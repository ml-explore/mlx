// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <iostream>

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

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
  auto x = ensure_row_contiguous(x_pre);
  auto w = ensure_row_contiguous(w_pre);
  auto scales = ensure_row_contiguous(scales_pre);
  auto biases = ensure_row_contiguous(biases_pre);

  int D = x.shape(-1);
  int B = x.size() / D;
  int O = out.shape(-1);
  if (transpose_) {
    // Route to the fast qmv kernel that has no bounds checking
    if (B < 6 && O % 8 == 0 && D % 512 == 0 && D >= 512) {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "qmv_fast_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "qmv_fast", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 8;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size(O / bo, B, 1);

      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder->setBytes(&D, sizeof(int), 5);
      compute_encoder->setBytes(&O, sizeof(int), 6);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Route to the qmv kernel
    else if (B < 6) {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "qmv_" << type_string << "_gs_" << group_size_ << "_b_" << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "qmv", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 8;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size((O + bo - 1) / bo, B, 1);

      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder->setBytes(&D, sizeof(int), 5);
      compute_encoder->setBytes(&O, sizeof(int), 6);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Route to the qmm_t kernel
    else {
      std::ostringstream kname;
      std::string aligned_n = (O % 32) == 0 ? "true" : "false";
      auto type_string = get_type_string(x.dtype());
      kname << "qmm_t_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_ << "_alN_" << aligned_n;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "qmm_t", type_string, group_size_, bits_, aligned_n);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      int bk = 32;
      MTL::Size group_dims = MTL::Size(32, wn, wm);
      MTL::Size grid_dims = MTL::Size((O + bn - 1) / bn, (B + bm - 1) / bm, 1);

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder->setBytes(&B, sizeof(int), 5);
      compute_encoder->setBytes(&O, sizeof(int), 6);
      compute_encoder->setBytes(&D, sizeof(int), 7);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }
  } else {
    // Route to the qvm kernel
    if (B < 4) {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "qvm_" << type_string << "_gs_" << group_size_ << "_b_" << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "qvm", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 64;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size(O / bo, B, 1);

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder->setBytes(&D, sizeof(int), 5);
      compute_encoder->setBytes(&O, sizeof(int), 6);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Route to the qmm_n kernel
    else {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "qmm_n_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "qmm_n", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      int bk = 32;
      MTL::Size group_dims = MTL::Size(32, wn, wm);
      MTL::Size grid_dims = MTL::Size(O / bn, (B + bm - 1) / bm, 1);

      if ((O % bn) != 0) {
        std::ostringstream msg;
        msg << "[quantized_matmul] The output size should be divisible by "
            << bn << " but received " << O << ".";
        throw std::runtime_error(msg.str());
      }

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder->setBytes(&B, sizeof(int), 5);
      compute_encoder->setBytes(&O, sizeof(int), 6);
      compute_encoder->setBytes(&D, sizeof(int), 7);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 6);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];
  auto& lhs_indices = inputs[4];
  auto& rhs_indices = inputs[5];

  // TODO: collapse batch dims
  auto& batch_shape = lhs_indices.shape();
  int batch_ndims = batch_shape.size();
  auto& lhs_strides = lhs_indices.strides();
  auto& rhs_strides = rhs_indices.strides();

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

  int D = x.shape(-1);
  int B = x.shape(-2);
  int O = out.shape(-1);
  int N = out.size() / B / O;
  if (transpose_) {
    // Route to the fast bs_qmv kernel that has no bounds checking
    if (B < 6 && O % 8 == 0 && D % 512 == 0 && D >= 512) {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "bs_qmv_fast_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "bs_qmv_fast", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 8;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size(O / bo, B, N);

      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_input_array(lhs_indices, 4);
      compute_encoder.set_input_array(rhs_indices, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder->setBytes(&D, sizeof(int), 7);
      compute_encoder->setBytes(&O, sizeof(int), 8);

      compute_encoder->setBytes(&batch_ndims, sizeof(int), 9);
      set_vector_bytes(compute_encoder, batch_shape, 10);
      set_vector_bytes(compute_encoder, lhs_strides, 11);
      set_vector_bytes(compute_encoder, rhs_strides, 12);

      compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 13);
      set_vector_bytes(compute_encoder, x_shape, 14);
      set_vector_bytes(compute_encoder, x_strides, 15);
      compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 16);
      set_vector_bytes(compute_encoder, w_shape, 17);
      set_vector_bytes(compute_encoder, w_strides, 18);
      set_vector_bytes(compute_encoder, s_strides, 19);
      set_vector_bytes(compute_encoder, b_strides, 20);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    else if (B < 6) {
      std::ostringstream kname;
      auto type_string = get_type_string(x.dtype());
      kname << "bs_qmv_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "bs_qmv", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 8;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size((O + bo - 1) / bo, B, N);

      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_input_array(lhs_indices, 4);
      compute_encoder.set_input_array(rhs_indices, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder->setBytes(&D, sizeof(int), 7);
      compute_encoder->setBytes(&O, sizeof(int), 8);

      compute_encoder->setBytes(&batch_ndims, sizeof(int), 9);
      set_vector_bytes(compute_encoder, batch_shape, 10);
      set_vector_bytes(compute_encoder, lhs_strides, 11);
      set_vector_bytes(compute_encoder, rhs_strides, 12);

      compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 13);
      set_vector_bytes(compute_encoder, x_shape, 14);
      set_vector_bytes(compute_encoder, x_strides, 15);
      compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 16);
      set_vector_bytes(compute_encoder, w_shape, 17);
      set_vector_bytes(compute_encoder, w_strides, 18);
      set_vector_bytes(compute_encoder, s_strides, 19);
      set_vector_bytes(compute_encoder, b_strides, 20);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Route to the bs_qmm_t
    else {
      std::ostringstream kname;
      std::string aligned_n = (O % 32) == 0 ? "true" : "false";
      auto type_string = get_type_string(out.dtype());
      kname << "bs_qmm_t_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_ << "_alN_" << aligned_n;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "bs_qmm_t", type_string, group_size_, bits_, aligned_n);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      int bk = 32;
      MTL::Size group_dims = MTL::Size(32, wn, wm);
      MTL::Size grid_dims = MTL::Size((O + bn - 1) / bn, (B + bm - 1) / bm, N);

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_input_array(lhs_indices, 4);
      compute_encoder.set_input_array(rhs_indices, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder->setBytes(&B, sizeof(int), 7);
      compute_encoder->setBytes(&O, sizeof(int), 8);
      compute_encoder->setBytes(&D, sizeof(int), 9);

      compute_encoder->setBytes(&batch_ndims, sizeof(int), 10);
      set_vector_bytes(compute_encoder, batch_shape, 11);
      set_vector_bytes(compute_encoder, lhs_strides, 12);
      set_vector_bytes(compute_encoder, rhs_strides, 13);

      compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 14);
      set_vector_bytes(compute_encoder, x_shape, 15);
      set_vector_bytes(compute_encoder, x_strides, 16);
      compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 17);
      set_vector_bytes(compute_encoder, w_shape, 18);
      set_vector_bytes(compute_encoder, w_strides, 19);
      set_vector_bytes(compute_encoder, s_strides, 20);
      set_vector_bytes(compute_encoder, b_strides, 21);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }
  } else {
    // Route to the bs_qvm kernel
    if (B < 4) {
      std::ostringstream kname;
      auto type_string = get_type_string(out.dtype());
      kname << "bs_qvm_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "bs_qvm", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int bo = 64;
      int bd = 32;
      MTL::Size group_dims = MTL::Size(bd, 2, 1);
      MTL::Size grid_dims = MTL::Size(O / bo, B, N);

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_input_array(lhs_indices, 4);
      compute_encoder.set_input_array(rhs_indices, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder->setBytes(&D, sizeof(int), 7);
      compute_encoder->setBytes(&O, sizeof(int), 8);

      compute_encoder->setBytes(&batch_ndims, sizeof(int), 9);
      set_vector_bytes(compute_encoder, batch_shape, 10);
      set_vector_bytes(compute_encoder, lhs_strides, 11);
      set_vector_bytes(compute_encoder, rhs_strides, 12);

      compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 13);
      set_vector_bytes(compute_encoder, x_shape, 14);
      set_vector_bytes(compute_encoder, x_strides, 15);
      compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 16);
      set_vector_bytes(compute_encoder, w_shape, 17);
      set_vector_bytes(compute_encoder, w_strides, 18);
      set_vector_bytes(compute_encoder, s_strides, 19);
      set_vector_bytes(compute_encoder, b_strides, 20);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Route to bs_qmm_n
    else {
      std::ostringstream kname;
      auto type_string = get_type_string(out.dtype());
      kname << "bs_qmm_n_" << type_string << "_gs_" << group_size_ << "_b_"
            << bits_;

      // Encode and dispatch kernel
      auto& compute_encoder = d.get_command_encoder(s.index);
      auto template_def = get_template_definition(
          kname.str(), "bs_qmm_n", type_string, group_size_, bits_);
      auto kernel = get_quantized_kernel(d, kname.str(), template_def);
      compute_encoder->setComputePipelineState(kernel);

      int wn = 2;
      int wm = 2;
      int bm = 32;
      int bn = 32;
      int bk = 32;
      MTL::Size group_dims = MTL::Size(32, wn, wm);
      MTL::Size grid_dims = MTL::Size(O / bn, (B + bm - 1) / bm, N);

      if ((O % bn) != 0) {
        std::ostringstream msg;
        msg << "[quantized_matmul] The output size should be divisible by "
            << bn << " but received " << O << ".";
        throw std::runtime_error(msg.str());
      }

      compute_encoder.set_input_array(x, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(scales, 2);
      compute_encoder.set_input_array(biases, 3);
      compute_encoder.set_input_array(lhs_indices, 4);
      compute_encoder.set_input_array(rhs_indices, 5);
      compute_encoder.set_output_array(out, 6);
      compute_encoder->setBytes(&B, sizeof(int), 7);
      compute_encoder->setBytes(&O, sizeof(int), 8);
      compute_encoder->setBytes(&D, sizeof(int), 9);

      compute_encoder->setBytes(&batch_ndims, sizeof(int), 10);
      set_vector_bytes(compute_encoder, batch_shape, 11);
      set_vector_bytes(compute_encoder, lhs_strides, 12);
      set_vector_bytes(compute_encoder, rhs_strides, 13);

      compute_encoder->setBytes(&x_batch_ndims, sizeof(int), 14);
      set_vector_bytes(compute_encoder, x_shape, 15);
      set_vector_bytes(compute_encoder, x_strides, 16);
      compute_encoder->setBytes(&w_batch_ndims, sizeof(int), 17);
      set_vector_bytes(compute_encoder, w_shape, 18);
      set_vector_bytes(compute_encoder, w_strides, 19);
      set_vector_bytes(compute_encoder, s_strides, 20);
      set_vector_bytes(compute_encoder, b_strides, 21);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }
  }
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
  auto kernel_func = dequantize_ ? "affine_dequantize" : "affine_quantize";
  kernel_func =
      compute_scale_bias ? "affine_quantize" : "affine_quantize_with_params";
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
  int per_thread = compute_scale_bias
      ? std::max(group_size_ / simd_size, packs_per_int)
      : packs_per_int;
  size_t nthreads =
      dequantize_ ? w.size() * uint8_per_uint32 : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  auto grid_dims = MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatchThreads(grid_dims, group_dims);

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
