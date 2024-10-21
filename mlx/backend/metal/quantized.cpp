// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
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
  d.add_temporaries(std::move(copies), s.index);
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
    if (B < 4) {
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
