// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

constexpr int n_per_thread = 4;

bool RoPE::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  int64_t strides[3];
  int64_t out_strides[3];
  bool donated = false;
  int ndim = in.ndim();
  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  size_t mat_size = T * D;
  bool large = in.data_size() > INT32_MAX || in.size() > INT32_MAX;

  int dispatch_ndim = ndim;
  while (in.shape(-dispatch_ndim) == 1 && dispatch_ndim > 3) {
    dispatch_ndim--;
  }

  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }

  bool head_seq_transpose = false;

  if (dims_ < D) {
    donated = true;
    auto ctype =
        (in.flags().row_contiguous) ? CopyType::Vector : CopyType::General;
    copy_gpu(in, out, ctype, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  } else if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      donated = true;
      out.copy_shared_buffer(in);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (dispatch_ndim == 3) {
    // Handle non-contiguous 3D inputs
    out.set_data(allocator::malloc(out.nbytes()));
    strides[0] = in.strides()[ndim - 3];
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (
      ndim == 4 &&
      // batch dim is regularly strided
      in.strides()[0] == T * N * D &&
      // sequence and head dimensions are transposed
      in.strides()[1] == D && in.strides()[2] == N * D) {
    head_seq_transpose = true;
    out.set_data(allocator::malloc(out.nbytes()));
    strides[0] = in.strides()[1];
    strides[1] = in.strides()[2];
    strides[2] = in.strides()[3];
  } else {
    // Copy non-contiguous > 3D inputs into the output and treat
    // input as donated
    donated = true;
    copy_gpu(in, out, CopyType::General, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  }
  out_strides[0] = mat_size;
  out_strides[1] = out.strides()[ndim - 2];
  out_strides[2] = out.strides()[ndim - 1];

  // Special case for inference (single time step, contiguous, one offset)
  auto& offset = inputs[1];
  bool single = in.flags().row_contiguous && T == 1 && offset.size() == 1;

  bool with_freqs = inputs.size() == 3;
  std::string kname;
  concatenate(
      kname,
      "rope_",
      single ? "single_" : "",
      (with_freqs) ? "freqs_" : "",
      large ? "large_" : "",
      type_to_name(in));
  std::string hash_name;
  concatenate(
      hash_name,
      kname,
      "_",
      forward_ ? "" : "vjp_",
      traditional_ ? "traditional_" : "",
      head_seq_transpose ? "transpose" : "");
  metal::MTLFCList func_consts = {
      {&forward_, MTL::DataType::DataTypeBool, 1},
      {&traditional_, MTL::DataType::DataTypeBool, 2},
      {&head_seq_transpose, MTL::DataType::DataTypeBool, 3}};

  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  auto& compute_encoder = d.get_command_encoder(s.index);

  float base = std::log2(base_);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(donated ? out : in, 0);
  compute_encoder.set_output_array(out, 1);

  compute_encoder.set_input_array(offset, 2);
  compute_encoder.set_bytes(scale_, 3);

  MTL::Size group_dims;
  MTL::Size grid_dims;
  if (single) {
    compute_encoder.set_bytes(out_strides, 1, 4);
    uint32_t dim0 = dims_ / 2;
    group_dims = get_block_dims(dim0, N, 1);
    grid_dims = MTL::Size(dim0, N, 1);
  } else {
    compute_encoder.set_bytes(strides, 3, 4);
    compute_encoder.set_bytes(out_strides, 3, 5);
    int64_t offset_stride = 0;
    if (offset.ndim() > 0) {
      offset_stride = offset.strides()[0];
    }
    compute_encoder.set_bytes(offset_stride, 6);
    compute_encoder.set_bytes(N, 7);
    uint32_t dim0 = dims_ / 2;
    uint32_t dim1 = T;
    uint32_t dim2 = B * ((N + n_per_thread - 1) / n_per_thread);
    group_dims = get_block_dims(dim0, dim1, dim2);
    grid_dims = MTL::Size(dim0, dim1, dim2);
  }

  if (with_freqs) {
    auto& freqs = inputs[2];
    compute_encoder.set_input_array(freqs, 10);
    auto freq_stride = freqs.strides()[0];
    compute_encoder.set_bytes(freq_stride, 11);
  } else {
    compute_encoder.set_bytes(base, 10);
  }
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
