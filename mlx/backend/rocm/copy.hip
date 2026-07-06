// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace mlx::core {

namespace {
// Env-gated (MLX_COUNT_COPIES) attribution of general copies by shape, to find
// which model ops generate the copy_gg/copy_g kernels. Prints a histogram at exit.
struct CopyStats {
  std::mutex mu;
  std::map<std::string, long> hist;
  bool on = std::getenv("MLX_COUNT_COPIES") != nullptr;
  void record(CopyType ct, const Shape& sh) {
    if (!on) return;
    std::string key = (ct == CopyType::GeneralGeneral) ? "GG " : "G  ";
    key += "[";
    for (size_t i = 0; i < sh.size(); ++i) {
      key += std::to_string(sh[i]);
      if (i + 1 < sh.size()) key += ",";
    }
    key += "]";
    std::lock_guard<std::mutex> g(mu);
    hist[key]++;
  }
  ~CopyStats() {
    if (!on) return;
    long tot = 0;
    for (auto& kv : hist) tot += kv.second;
    std::vector<std::pair<std::string, long>> v(hist.begin(), hist.end());
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    fprintf(stderr, "\n=== MLX_COUNT_COPIES: %ld general copies, top shapes ===\n",
            tot);
    for (size_t i = 0; i < v.size() && i < 40; ++i)
      fprintf(stderr, "%8ld  %s\n", v[i].second, v[i].first.c_str());
  }
};
CopyStats g_copy_stats;
} // namespace

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);
  bool donated = set_copy_output_data(
      in, out, ctype, [&](auto n) { return mlx::core::rocm::malloc_async(n, encoder); });
  if (donated && in.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t offset_in,
    int64_t offset_out,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_offset_in,
    std::optional<array> dynamic_offset_out) {
  if (out.size() == 0) {
    return;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Handle dynamic offsets
  if (dynamic_offset_in.has_value() || dynamic_offset_out.has_value()) {
    auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
        shape, std::vector{strides_in, strides_out}, INT32_MAX);
    
    // Create zero offset arrays for missing dynamic offsets
    // We need to allocate and initialize on GPU to avoid hipDeviceSynchronize
    if (!dynamic_offset_in) {
      dynamic_offset_in = array({1}, int64, nullptr, {});
      dynamic_offset_in->set_data(mlx::core::rocm::malloc_async(sizeof(int64_t), encoder));
      encoder.add_temporary(*dynamic_offset_in);
      // Initialize to zero on GPU using hipMemset
      int64_t* ptr = gpu_ptr<int64_t>(*dynamic_offset_in);
      encoder.launch_kernel([ptr](hipStream_t stream) {
        (void)hipMemsetAsync(ptr, 0, sizeof(int64_t), stream);
      });
    }
    if (!dynamic_offset_out) {
      dynamic_offset_out = array({1}, int64, nullptr, {});
      dynamic_offset_out->set_data(mlx::core::rocm::malloc_async(sizeof(int64_t), encoder));
      encoder.add_temporary(*dynamic_offset_out);
      // Initialize to zero on GPU using hipMemset
      int64_t* ptr = gpu_ptr<int64_t>(*dynamic_offset_out);
      encoder.launch_kernel([ptr](hipStream_t stream) {
        (void)hipMemsetAsync(ptr, 0, sizeof(int64_t), stream);
      });
    }
    encoder.set_input_array(*dynamic_offset_in);
    encoder.set_input_array(*dynamic_offset_out);
    
    copy_general_dynamic(
        encoder,
        ctype,
        in,
        out,
        offset_in,
        offset_out,
        shape_collapsed,
        strides_vec[0],
        strides_vec[1],
        *dynamic_offset_in,
        *dynamic_offset_out);
    return;
  }
  
  if (ctype == CopyType::Scalar || ctype == CopyType::Vector) {
    copy_contiguous(encoder, ctype, in, out, offset_in, offset_out);
    return;
  }

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    g_copy_stats.record(ctype, shape);
    auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
        shape, std::vector{strides_in, strides_out}, INT32_MAX);
    if (ctype == CopyType::General) {
      copy_general_input(
          encoder,
          ctype,
          in,
          out,
          offset_in,
          offset_out,
          shape_collapsed,
          strides_vec[0]);
    } else {
      copy_general(
          encoder,
          ctype,
          in,
          out,
          offset_in,
          offset_out,
          shape_collapsed,
          strides_vec[0],
          strides_vec[1]);
    }
    return;
  }
}

void fill_gpu(const array& in, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  auto& encoder = rocm::get_command_encoder(s);
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  copy_contiguous(encoder, CopyType::Scalar, in, out, 0, 0);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    auto& encoder = rocm::get_command_encoder(s);
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

} // namespace mlx::core
