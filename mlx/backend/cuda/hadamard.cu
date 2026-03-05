// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace mlx::core {

namespace {

constexpr int MAX_HADAMARD_THREADS_PER_BLOCK = 256;

std::string gen_hadamard_codelet(int m) {
  std::ostringstream source;
  source << "namespace mlx::core::cu {\n";
  source << "__device__ __forceinline__ void hadamard_radix_m(float* x) {\n";
  if (m == 1) {
    source << "}\n";
    source << "} // namespace mlx::core::cu\n";
    return source.str();
  }

  auto h_matrices = hadamard_matrices();
  auto it = h_matrices.find(m);
  if (it == h_matrices.end()) {
    throw std::runtime_error("[hadamard] Invalid radix m.");
  }
  auto& matrix = it->second;

  source << "  float tmp[" << m << "];\n";
  auto start = 1;
  auto end = matrix.find('\n', start);
  int row_idx = 0;
  while (end != std::string_view::npos) {
    auto row = matrix.substr(start, end - start);
    source << "  tmp[" << row_idx << "] =";
    for (int i = 0; i < row.length(); ++i) {
      source << " " << row[i] << " x[" << i << "]";
    }
    source << ";\n";
    start = end + 1;
    end = matrix.find('\n', start);
    row_idx++;
  }
  source << "  #pragma unroll\n";
  source << "  for (int i = 0; i < " << m << "; ++i) { x[i] = tmp[i]; }\n";
  source << "}\n";
  source << "} // namespace mlx::core::cu\n";
  return source.str();
}

std::string hadamard_n_kernel_name(
    const Dtype& dtype,
    int n,
    int max_radix,
    int read_width,
    int stride) {
  return fmt::format(
      "mlx::core::cu::hadamard_n<{}, {}, {}, {}, {}>",
      dtype_to_cuda_type(dtype),
      n,
      max_radix,
      read_width,
      stride);
}

std::string
hadamard_m_kernel_name(const Dtype& dtype, int n, int m, int read_width) {
  return fmt::format(
      "mlx::core::cu::hadamard_m<{}, {}, {}, {}>",
      dtype_to_cuda_type(dtype),
      n,
      m,
      read_width);
}

void hadamard_mn_contiguous(
    const array& x,
    array& y,
    int m,
    int n1,
    int n2,
    float scale,
    const Stream& s) {
  const int n = n1 * n2;
  const int read_width_n1 = (n1 == 2) ? 2 : 4;
  const int read_width_n2 = (n2 == 2) ? 2 : 4;
  const int read_width_m = (n == 2 || m == 28) ? 2 : 4;
  const int max_radix_1 = std::min(n1, 16);
  const int max_radix_2 = std::min(n2, 16);
  const float scale_n1 = 1.0f;
  const float scale_n2 = (m == 1) ? scale : 1.0f;
  const float scale_m = scale;

  const std::string n1_kernel_name =
      hadamard_n_kernel_name(x.dtype(), n1, max_radix_1, read_width_n1, n2);
  const std::string n2_kernel_name =
      hadamard_n_kernel_name(x.dtype(), n2, max_radix_2, read_width_n2, 1);
  const std::string m_kernel_name =
      hadamard_m_kernel_name(x.dtype(), n, m, read_width_m);

  const std::string module_name = fmt::format(
      "hadamard_{}_{}_{}_{}_{}_{}_{}_{}",
      dtype_to_string(x.dtype()),
      n,
      m,
      n1,
      n2,
      read_width_n1,
      read_width_n2,
      read_width_m);

  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names = {n2_kernel_name};
    if (n1 > 1) {
      kernel_names.push_back(n1_kernel_name);
    }
    if (m > 1) {
      kernel_names.push_back(m_kernel_name);
    }

    std::string source = R"(
        #include "mlx/backend/cuda/device/utils.cuh"
    )";
    source += gen_hadamard_codelet(m);
    source += R"(
        #include "mlx/backend/cuda/device/hadamard.cuh"
    )";

    return std::make_tuple(false, std::move(source), std::move(kernel_names));
  });

  auto& encoder = cu::get_command_encoder(s);

  if (n1 > 1) {
    const int64_t num_transforms = x.size() / n1;
    const uint32_t num_blocks =
        static_cast<uint32_t>(std::min<int64_t>(num_transforms, 65535));

    encoder.set_input_array(x);
    encoder.set_output_array(y);

    cu::KernelArgs args;
    args.append(x);
    args.append(y);
    args.append(scale_n1);
    args.append(num_transforms);

    auto kernel = mod.get_kernel(n1_kernel_name);
    encoder.add_kernel_node_raw(
        kernel, num_blocks, n1 / max_radix_1, {}, 0, args.args());
  }

  {
    const auto& in = (n1 > 1) ? y : x;
    const int64_t num_transforms = x.size() / n2;
    const uint32_t num_blocks =
        static_cast<uint32_t>(std::min<int64_t>(num_transforms, 65535));

    encoder.set_input_array(in);
    encoder.set_output_array(y);

    cu::KernelArgs args;
    args.append(in);
    args.append(y);
    args.append(scale_n2);
    args.append(num_transforms);

    auto kernel = mod.get_kernel(n2_kernel_name);
    encoder.add_kernel_node_raw(
        kernel, num_blocks, n2 / max_radix_2, {}, 0, args.args());
  }

  if (m > 1) {
    const int64_t num_tasks = x.size() / (m * read_width_m);
    const uint32_t block_dim = static_cast<uint32_t>(
        std::min<int64_t>(num_tasks, MAX_HADAMARD_THREADS_PER_BLOCK));
    const uint32_t num_blocks = static_cast<uint32_t>(
        std::min<int64_t>((num_tasks + block_dim - 1) / block_dim, 65535));

    encoder.set_input_array(y);
    encoder.set_output_array(y);

    cu::KernelArgs args;
    args.append(y);
    args.append(y);
    args.append(scale_m);
    args.append(num_tasks);

    auto kernel = mod.get_kernel(m_kernel_name);
    encoder.add_kernel_node_raw(
        kernel, num_blocks, block_dim, {}, 0, args.args());
  }
}

} // namespace

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Hadamard::eval_gpu");
  assert(inputs.size() == 1);

  auto& in = inputs[0];
  if (in.dtype() != float16 && in.dtype() != bfloat16 &&
      in.dtype() != float32) {
    throw std::invalid_argument("[hadamard] Unsupported type.");
  }

  // n = m * 2^k where m in (1, 12, 20, 28)
  auto [n, m] = decompose_hadamard(in.shape().back());
  int n1 = 1;
  int n2 = n;
  if (n > 8192) {
    for (n2 = 2; n2 * n2 < n; n2 *= 2) {
    }
    n1 = n / n2;
  }

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
    }
    hadamard_mn_contiguous(in, out, m, n1, n2, scale_, s);
  } else {
    copy_gpu(in, out, CopyType::General, s);
    hadamard_mn_contiguous(out, out, m, n1, n2, scale_, s);
  }
}

} // namespace mlx::core
