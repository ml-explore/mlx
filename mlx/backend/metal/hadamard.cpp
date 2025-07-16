// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_HADAMARD_THREADS_PER_GROUP = 256;

std::string gen_hadamard_codelet(int m) {
  // Generate a O(m^2) hadamard codelet for a given M
  // using the hadamard matrices above
  //
  // e.g. m = 2
  // METAL_FUNC void hadamard_m(thread float *x) {
  //   float tmp[2];
  //   tmp[0] = + x[0] + x[1];
  //   tmp[1] = + x[0] - x[1];
  //   for (int i = 0; i < 2; i++) { x[i] = tmp[i]; }
  // }
  //
  auto h_matrices = hadamard_matrices();
  auto& matrix = h_matrices[m];

  std::ostringstream source;
  source << "METAL_FUNC void hadamard_radix_m(thread float *x) {" << std::endl;
  if (m == 1) {
    source << "}" << std::endl;
    return source.str();
  }
  source << "  float tmp[" << m << "];" << std::endl;
  auto start = 1;
  auto end = matrix.find('\n', start);

  int index = 0;
  while (end != std::string_view::npos) {
    source << "  tmp[" << index << "] = ";
    auto row = matrix.substr(start, end - start);
    for (int i = 0; i < row.length(); i++) {
      source << " " << row[i] << " x[" << i << "]";
    }
    source << ";" << std::endl;
    start = end + 1;
    end = matrix.find('\n', start);
    index++;
  }
  source << "  for (int i = 0; i < " << m << "; i++) { x[i] = tmp[i]; }"
         << std::endl;
  source << "}" << std::endl;
  return source.str();
}

void hadamard_mn_contiguous(
    const array& x,
    array& y,
    int m,
    int n1,
    int n2,
    float scale,
    metal::Device& d,
    const Stream& s) {
  int n = n1 * n2;
  int read_width_n1 = n1 == 2 ? 2 : 4;
  int read_width_n2 = n2 == 2 ? 2 : 4;
  int read_width_m = (n == 2 || m == 28) ? 2 : 4;
  int max_radix_1 = std::min(n1, 16);
  int max_radix_2 = std::min(n2, 16);
  float scale_n1 = 1.0;
  float scale_n2 = (m == 1) ? scale : 1.0;
  float scale_m = scale;

  // n2 is a row contiguous power of 2 hadamard transform
  MTL::Size group_dims_n2(n2 / max_radix_2, 1, 1);
  MTL::Size grid_dims_n2(n2 / max_radix_2, x.size() / n2, 1);

  // n1 is a strided power of 2 hadamard transform with stride n2
  MTL::Size group_dims_n1(n1 / max_radix_1, 1, 1);
  MTL::Size grid_dims_n1(n1 / max_radix_1, x.size() / n, n2);

  // m is a strided hadamard transform with stride n = n1 * n2
  MTL::Size group_dims_m(
      std::min(n / read_width_m, MAX_HADAMARD_THREADS_PER_GROUP), 1, 1);
  MTL::Size grid_dims_m(
      group_dims_m.width, x.size() / m / read_width_m / group_dims_m.width, 1);

  // Make the kernel
  std::string kname;
  kname.reserve(32);
  concatenate(kname, "hadamard_", n * m, "_", type_to_name(x));
  auto lib = d.get_library(kname, [&]() {
    std::string kernel;
    concatenate(
        kernel,
        metal::utils(),
        gen_hadamard_codelet(m),
        metal::hadamard(),
        get_template_definition(
            "n2" + kname,
            "hadamard_n",
            get_type_string(x.dtype()),
            n2,
            max_radix_2,
            read_width_n2));
    if (n1 > 1) {
      kernel += get_template_definition(
          "n1" + kname,
          "hadamard_n",
          get_type_string(x.dtype()),
          n1,
          max_radix_1,
          read_width_n1,
          n2);
    }
    if (m > 1) {
      kernel += get_template_definition(
          "m" + kname,
          "hadamard_m",
          get_type_string(x.dtype()),
          n,
          m,
          read_width_m);
    }
    return kernel;
  });

  // Launch the strided transform for n1
  if (n1 > 1) {
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel("n1" + kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_output_array(y, 1);
    compute_encoder.set_bytes(scale_n1, 2);
    compute_encoder.dispatch_threads(grid_dims_n1, group_dims_n1);
  }

  // Launch the transform for n2
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel("n2" + kname, lib);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(n1 > 1 ? y : x, 0);
  compute_encoder.set_output_array(y, 1);
  compute_encoder.set_bytes(scale_n2, 2);
  compute_encoder.dispatch_threads(grid_dims_n2, group_dims_n2);

  // Launch the strided transform for m
  if (m > 1) {
    auto kernel = d.get_kernel("m" + kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(y, 0);
    compute_encoder.set_output_array(y, 1);
    compute_encoder.set_bytes(scale_m, 2);
    compute_encoder.dispatch_threads(grid_dims_m, group_dims_m);
  }
}

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  // Split the hadamard transform so that all of them work on vectors smaller
  // than 8192 elements.
  //
  // We decompose it in the following way:
  //
  // n = m * n1 * n2 = m * 2^k1 * 2^k2
  //
  // where m is in (1, 12, 20, 28) and n1 and n2 <= 8192
  auto [n, m] = decompose_hadamard(in.shape().back());
  int n1 = 1, n2 = n;
  if (n > 8192) {
    for (n2 = 2; n2 * n2 < n; n2 *= 2) {
    }
    n1 = n / n2;
  }

  if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    hadamard_mn_contiguous(in, out, m, n1, n2, scale_, d, s);
  } else {
    copy_gpu(in, out, CopyType::General, s);
    hadamard_mn_contiguous(out, out, m, n1, n2, scale_, d, s);
  }
}

} // namespace mlx::core
