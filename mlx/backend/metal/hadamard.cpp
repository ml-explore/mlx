// Copyright © 2024 Apple Inc.

#include <map>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_HADAMARD_THREADS_PER_GROUP = 256;
constexpr int MAX_HADAMARD_BYTES = 32768; // 32KB

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

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  auto& in = inputs[0];

  std::vector<array> copies;
  // Only support the last axis for now
  int axis = in.ndim() - 1;
  auto check_input = [&copies, &s](const array& x) {
    // TODO(alexbarron) pass strides to kernel to relax this constraint
    bool no_copy = x.flags().row_contiguous;
    if (no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  const array& in_contiguous = check_input(in);

  if (in_contiguous.is_donatable()) {
    out.move_shared_buffer(in_contiguous);
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  int n, m;
  std::tie(n, m) = decompose_hadamard(in.shape(axis));

  if (n * (int)size_of(in.dtype()) > MAX_HADAMARD_BYTES) {
    throw std::invalid_argument(
        "[hadamard] For n = m*2^k, 2^k > 8192 for FP32 or 2^k > 16384 for FP16/BF16 NYI");
  }

  int max_radix = std::min(n, 16);
  // Use read_width 2 for m = 28 to avoid register spilling
  int read_width = (n == 2 || m == 28) ? 2 : 4;

  std::ostringstream kname;
  kname << "hadamard_" << n * m << "_" << type_to_name(out);
  auto kernel_name = kname.str();
  auto& d = metal::device(s.device);
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto codelet = gen_hadamard_codelet(m);
    kernel_source << metal::utils() << codelet << metal::hadamard();
    kernel_source << get_template_definition(
        "n" + kernel_name,
        "hadamard_n",
        get_type_string(in.dtype()),
        n,
        max_radix,
        read_width);
    kernel_source << get_template_definition(
        "m" + kernel_name,
        "hadamard_m",
        get_type_string(in.dtype()),
        n,
        m,
        read_width);
    return kernel_source.str();
  });

  int batch_size = in.size() / n;
  int threads_per = n / max_radix;

  auto& compute_encoder = d.get_command_encoder(s.index);

  auto launch_hadamard = [&](const array& in,
                             array& out,
                             const std::string& kernel_name,
                             float scale) {
    auto kernel = d.get_kernel(kernel_name, lib);
    assert(threads_per <= kernel->maxTotalThreadsPerThreadgroup());

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&scale, sizeof(float), 2);

    MTL::Size group_dims = MTL::Size(1, threads_per, 1);
    MTL::Size grid_dims = MTL::Size(batch_size, threads_per, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  };

  if (m > 1) {
    // When m is greater than 1, we decompose the
    // computation into two uploads to the GPU:
    //
    // e.g. len(x) = 12*4 = 48, m = 12, n = 4
    //
    // y = h48 @ x
    //
    // Upload 1:
    // tmp = a.reshape(12, 4) @ h4
    //
    // Upload 2:
    // y = h12 @ tmp
    array temp(in.shape(), in.dtype(), nullptr, {});
    temp.set_data(allocator::malloc_or_wait(temp.nbytes()));
    copies.push_back(temp);

    launch_hadamard(in_contiguous, temp, "n" + kernel_name, 1.0);

    // Metal sometimes reports 256 max threads per group for hadamard_m kernel
    threads_per = std::min(n / read_width, MAX_HADAMARD_THREADS_PER_GROUP);
    batch_size = in.size() / m / read_width / threads_per;
    launch_hadamard(temp, out, "m" + kernel_name, scale_);
  } else {
    launch_hadamard(in_contiguous, out, "n" + kernel_name, scale_);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
