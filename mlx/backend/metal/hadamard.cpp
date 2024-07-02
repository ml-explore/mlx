#include <map>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#include <iostream>

namespace mlx::core {

constexpr int MAX_HADAMARD_THREADS_PER_GROUP = 256;

// From http://neilsloane.com/hadamard/
constexpr std::string_view h12 = R"(
+-++++++++++
--+-+-+-+-+-
+++-++----++
+---+--+-++-
+++++-++----
+-+---+--+-+
++--+++-++--
+--++---+--+
++----+++-++
+--+-++---+-
++++----+++-
+-+--+-++---
)";

constexpr std::string_view h20 = R"(
+----+----++--++-++-
-+----+---+++---+-++
--+----+---+++-+-+-+
---+----+---+++++-+-
----+----++--++-++-+
-+++++-----+--+++--+
+-+++-+---+-+--+++--
++-++--+---+-+--+++-
+++-+---+---+-+--+++
++++-----++--+-+--++
--++-+-++-+-----++++
---++-+-++-+---+-+++
+---++-+-+--+--++-++
++---++-+----+-+++-+
-++---++-+----+++++-
-+--+--++-+----+----
+-+-----++-+----+---
-+-+-+---+--+----+--
--+-+++------+----+-
+--+--++------+----+
)";

constexpr std::string_view h28 = R"(
+------++----++-+--+-+--++--
-+-----+++-----+-+--+-+--++-
--+-----+++---+-+-+----+--++
---+-----+++---+-+-+-+--+--+
----+-----+++---+-+-+++--+--
-----+-----++++--+-+--++--+-
------++----++-+--+-+--++--+
--++++-+-------++--+++-+--+-
---++++-+-----+-++--+-+-+--+
+---+++--+----++-++--+-+-+--
++---++---+----++-++--+-+-+-
+++---+----+----++-++--+-+-+
++++--------+-+--++-++--+-+-
-++++--------+++--++--+--+-+
-+-++-++--++--+--------++++-
+-+-++--+--++--+--------++++
-+-+-++--+--++--+----+---+++
+-+-+-++--+--+---+---++---++
++-+-+-++--+------+--+++---+
-++-+-+-++--+------+-++++---
+-++-+---++--+------+-++++--
-++--++-+-++-+++----++------
+-++--++-+-++-+++-----+-----
++-++---+-+-++-+++-----+----
-++-++-+-+-+-+--+++-----+---
--++-++++-+-+----+++-----+--
+--++-+-++-+-+----+++-----+-
++--++-+-++-+-+----++------+
)";

inline const std::map<int, std::string_view> hadamard_matrices() {
  return {{12, h12}, {20, h20}, {28, h28}};
}

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

void launch_hadamard(
    const array& in,
    array& out,
    int batch_size,
    int threads_per,
    const std::string kernel_name,
    float scale,
    const Stream& s) {
  auto& d = metal::device(s.device);

  const auto& lib_name = kernel_name.substr(1);
  auto lib = d.get_library(lib_name);
  auto kernel = d.get_kernel(kernel_name, lib);
  assert(threads_per <= kernel->maxTotalThreadsPerThreadgroup());

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&scale, sizeof(float), 2);

  MTL::Size group_dims = MTL::Size(1, threads_per, 1);
  MTL::Size grid_dims = MTL::Size(batch_size, threads_per, 1);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
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

  int n = in.shape(axis);
  int m = 1;
  if (!is_power_of_2(n)) {
    auto h_matrices = hadamard_matrices();
    for (auto [factor, _] : h_matrices) {
      if (n % factor == 0) {
        m = factor;
        n /= factor;
        break;
      }
    }
    if (m == 1) {
      throw std::invalid_argument(
          "[hadamard] Only supports n = m*2^k where m in (1, 12, 20, 28).");
    }
  }

  int max_radix = std::min(n, 16);
  // Use read_width 2 for m = 28 to avoid register spilling
  int read_width = (n == 2 || m == 28) ? 2 : 4;

  std::ostringstream kname;
  kname << "hadamard_" << n * m << "_" << type_to_name(out);
  auto kernel_name = kname.str();
  auto& d = metal::device(s.device);
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
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
    lib = d.get_library(lib_name, kernel_source.str());
  }

  int batch_size = in.size() / n;
  int threads_per = n / max_radix;

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

    launch_hadamard(
        in_contiguous,
        temp,
        batch_size,
        threads_per,
        "n" + kernel_name,
        1.0,
        s);

    // Metal sometimes reports 256 max threads per group for hadamard_m kernel
    threads_per = std::min(n / read_width, MAX_HADAMARD_THREADS_PER_GROUP);
    batch_size = in.size() / m / read_width / threads_per;
    launch_hadamard(
        temp, out, batch_size, threads_per, "m" + kernel_name, scale_, s);
  } else {
    launch_hadamard(
        in_contiguous,
        out,
        batch_size,
        threads_per,
        "n" + kernel_name,
        scale_,
        s);
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core