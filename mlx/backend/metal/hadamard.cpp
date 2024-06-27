#include <map>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#include <iostream>

namespace mlx::core {

// From http://neilsloane.com/hadamard/

constexpr std::string_view h12 = R"(
+-----------
++-+---+++-+
+++-+---+++-
+-++-+---+++
++-++-+---++
+++-++-+---+
++++-++-+---
+-+++-++-+--
+--+++-++-+-
+---+++-++-+
++---+++-++-
+-+---+++-++
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

constexpr std::string_view h40 = R"(
+-------------------+-------------------
++-++----+-+-++++--+++-++----+-+-++++--+
+++-++----+-+-++++--+++-++----+-+-++++--
+-++-++----+-+-++++-+-++-++----+-+-++++-
+--++-++----+-+-+++++--++-++----+-+-++++
++--++-++----+-+-+++++--++-++----+-+-+++
+++--++-++----+-+-+++++--++-++----+-+-++
++++--++-++----+-+-+++++--++-++----+-+-+
+++++--++-++----+-+-+++++--++-++----+-+-
+-++++--++-++----+-++-++++--++-++----+-+
++-++++--++-++----+-++-++++--++-++----+-
+-+-++++--++-++----++-+-++++--++-++----+
++-+-++++--++-++----++-+-++++--++-++----
+-+-+-++++--++-++---+-+-+-++++--++-++---
+--+-+-++++--++-++--+--+-+-++++--++-++--
+---+-+-++++--++-++-+---+-+-++++--++-++-
+----+-+-++++--++-+++----+-+-++++--++-++
++----+-+-++++--++-+++----+-+-++++--++-+
+++----+-+-++++--++-+++----+-+-++++--++-
+-++----+-+-++++--+++-++----+-+-++++--++
+--------------------+++++++++++++++++++
++-++----+-+-++++--+--+--++++-+-+----++-
+++-++----+-+-++++-----+--++++-+-+----++
+-++-++----+-+-++++--+--+--++++-+-+----+
+--++-++----+-+-++++-++--+--++++-+-+----
++--++-++----+-+-+++--++--+--++++-+-+---
+++--++-++----+-+-++---++--+--++++-+-+--
++++--++-++----+-+-+----++--+--++++-+-+-
+++++--++-++----+-+------++--+--++++-+-+
+-++++--++-++----+-+-+----++--+--++++-+-
++-++++--++-++----+---+----++--+--++++-+
+-+-++++--++-++----+-+-+----++--+--++++-
++-+-++++--++-++------+-+----++--+--++++
+-+-+-++++--++-++----+-+-+----++--+--+++
+--+-+-++++--++-++---++-+-+----++--+--++
+---+-+-++++--++-++--+++-+-+----++--+--+
+----+-+-++++--++-++-++++-+-+----++--+--
++----+-+-++++--++-+--++++-+-+----++--+-
+++----+-+-++++--++----++++-+-+----++--+
+-++----+-+-++++--++-+--++++-+-+----++--
)";

inline const std::map<int, std::string_view> hadamard_matrices() {
  return {{12, h12}, {20, h20}, {28, h28}, {40, h40}};
}

std::string gen_hadamard_codelet(int m) {
  // Generate a O(m^2) hadamard codelet for a given M
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
  source << "METAL_FUNC void hadamard_m(thread float *x) {" << std::endl;
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
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  int n = in.shape(axis_);
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
  }

  // std::cout << "m " << m << std::endl;
  // std::cout << "n " << n << std::endl;

  constexpr int max_radix = 16;
  int batch_size = in.size() / n / m;
  int threads_per = n / max_radix;

  std::ostringstream kname;
  kname << "hadamard_" << n << "_" << type_to_name(out);
  auto kernel_name = kname.str();
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    auto codelet = gen_hadamard_codelet(m);
    auto template_def = get_template_definition(
        kernel_name, "hadamard", get_type_string(in.dtype()), n);
    kernel_source << metal::utils() << codelet << metal::hadamard()
                  << template_def;
    // std::cout << "kernel source " << kernel_source.str() << std::endl;
    lib = d.get_library(lib_name, kernel_source.str());
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  // TODO ensure contiguity

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);

  // std::cout << "batch size " << batch_size << std::endl;
  // std::cout << "threads_per " << threads_per << std::endl;

  MTL::Size group_dims = MTL::Size(1, threads_per, 1);
  MTL::Size grid_dims = MTL::Size(batch_size, threads_per, 1);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core