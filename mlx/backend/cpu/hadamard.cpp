// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

// n = 2^k component
template <typename T>
void hadamard_n(T* out, int n, int m, float scale, size_t size) {
  for (int b = 0; b < size / n; b++) {
    size_t loc = b * n;
    T* data_ptr = out + loc;
    int h = 1;
    int n_over_2 = n / 2;
    while (h < n) {
      for (int i = 0; i < n / 2; i++) {
        int k = i & (h - 1);
        int j = ((i - k) << 1) + k;
        float x = *(data_ptr + j);
        float y = *(data_ptr + j + h);
        *(data_ptr + j) = x + y;
        *(data_ptr + j + h) = x - y;
        if (h == n_over_2) {
          *(data_ptr + j) *= scale;
          *(data_ptr + j + h) *= scale;
        }
      }
      h <<= 1;
    }
  }
}

// m component
template <typename T>
void hadamard_m(T* out, int n, int m, float scale, size_t size) {
  auto h_matrices = hadamard_matrices();
  auto& matrix = h_matrices[m];
  auto start = 1;
  auto end = matrix.find('\n', start);
  std::vector<bool> hmat_vec;
  while (end != std::string_view::npos) {
    auto row = matrix.substr(start, end - start);
    for (int i = 0; i < row.length(); i++) {
      hmat_vec.push_back(row[i] == '+');
    }
    start = end + 1;
    end = matrix.find('\n', start);
  }

  for (int b = 0; b < size / m / n; b++) {
    size_t loc = b * n * m;
    T* data_ptr = out + loc;
    for (int i = 0; i < n; i++) {
      std::vector<float> out(m);
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          float x = *(data_ptr + i + k * n);
          if (hmat_vec[k + j * m]) {
            out[j] += x;
          } else {
            out[j] -= x;
          }
        }
      }
      for (int j = 0; j < m; j++) {
        *(data_ptr + i + j * n) = out[j] * scale;
      }
    }
  }
}

template <typename T>
void hadamard(array& out, int n, int m, float scale, Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  auto out_ptr = out.data<T>();
  encoder.dispatch([out_ptr, size = out.size(), n, m, scale]() {
    float n_scale = m > 1 ? 1.0 : scale;
    hadamard_n<T>(out_ptr, n, m, n_scale, size);
    if (m > 1) {
      hadamard_m<T>(out_ptr, n, m, scale, size);
    }
  });
}

void Hadamard::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Copy input to output
  copy(in, out, CopyType::General, stream());

  int axis = out.ndim() - 1;
  auto [n, m] = decompose_hadamard(out.shape(axis));

  switch (in.dtype()) {
    case float32:
      return hadamard<float>(out, n, m, scale_, stream());
    case float16:
      return hadamard<float16_t>(out, n, m, scale_, stream());
    case bfloat16:
      return hadamard<bfloat16_t>(out, n, m, scale_, stream());
    default:
      throw std::invalid_argument("[hadamard] Unsupported type.");
  }
}

} // namespace mlx::core
