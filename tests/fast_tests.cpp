// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/fast.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

void check_rope_float32(bool traditional) {
  constexpr int B = 1;
  constexpr int H = 1;
  constexpr int T = 2;
  constexpr int D = 16;
  constexpr int dims = 16;
  constexpr int half_dims = dims / 2;
  constexpr float base = 10000.0f;
  constexpr float scale = 1.0f;

  std::vector<float> values(B * H * T * D);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = 0.125f * static_cast<float>(i - 9);
  }

  array x(values.data(), {B, H, T, D});
  array y = fast::rope(x, dims, traditional, base, scale, 0);
  y.eval();

  std::vector<float> expected(values.size());
  const float log_base = std::log(base);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < half_dims; ++j) {
      const float theta = static_cast<float>(t) *
          std::exp(static_cast<float>(-j) * log_base / half_dims);
      const float c = std::cos(theta);
      const float s = std::sin(theta);

      if (traditional) {
        const size_t idx = static_cast<size_t>(t) * D + 2 * j;
        const float x0 = values[idx];
        const float x1 = values[idx + 1];
        expected[idx] = x0 * c - x1 * s;
        expected[idx + 1] = x0 * s + x1 * c;
      } else {
        const size_t idx0 = static_cast<size_t>(t) * D + j;
        const size_t idx1 = static_cast<size_t>(t) * D + j + half_dims;
        const float x0 = values[idx0];
        const float x1 = values[idx1];
        expected[idx0] = x0 * c - x1 * s;
        expected[idx1] = x0 * s + x1 * c;
      }
    }
  }

  const float* out = y.data<float>();
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5));
  }
}

template <typename T>
void check_rope_typed(bool traditional, float tol) {
  constexpr int B = 1;
  constexpr int H = 1;
  constexpr int T_len = 2;
  constexpr int D = 16;
  constexpr int dims = 16;
  constexpr int half_dims = dims / 2;
  constexpr float base = 10000.0f;
  constexpr float scale = 1.0f;

  std::vector<T> values(B * H * T_len * D);
  std::vector<float> values_f32(values.size());
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values_f32[i] = 0.125f * static_cast<float>(i - 9);
    values[i] = static_cast<T>(values_f32[i]);
  }

  array x(values.data(), {B, H, T_len, D});
  array y = fast::rope(x, dims, traditional, base, scale, 0);
  y.eval();

  std::vector<float> expected(values.size());
  const float log_base = std::log(base);
  for (int t = 0; t < T_len; ++t) {
    for (int j = 0; j < half_dims; ++j) {
      const float theta = static_cast<float>(t) *
          std::exp(static_cast<float>(-j) * log_base / half_dims);
      const float c = std::cos(theta);
      const float s = std::sin(theta);

      if (traditional) {
        const size_t idx = static_cast<size_t>(t) * D + 2 * j;
        const float x0 = values_f32[idx];
        const float x1 = values_f32[idx + 1];
        expected[idx] = x0 * c - x1 * s;
        expected[idx + 1] = x0 * s + x1 * c;
      } else {
        const size_t idx0 = static_cast<size_t>(t) * D + j;
        const size_t idx1 = static_cast<size_t>(t) * D + j + half_dims;
        const float x0 = values_f32[idx0];
        const float x1 = values_f32[idx1];
        expected[idx0] = x0 * c - x1 * s;
        expected[idx1] = x0 * s + x1 * c;
      }
    }
  }

  const T* out = y.data<T>();
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    CHECK(
        static_cast<float>(out[i]) ==
        doctest::Approx(expected[i]).epsilon(tol));
  }
}

void check_rms_norm_float32(bool has_weight) {
  constexpr int rows = 2;
  constexpr int width = 16;
  constexpr float eps = 1e-5f;

  std::vector<float> values(rows * width);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = 0.2f * static_cast<float>(i - 11);
  }

  std::vector<float> weights(width);
  for (int i = 0; i < width; ++i) {
    weights[i] = 0.75f + 0.03f * static_cast<float>(i);
  }

  array x(values.data(), {rows, width});
  std::optional<array> weight;
  if (has_weight) {
    weight = array(weights.data(), {width});
  }

  array y = fast::rms_norm(x, weight, eps);
  y.eval();

  std::vector<float> expected(values.size());
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = values.data() + row * width;
    float sum_sq = 0.0f;
    for (int i = 0; i < width; ++i) {
      sum_sq += row_ptr[i] * row_ptr[i];
    }
    const float scale = 1.0f / std::sqrt(sum_sq / width + eps);
    for (int i = 0; i < width; ++i) {
      float value = row_ptr[i] * scale;
      if (has_weight) {
        value *= weights[i];
      }
      expected[row * width + i] = value;
    }
  }

  const float* out = y.data<float>();
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5));
  }
}

void check_layer_norm_float32(bool has_weight, bool has_bias) {
  constexpr int rows = 2;
  constexpr int width = 16;
  constexpr float eps = 1e-5f;

  std::vector<float> values(rows * width);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = 0.15f * static_cast<float>(i - 7);
  }

  std::vector<float> weights(width);
  std::vector<float> biases(width);
  for (int i = 0; i < width; ++i) {
    weights[i] = 0.8f + 0.025f * static_cast<float>(i);
    biases[i] = -0.2f + 0.01f * static_cast<float>(i);
  }

  array x(values.data(), {rows, width});
  std::optional<array> weight;
  std::optional<array> bias;
  if (has_weight) {
    weight = array(weights.data(), {width});
  }
  if (has_bias) {
    bias = array(biases.data(), {width});
  }

  array y = fast::layer_norm(x, weight, bias, eps);
  y.eval();

  std::vector<float> expected(values.size());
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = values.data() + row * width;
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
      sum += row_ptr[i];
    }
    const float mean = sum / width;

    float variance = 0.0f;
    for (int i = 0; i < width; ++i) {
      const float diff = row_ptr[i] - mean;
      variance += diff * diff;
    }
    variance /= width;
    const float scale = 1.0f / std::sqrt(variance + eps);

    for (int i = 0; i < width; ++i) {
      float value = (row_ptr[i] - mean) * scale;
      if (has_weight) {
        value *= weights[i];
      }
      if (has_bias) {
        value += biases[i];
      }
      expected[row * width + i] = value;
    }
  }

  const float* out = y.data<float>();
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5));
  }
}

void check_quantized_matmul_float32(int bits) {
  constexpr int M = 32;
  constexpr int N = 8;
  constexpr int K = 64;
  constexpr int group_size = 32;

  std::vector<float> x_values(M * K);
  for (int i = 0; i < static_cast<int>(x_values.size()); ++i) {
    x_values[i] = 0.02f * static_cast<float>((i % 37) - 18);
  }

  std::vector<float> w_values(N * K);
  for (int i = 0; i < static_cast<int>(w_values.size()); ++i) {
    w_values[i] = 0.03f * static_cast<float>((i % 29) - 14);
  }

  array x(x_values.data(), {M, K});
  array w(w_values.data(), {N, K});
  auto q = quantize(w, group_size, bits);
  array y = quantized_matmul(
      x,
      q[0],
      q[1],
      q[2],
      /* transpose = */ true,
      group_size,
      bits);
  array expected =
      matmul(x, transpose(dequantize(q[0], q[1], q[2], group_size, bits)));

  y.eval();
  expected.eval();
  CHECK(allclose(y, expected, 1e-5, 1e-5).item<bool>());
}

void check_quantized_matmul_token_float32(int bits, int M) {
  constexpr int N = 256;
  constexpr int K = 64;
  constexpr int group_size = 32;
  constexpr int q_pattern[group_size] = {
      -127, -119, -111, -103, -95, -87, -79, -71, -63, -55, -47,
      -39,  -31,  -23,  -15,  -7,  7,   15,  23,  31,  39,  47,
      55,   63,   71,   79,   87,  95,  103, 111, 119, 127};

  std::vector<float> x_values(M * K);
  for (int i = 0; i < static_cast<int>(x_values.size()); ++i) {
    x_values[i] = static_cast<float>(q_pattern[i % group_size]) / 127.0f;
  }

  std::vector<float> w_values(N * K);
  for (int i = 0; i < static_cast<int>(w_values.size()); ++i) {
    w_values[i] = 0.015f * static_cast<float>((i % 43) - 21);
  }

  array x(x_values.data(), {M, K});
  array w(w_values.data(), {N, K});
  auto q = quantize(w, group_size, bits);
  array y = quantized_matmul(
      x,
      q[0],
      q[1],
      q[2],
      /* transpose = */ true,
      group_size,
      bits);
  array expected =
      matmul(x, transpose(dequantize(q[0], q[1], q[2], group_size, bits)));

  y.eval();
  expected.eval();
  CHECK(allclose(y, expected, 1e-4, 1e-4).item<bool>());
}

uint32_t pack_u8x4(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
  return static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 8) |
      (static_cast<uint32_t>(v2) << 16) | (static_cast<uint32_t>(v3) << 24);
}

void check_quantized_matmul_8bit_high_values() {
  constexpr int M = 3;
  constexpr int N = 72;
  constexpr int K = 64;
  constexpr int bits = 8;
  constexpr int group_size = 32;
  constexpr int pack_factor = 4;
  constexpr int packs_per_col = K / pack_factor;
  constexpr int groups_per_col = K / group_size;
  static constexpr uint8_t q_pattern[16] = {
      0, 1, 2, 3, 7, 31, 63, 95, 127, 128, 129, 191, 223, 240, 254, 255};

  auto q_value = [](int n, int k) { return q_pattern[(n * 5 + k * 3) & 15]; };

  std::vector<uint32_t> w_values(N * packs_per_col);
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < packs_per_col; ++p) {
      const int k = p * pack_factor;
      w_values[n * packs_per_col + p] = pack_u8x4(
          q_value(n, k),
          q_value(n, k + 1),
          q_value(n, k + 2),
          q_value(n, k + 3));
    }
  }

  std::vector<float> scales(N * groups_per_col);
  std::vector<float> biases(N * groups_per_col);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < groups_per_col; ++g) {
      scales[n * groups_per_col + g] =
          0.0015f * static_cast<float>((n % 7) + 1) +
          0.00025f * static_cast<float>(g);
      biases[n * groups_per_col + g] = -0.08f +
          0.002f * static_cast<float>(n % 11) + 0.015f * static_cast<float>(g);
    }
  }

  std::vector<float> x_values(M * K);
  for (int i = 0; i < static_cast<int>(x_values.size()); ++i) {
    x_values[i] = 0.03125f * static_cast<float>((i * 7) % 31 - 15);
  }

  std::vector<int8_t> x_q(M * K);
  std::vector<float> x_scales(M * groups_per_col);
  std::vector<float> x_group_sums(M * groups_per_col);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < groups_per_col; ++g) {
      float sum_x = 0.0f;
      float max_value = 0.0f;
      for (int k = 0; k < group_size; ++k) {
        const int kk = g * group_size + k;
        const float x = x_values[m * K + kk];
        sum_x += x;
        max_value = std::max(max_value, std::abs(x));
      }

      const float inv_scale = max_value > 0.0f ? 127.0f / max_value : 0.0f;
      x_scales[m * groups_per_col + g] = max_value / 127.0f;
      x_group_sums[m * groups_per_col + g] = sum_x;
      for (int k = 0; k < group_size; ++k) {
        const int kk = g * group_size + k;
        int q =
            static_cast<int>(std::nearbyint(x_values[m * K + kk] * inv_scale));
        q = std::min(127, std::max(-127, q));
        x_q[m * K + kk] = static_cast<int8_t>(q);
      }
    }
  }

  std::vector<float> expected(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int g = 0; g < groups_per_col; ++g) {
        const float scale = scales[n * groups_per_col + g];
        const float bias = biases[n * groups_per_col + g];
        int dot = 0;
        for (int k = 0; k < group_size; ++k) {
          const int kk = g * group_size + k;
          dot += static_cast<int>(x_q[m * K + kk]) *
              static_cast<int>(q_value(n, kk));
        }
        sum +=
            scale * x_scales[m * groups_per_col + g] * static_cast<float>(dot) +
            bias * x_group_sums[m * groups_per_col + g];
      }
      expected[m * N + n] = sum;
    }
  }

  array x(x_values.data(), {M, K});
  array w(w_values.data(), {N, packs_per_col});
  array scale_arr(scales.data(), {N, groups_per_col});
  array bias_arr(biases.data(), {N, groups_per_col});
  array y = quantized_matmul(
      x, w, scale_arr, bias_arr, /* transpose = */ true, group_size, bits);

  y.eval();
  const float* out = y.data<float>();
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    CHECK(std::abs(out[i] - expected[i]) < 2e-4f);
  }
}

template <typename T>
void check_quantized_matmul_token_typed(int bits, int M, float tol) {
  constexpr int N = 256;
  constexpr int K = 64;
  constexpr int group_size = 32;
  constexpr int q_pattern[group_size] = {
      -127, -119, -111, -103, -95, -87, -79, -71, -63, -55, -47,
      -39,  -31,  -23,  -15,  -7,  7,   15,  23,  31,  39,  47,
      55,   63,   71,   79,   87,  95,  103, 111, 119, 127};

  std::vector<T> x_values(M * K);
  for (int i = 0; i < static_cast<int>(x_values.size()); ++i) {
    x_values[i] =
        static_cast<T>(static_cast<float>(q_pattern[i % group_size]) / 127.0f);
  }

  std::vector<T> w_values(N * K);
  for (int i = 0; i < static_cast<int>(w_values.size()); ++i) {
    w_values[i] = static_cast<T>(0.015f * static_cast<float>((i % 43) - 21));
  }

  array x(x_values.data(), {M, K});
  array w(w_values.data(), {N, K});
  auto q = quantize(w, group_size, bits);
  array y = quantized_matmul(
      x,
      q[0],
      q[1],
      q[2],
      /* transpose = */ true,
      group_size,
      bits);
  array expected =
      matmul(x, transpose(dequantize(q[0], q[1], q[2], group_size, bits)));

  y.eval();
  expected.eval();
  CHECK(allclose(y, expected, tol, tol).item<bool>());
}

} // namespace

TEST_CASE("test fast rope traditional float32") {
  check_rope_float32(true);
}

TEST_CASE("test fast rope non-traditional float32") {
  check_rope_float32(false);
}

TEST_CASE("test fast rope float16") {
  check_rope_typed<float16_t>(true, 2e-3f);
  check_rope_typed<float16_t>(false, 2e-3f);
}

TEST_CASE("test fast rms_norm float32") {
  check_rms_norm_float32(false);
  check_rms_norm_float32(true);
}

TEST_CASE("test fast layer_norm float32") {
  check_layer_norm_float32(false, false);
  check_layer_norm_float32(true, true);
}

TEST_CASE("test fast quantized matmul prompt float32") {
  check_quantized_matmul_float32(4);
  check_quantized_matmul_float32(8);
}

TEST_CASE("test fast quantized matmul token float32") {
  check_quantized_matmul_token_float32(4, 1);
  check_quantized_matmul_token_float32(8, 1);
  check_quantized_matmul_token_float32(4, 4);
  check_quantized_matmul_token_float32(8, 4);
  check_quantized_matmul_8bit_high_values();
}

TEST_CASE("test fast quantized matmul token float16") {
  check_quantized_matmul_token_typed<float16_t>(4, 1, 5e-3f);
  check_quantized_matmul_token_typed<float16_t>(8, 1, 5e-3f);
}

TEST_CASE("test fast quantized matmul token bfloat16") {
  check_quantized_matmul_token_typed<bfloat16_t>(4, 1, 8e-3f);
  check_quantized_matmul_token_typed<bfloat16_t>(8, 1, 8e-3f);
}
