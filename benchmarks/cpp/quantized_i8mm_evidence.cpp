// Copyright © 2026 Apple Inc.
//
// Evidence-only prototype for mlx#4197. This is deliberately not wired into
// the MLX CPU backend. It uses the same affine 8-bit/group-64 weight layout as
// qmm_t, quantizes each activation group once, and times that preparation
// separately from the I8MM dot stage.
//
// Build through the opt-in CMake target on Apple arm64 so the frozen source
// SHA256 is embedded in the candidate.
//
// See README_i8mm_evidence.md for the MLX baseline runner and Highway handoff.

#include <arm_neon.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef __ARM_FEATURE_MATMUL_INT8
#error "Compile this evidence prototype with -march=armv8.6-a+i8mm"
#endif

#ifndef MLX_I8MM_EVIDENCE_SOURCE_SHA256
#error \
    "Build quantized_i8mm_evidence through its CMake target so source identity is embedded"
#endif

namespace {

constexpr uint32_t kFixtureMagic = 0x384d4d51; // "QMM8" in a little-endian u32.
constexpr uint32_t kFixtureVersion = 1;

struct FixtureHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t group_size;
  uint32_t reserved0;
  uint32_t reserved1;
};

static_assert(sizeof(FixtureHeader) == 8 * sizeof(uint32_t));

struct Fixture {
  int m;
  int n;
  int k;
  int group_size;
  std::vector<float> x; // [M, K]
  std::vector<uint8_t> weights; // raw MLX packed 8-bit weights, [N, K]
  std::vector<float> scales; // [N, K / group_size]
  std::vector<float> biases; // [N, K / group_size]
};

struct PreparedActivations {
  std::vector<int8_t> values; // [M, K]
  std::vector<float> scales; // [M, K / group_size]
  std::vector<float> sums; // raw, not quantized: [M, K / group_size]
};

struct ErrorMetrics {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double normalized_max_abs = 0.0;
  double rmse = 0.0;
  double cosine = 1.0;
  double reference_max_abs = 0.0;
};

bool i8mm_available() {
  int enabled = 0;
  size_t size = sizeof(enabled);
  return sysctlbyname(
             "hw.optional.arm.FEAT_I8MM", &enabled, &size, nullptr, 0) == 0 &&
      enabled == 1;
}

int groups_per_row(const Fixture& f) {
  if (f.group_size != 64 || f.k <= 0 || f.k % f.group_size != 0 ||
      f.group_size % 8 != 0) {
    throw std::invalid_argument(
        "prototype requires 8-bit affine weights with K divisible by group-64");
  }
  return f.k / f.group_size;
}

void require_finite(float value, const char* what) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(
        std::string("non-finite ") + what +
        " is not a valid int8 activation input");
  }
}

PreparedActivations quantize_activations_once(const Fixture& f) {
  const int groups = groups_per_row(f);
  PreparedActivations prepared;
  prepared.values.resize(static_cast<size_t>(f.m) * f.k);
  prepared.scales.resize(static_cast<size_t>(f.m) * groups);
  prepared.sums.resize(static_cast<size_t>(f.m) * groups);

  for (int m = 0; m < f.m; ++m) {
    for (int group = 0; group < groups; ++group) {
      const size_t offset = static_cast<size_t>(m) * f.k +
          static_cast<size_t>(group) * f.group_size;
      float max_abs = 0.0f;
      float raw_sum = 0.0f;
      for (int e = 0; e < f.group_size; ++e) {
        const float value = f.x[offset + e];
        require_finite(value, "activation");
        max_abs = std::max(max_abs, std::fabs(value));
        raw_sum += value;
      }

      // Match the #3019 Highway proposal: an all-zero group has scale zero,
      // while the values remain zero. Bias correction uses the exact raw sum.
      const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
      prepared.scales[static_cast<size_t>(m) * groups + group] =
          max_abs / 127.0f;
      prepared.sums[static_cast<size_t>(m) * groups + group] = raw_sum;
      for (int e = 0; e < f.group_size; ++e) {
        const float scaled = f.x[offset + e] * inv_scale;
        const int rounded = static_cast<int>(std::nearbyint(scaled));
        prepared.values[offset + e] =
            static_cast<int8_t>(std::min(127, std::max(-127, rounded)));
      }
    }
  }
  return prepared;
}

std::vector<float> reference_affine(const Fixture& f) {
  const int groups = groups_per_row(f);
  std::vector<float> out(static_cast<size_t>(f.m) * f.n, 0.0f);
  for (int m = 0; m < f.m; ++m) {
    for (int n = 0; n < f.n; ++n) {
      float acc = 0.0f;
      for (int group = 0; group < groups; ++group) {
        const size_t parameter = static_cast<size_t>(n) * groups + group;
        const size_t offset = static_cast<size_t>(group) * f.group_size;
        const float scale = f.scales[parameter];
        const float bias = f.biases[parameter];
        for (int e = 0; e < f.group_size; ++e) {
          const float x = f.x[static_cast<size_t>(m) * f.k + offset + e];
          const uint8_t w =
              f.weights[static_cast<size_t>(n) * f.k + offset + e];
          acc += x * (scale * static_cast<float>(w) + bias);
        }
      }
      out[static_cast<size_t>(m) * f.n + n] = acc;
    }
  }
  return out;
}

// Same integer/int8 approximation as the I8MM candidate, but scalar. The
// self-test requires I8MM to agree with this path; quality against affine MLX
// semantics is deliberately reported, never thresholded as release evidence.
std::vector<float> scalar_prequantized(
    const Fixture& f,
    const PreparedActivations& prepared) {
  const int groups = groups_per_row(f);
  std::vector<float> out(static_cast<size_t>(f.m) * f.n, 0.0f);
  for (int m = 0; m < f.m; ++m) {
    for (int n = 0; n < f.n; ++n) {
      float acc = 0.0f;
      for (int group = 0; group < groups; ++group) {
        int32_t dot = 0;
        const size_t offset = static_cast<size_t>(group) * f.group_size;
        for (int e = 0; e < f.group_size; ++e) {
          dot +=
              static_cast<int32_t>(
                  f.weights[static_cast<size_t>(n) * f.k + offset + e]) *
              static_cast<int32_t>(
                  prepared.values[static_cast<size_t>(m) * f.k + offset + e]);
        }
        const size_t parameter = static_cast<size_t>(n) * groups + group;
        const size_t x_parameter = static_cast<size_t>(m) * groups + group;
        acc += f.scales[parameter] * prepared.scales[x_parameter] *
            static_cast<float>(dot);
        acc += f.biases[parameter] * prepared.sums[x_parameter];
      }
      out[static_cast<size_t>(m) * f.n + n] = acc;
    }
  }
  return out;
}

std::array<int32_t, 4> i8mm_dot_two_by_two(
    const int8_t* a0,
    const int8_t* a1,
    const uint8_t* b0,
    const uint8_t* b1,
    int group_size) {
  int32x4_t acc = vdupq_n_s32(0);
  for (int e = 0; e < group_size; e += 8) {
    const int8x8_t row0 = vld1_s8(a0 + e);
    const int8x8_t row1 = a1 ? vld1_s8(a1 + e) : vdup_n_s8(0);
    const int8x16_t a = vcombine_s8(row0, row1);
    const uint8x8_t column0 = vld1_u8(b0 + e);
    const uint8x8_t column1 = b1 ? vld1_u8(b1 + e) : vdup_n_u8(0);
    const uint8x16_t b = vcombine_u8(column0, column1);
    // Mixed U8 x I8 I8MM. Highway currently exposes the equivalent U8*I8
    // dot primitive; its signed 2x2 primitive needs recentering or a new
    // mixed-type wrapper before this can be extracted there.
    acc = vusmmlaq_s32(acc, b, a);
  }
  std::array<int32_t, 4> lanes{};
  vst1q_s32(lanes.data(), acc);
  return lanes;
}

int32_t dotprod_u8_i8(
    const uint8_t* weights,
    const int8_t* activations,
    int group_size) {
  int32x4_t acc = vdupq_n_s32(0);
  for (int e = 0; e < group_size; e += 16) {
    acc = vusdotq_s32(acc, vld1q_u8(weights + e), vld1q_s8(activations + e));
  }
  return vaddvq_s32(acc);
}

void dotprod_prequantized_range(
    const Fixture& f,
    const PreparedActivations& prepared,
    int n_begin,
    int n_end,
    std::vector<float>* out) {
  const int groups = groups_per_row(f);
  if (n_begin < 0 || n_begin > n_end || n_end > f.n) {
    throw std::invalid_argument("invalid output-column range");
  }
  for (int m = 0; m < f.m; ++m) {
    for (int n = n_begin; n < n_end; ++n) {
      float acc = 0.0f;
      for (int group = 0; group < groups; ++group) {
        const size_t k_offset = static_cast<size_t>(group) * f.group_size;
        const size_t x_parameter = static_cast<size_t>(m) * groups + group;
        const size_t w_parameter = static_cast<size_t>(n) * groups + group;
        const int32_t dot = dotprod_u8_i8(
            f.weights.data() + static_cast<size_t>(n) * f.k + k_offset,
            prepared.values.data() + static_cast<size_t>(m) * f.k + k_offset,
            f.group_size);
        acc += f.scales[w_parameter] * prepared.scales[x_parameter] *
            static_cast<float>(dot);
        acc += f.biases[w_parameter] * prepared.sums[x_parameter];
      }
      (*out)[static_cast<size_t>(m) * f.n + n] = acc;
    }
  }
}

std::vector<float> dotprod_prequantized(
    const Fixture& f,
    const PreparedActivations& prepared,
    int chunk_columns) {
  std::vector<float> out(static_cast<size_t>(f.m) * f.n, 0.0f);
  for (int begin = 0; begin < f.n; begin += chunk_columns) {
    dotprod_prequantized_range(
        f, prepared, begin, std::min(begin + chunk_columns, f.n), &out);
  }
  return out;
}

void i8mm_prequantized_range(
    const Fixture& f,
    const PreparedActivations& prepared,
    int n_begin,
    int n_end,
    std::vector<float>* out) {
  const int groups = groups_per_row(f);
  if (n_begin < 0 || n_begin > n_end || n_end > f.n) {
    throw std::invalid_argument("invalid output-column range");
  }

  for (int m = 0; m < f.m; m += 2) {
    const bool has_second_row = m + 1 < f.m;
    for (int group = 0; group < groups; ++group) {
      const size_t k_offset = static_cast<size_t>(group) * f.group_size;
      const size_t x0 = static_cast<size_t>(m) * f.k + k_offset;
      const size_t x1 = static_cast<size_t>(m + 1) * f.k + k_offset;
      const size_t p0 = static_cast<size_t>(m) * groups + group;
      const size_t p1 = static_cast<size_t>(m + 1) * groups + group;
      const float x_scale0 = prepared.scales[p0];
      const float x_scale1 = has_second_row ? prepared.scales[p1] : 0.0f;
      const float x_sum0 = prepared.sums[p0];
      const float x_sum1 = has_second_row ? prepared.sums[p1] : 0.0f;

      for (int n = n_begin; n < n_end; n += 2) {
        const bool has_second_column = n + 1 < n_end;
        const uint8_t* w0 =
            f.weights.data() + static_cast<size_t>(n) * f.k + k_offset;
        const uint8_t* w1 = has_second_column
            ? f.weights.data() + static_cast<size_t>(n + 1) * f.k + k_offset
            : nullptr;
        const auto lanes = i8mm_dot_two_by_two(
            prepared.values.data() + x0,
            has_second_row ? prepared.values.data() + x1 : nullptr,
            w0,
            w1,
            f.group_size);
        const size_t c0 = static_cast<size_t>(n) * groups + group;
        (*out)[static_cast<size_t>(m) * f.n + n] +=
            f.scales[c0] * x_scale0 * static_cast<float>(lanes[0]) +
            f.biases[c0] * x_sum0;
        if (has_second_row) {
          (*out)[static_cast<size_t>(m + 1) * f.n + n] +=
              f.scales[c0] * x_scale1 * static_cast<float>(lanes[1]) +
              f.biases[c0] * x_sum1;
        }
        if (has_second_column) {
          const size_t c1 = static_cast<size_t>(n + 1) * groups + group;
          (*out)[static_cast<size_t>(m) * f.n + n + 1] +=
              f.scales[c1] * x_scale0 * static_cast<float>(lanes[2]) +
              f.biases[c1] * x_sum0;
          if (has_second_row) {
            (*out)[static_cast<size_t>(m + 1) * f.n + n + 1] +=
                f.scales[c1] * x_scale1 * static_cast<float>(lanes[3]) +
                f.biases[c1] * x_sum1;
          }
        }
      }
    }
  }
}

std::vector<float> i8mm_prequantized(
    const Fixture& f,
    const PreparedActivations& prepared,
    int chunk_columns) {
  std::vector<float> out(static_cast<size_t>(f.m) * f.n, 0.0f);
  for (int begin = 0; begin < f.n; begin += chunk_columns) {
    i8mm_prequantized_range(
        f, prepared, begin, std::min(begin + chunk_columns, f.n), &out);
  }
  return out;
}

std::vector<float> i8mm_requantized_per_chunk(
    const Fixture& f,
    int chunk_columns) {
  std::vector<float> out(static_cast<size_t>(f.m) * f.n, 0.0f);
  for (int begin = 0; begin < f.n; begin += chunk_columns) {
    // This intentionally represents the M>1 #3019 accounting bug: the same
    // activation rows are prepared again for every output-column chunk.
    const auto prepared = quantize_activations_once(f);
    i8mm_prequantized_range(
        f, prepared, begin, std::min(begin + chunk_columns, f.n), &out);
  }
  return out;
}

ErrorMetrics error_metrics(
    const std::vector<float>& candidate,
    const std::vector<float>& reference) {
  if (candidate.size() != reference.size() || candidate.empty()) {
    throw std::invalid_argument(
        "metric arrays must be equally sized and non-empty");
  }
  ErrorMetrics metrics;
  double squared_error = 0.0;
  double dot = 0.0;
  double candidate_norm = 0.0;
  double reference_norm = 0.0;
  for (size_t i = 0; i < candidate.size(); ++i) {
    const double got = candidate[i];
    const double expected = reference[i];
    const double abs_error = std::fabs(got - expected);
    metrics.max_abs = std::max(metrics.max_abs, abs_error);
    metrics.reference_max_abs =
        std::max(metrics.reference_max_abs, std::fabs(expected));
    squared_error += abs_error * abs_error;
    dot += got * expected;
    candidate_norm += got * got;
    reference_norm += expected * expected;
  }
  const double rel_floor = std::max(1e-6, metrics.reference_max_abs * 1e-6);
  for (size_t i = 0; i < candidate.size(); ++i) {
    metrics.max_rel = std::max(
        metrics.max_rel,
        std::fabs(static_cast<double>(candidate[i]) - reference[i]) /
            std::max(rel_floor, std::fabs(static_cast<double>(reference[i]))));
  }
  metrics.normalized_max_abs =
      metrics.max_abs / std::max(1.0, metrics.reference_max_abs);
  metrics.rmse =
      std::sqrt(squared_error / static_cast<double>(candidate.size()));
  if (candidate_norm > 0.0 && reference_norm > 0.0) {
    metrics.cosine = dot / std::sqrt(candidate_norm * reference_norm);
  }
  return metrics;
}

template <typename Function>
double median_ms(int iterations, Function&& function) {
  if (iterations <= 0) {
    return 0.0;
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

Fixture random_fixture(int m, int n, int k, int group_size, uint32_t seed) {
  Fixture f{m, n, k, group_size};
  const int groups = groups_per_row(f);
  std::mt19937 rng(seed);
  std::normal_distribution<float> activation(0.0f, 0.5f);
  std::uniform_int_distribution<int> weight(0, 255);
  std::uniform_real_distribution<float> scale(0.001f, 0.03f);
  std::uniform_real_distribution<float> bias(-0.02f, 0.02f);
  f.x.resize(static_cast<size_t>(m) * k);
  f.weights.resize(static_cast<size_t>(n) * k);
  f.scales.resize(static_cast<size_t>(n) * groups);
  f.biases.resize(static_cast<size_t>(n) * groups);
  for (float& value : f.x)
    value = activation(rng);
  for (uint8_t& value : f.weights)
    value = static_cast<uint8_t>(weight(rng));
  for (float& value : f.scales)
    value = scale(rng);
  for (float& value : f.biases)
    value = bias(rng);
  return f;
}

void write_output(const std::string& path, const std::vector<float>& output) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    throw std::runtime_error("could not open output file: " + path);
  file.write(
      reinterpret_cast<const char*>(output.data()),
      static_cast<std::streamsize>(output.size() * sizeof(float)));
  if (!file)
    throw std::runtime_error("could not write output file: " + path);
}

Fixture read_fixture(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("could not open fixture: " + path);
  FixtureHeader header{};
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!file || header.magic != kFixtureMagic ||
      header.version != kFixtureVersion) {
    throw std::runtime_error("fixture header is not an i8mm evidence fixture");
  }
  Fixture f{
      static_cast<int>(header.m),
      static_cast<int>(header.n),
      static_cast<int>(header.k),
      static_cast<int>(header.group_size)};
  const int groups = groups_per_row(f);
  f.x.resize(static_cast<size_t>(f.m) * f.k);
  f.weights.resize(static_cast<size_t>(f.n) * f.k);
  f.scales.resize(static_cast<size_t>(f.n) * groups);
  f.biases.resize(static_cast<size_t>(f.n) * groups);
  auto read = [&](auto* data, size_t count) {
    file.read(
        reinterpret_cast<char*>(data),
        static_cast<std::streamsize>(count * sizeof(*data)));
    if (!file)
      throw std::runtime_error("fixture ended before all tensors were read");
  };
  read(f.x.data(), f.x.size());
  read(f.weights.data(), f.weights.size());
  read(f.scales.data(), f.scales.size());
  read(f.biases.data(), f.biases.size());
  return f;
}

void require_i8mm_matches_scalar(const Fixture& f, int chunk_columns) {
  const auto prepared = quantize_activations_once(f);
  const int groups = groups_per_row(f);
  for (int m = 0; m < f.m; m += 2) {
    for (int n = 0; n < f.n; n += 2) {
      for (int group = 0; group < groups; ++group) {
        const size_t offset = static_cast<size_t>(group) * f.group_size;
        const int8_t* a0 =
            prepared.values.data() + static_cast<size_t>(m) * f.k + offset;
        const int8_t* a1 = m + 1 < f.m
            ? prepared.values.data() + static_cast<size_t>(m + 1) * f.k + offset
            : nullptr;
        const uint8_t* b0 =
            f.weights.data() + static_cast<size_t>(n) * f.k + offset;
        const uint8_t* b1 = n + 1 < f.n
            ? f.weights.data() + static_cast<size_t>(n + 1) * f.k + offset
            : nullptr;
        const auto lanes = i8mm_dot_two_by_two(a0, a1, b0, b1, f.group_size);
        auto dot = [&](const int8_t* a, const uint8_t* b) {
          if (!a || !b)
            return int32_t{0};
          int32_t value = 0;
          for (int e = 0; e < f.group_size; ++e) {
            value += static_cast<int32_t>(a[e]) * static_cast<int32_t>(b[e]);
          }
          return value;
        };
        if (lanes[0] != dot(a0, b0) || lanes[1] != dot(a1, b0) ||
            lanes[2] != dot(a0, b1) || lanes[3] != dot(a1, b1)) {
          throw std::runtime_error(
              "I8MM two-by-two integer tile mapping failed");
        }
        if (dotprod_u8_i8(b0, a0, f.group_size) != dot(a0, b0) ||
            (a1 && dotprod_u8_i8(b0, a1, f.group_size) != dot(a1, b0)) ||
            (b1 && dotprod_u8_i8(b1, a0, f.group_size) != dot(a0, b1)) ||
            (a1 && b1 && dotprod_u8_i8(b1, a1, f.group_size) != dot(a1, b1))) {
          throw std::runtime_error("U8xI8 dot-product control mapping failed");
        }
      }
    }
  }
  const auto scalar = scalar_prequantized(f, prepared);
  const auto dot_control = dotprod_prequantized(f, prepared, chunk_columns);
  const auto i8mm = i8mm_prequantized(f, prepared, chunk_columns);
  const auto dot_error = error_metrics(dot_control, scalar);
  const auto error = error_metrics(i8mm, scalar);
  // The integer tile is checked exactly above. This looser bound is only for
  // fp32 group accumulation order in the final affine correction.
  if (error.max_abs > 2e-3 || error.rmse > 5e-4) {
    throw std::runtime_error(
        "I8MM did not match the scalar prequantized oracle: max_abs=" +
        std::to_string(error.max_abs) + ", rmse=" + std::to_string(error.rmse));
  }
  if (dot_error.max_abs > 2e-3 || dot_error.rmse > 5e-4) {
    throw std::runtime_error(
        "dot-product control did not match the scalar prequantized oracle");
  }
  const auto method_error = error_metrics(i8mm, dot_control);
  if (method_error.max_abs > 2e-3 || method_error.rmse > 5e-4 ||
      method_error.cosine < 0.999999) {
    throw std::runtime_error(
        "I8MM and dot-product controls did not preserve the same approximation");
  }
  const auto repeated = i8mm_requantized_per_chunk(f, chunk_columns);
  const auto reuse_error = error_metrics(repeated, i8mm);
  if (reuse_error.max_abs > 1e-4) {
    throw std::runtime_error(
        "chunked activation reuse changed candidate output");
  }
}

void self_test() {
  if (!i8mm_available()) {
    throw std::runtime_error("I8MM is not exposed to this process");
  }
  // Odd M and N are true output tails. K tails are invalid for MLX group-64
  // affine QMM and are checked as rejection rather than silently padded.
  for (const auto& shape : std::array<std::array<int, 3>, 4>{
           {{{1, 1, 64}}, {{2, 3, 64}}, {{3, 5, 128}}, {{7, 17, 64}}}}) {
    require_i8mm_matches_scalar(
        random_fixture(shape[0], shape[1], shape[2], 64, 17 + shape[0]), 5);
  }

  auto zeros = random_fixture(3, 5, 64, 64, 29);
  std::fill(zeros.x.begin(), zeros.x.end(), 0.0f);
  require_i8mm_matches_scalar(zeros, 3);
  const auto zero_out =
      i8mm_prequantized(zeros, quantize_activations_once(zeros), 3);
  for (float value : zero_out) {
    if (value != 0.0f)
      throw std::runtime_error("zero activation did not remain zero");
  }

  auto saturated = random_fixture(3, 5, 64, 64, 31);
  for (size_t i = 0; i < saturated.x.size(); ++i) {
    saturated.x[i] = (i % 4 == 0) ? -1000.0f
        : (i % 4 == 1)            ? 1000.0f
                                  : 0.03125f;
  }
  require_i8mm_matches_scalar(saturated, 3);
  const auto prepared = quantize_activations_once(saturated);
  for (int8_t value : prepared.values) {
    if (value < -127 || value > 127) {
      throw std::runtime_error(
          "activation quantizer exceeded symmetric int8 range");
    }
  }

  auto nonfinite = random_fixture(1, 1, 64, 64, 37);
  nonfinite.x[0] = std::numeric_limits<float>::quiet_NaN();
  bool rejected = false;
  try {
    (void)quantize_activations_once(nonfinite);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("NaN activation was not rejected");

  nonfinite.x[0] = std::numeric_limits<float>::infinity();
  rejected = false;
  try {
    (void)quantize_activations_once(nonfinite);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("infinite activation was not rejected");

  bool invalid_shape_rejected = false;
  try {
    (void)random_fixture(1, 1, 65, 64, 41);
  } catch (const std::invalid_argument&) {
    invalid_shape_rejected = true;
  }
  if (!invalid_shape_rejected) {
    throw std::runtime_error("invalid K tail was not rejected");
  }
  std::puts(
      "PASS: affine-QMM controls (dot/I8MM, tails, reuse, metrics, edge cases)");
}

struct Options {
  bool self_test = false;
  bool identity = false;
  bool correctness = false;
  bool timing_only = false;
  std::string input;
  std::string dot_output;
  std::string i8mm_output;
  int bench_iterations = 0;
  int bench_warmups = 0;
  int trial_index = 0;
  int chunk_columns = 64;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--self-test") {
      options.self_test = true;
    } else if (arg == "--identity") {
      options.identity = true;
    } else if (arg == "--correctness") {
      options.correctness = true;
    } else if (arg == "--timing-only") {
      options.timing_only = true;
    } else if (
        arg == "--input" || arg == "--dot-output" || arg == "--i8mm-output" ||
        arg == "--bench-iters" || arg == "--bench-warmups" ||
        arg == "--trial-index" || arg == "--chunk-columns") {
      if (++i == argc)
        throw std::invalid_argument("missing value after " + arg);
      const std::string value = argv[i];
      if (arg == "--input")
        options.input = value;
      if (arg == "--dot-output")
        options.dot_output = value;
      if (arg == "--i8mm-output")
        options.i8mm_output = value;
      if (arg == "--bench-iters")
        options.bench_iterations = std::stoi(value);
      if (arg == "--bench-warmups")
        options.bench_warmups = std::stoi(value);
      if (arg == "--trial-index")
        options.trial_index = std::stoi(value);
      if (arg == "--chunk-columns")
        options.chunk_columns = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  const int mode_count = static_cast<int>(options.self_test) +
      static_cast<int>(options.identity) +
      static_cast<int>(options.correctness) +
      static_cast<int>(options.timing_only);
  if (mode_count != 1) {
    throw std::invalid_argument(
        "use exactly one of --self-test, --identity, --correctness, or --timing-only");
  }
  if (options.correctness &&
      (options.input.empty() || options.dot_output.empty() ||
       options.i8mm_output.empty())) {
    throw std::invalid_argument(
        "--correctness requires --input, --dot-output, and --i8mm-output");
  }
  if (options.timing_only && options.input.empty()) {
    throw std::invalid_argument("--timing-only requires --input");
  }
  if (options.chunk_columns <= 0 || options.bench_iterations < 0 ||
      options.bench_warmups < 0 || options.trial_index < 0) {
    throw std::invalid_argument("benchmark arguments must be non-negative");
  }
  if (options.timing_only &&
      (options.bench_iterations < 1 || options.bench_warmups < 1)) {
    throw std::invalid_argument(
        "--timing-only requires at least one benchmark iteration and warmup");
  }
  return options;
}

void print_identity() {
  std::printf(
      "{\"identity_version\":1,\"candidate_source_sha256\":\"%s\"}\n",
      MLX_I8MM_EVIDENCE_SOURCE_SHA256);
}

volatile float benchmark_sink = 0.0f;

template <typename Function>
double time_once(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TimingSamples {
  std::vector<double> preparation;
  std::vector<double> dot_control;
  std::vector<double> i8mm;
  std::vector<double> dot_end_to_end;
  std::vector<double> i8mm_end_to_end;
  std::vector<double> repeated_preparation;
};

double median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run_candidate_pair(
    const Fixture& fixture,
    const PreparedActivations& prepared,
    int chunk_columns,
    bool i8mm_first,
    double* dot_ms,
    double* i8mm_ms) {
  auto dot = [&] {
    const auto value = dotprod_prequantized(fixture, prepared, chunk_columns);
    benchmark_sink += value.front();
  };
  auto i8mm = [&] {
    const auto value = i8mm_prequantized(fixture, prepared, chunk_columns);
    benchmark_sink += value.front();
  };
  if (i8mm_first) {
    *i8mm_ms = time_once(i8mm);
    *dot_ms = time_once(dot);
  } else {
    *dot_ms = time_once(dot);
    *i8mm_ms = time_once(i8mm);
  }
}

void run_end_to_end_pair(
    const Fixture& fixture,
    int chunk_columns,
    bool i8mm_first,
    double* dot_ms,
    double* i8mm_ms) {
  auto dot = [&] {
    const auto prepared = quantize_activations_once(fixture);
    const auto value = dotprod_prequantized(fixture, prepared, chunk_columns);
    benchmark_sink += value.back();
  };
  auto i8mm = [&] {
    const auto prepared = quantize_activations_once(fixture);
    const auto value = i8mm_prequantized(fixture, prepared, chunk_columns);
    benchmark_sink += value.back();
  };
  if (i8mm_first) {
    *i8mm_ms = time_once(i8mm);
    *dot_ms = time_once(dot);
  } else {
    *dot_ms = time_once(dot);
    *i8mm_ms = time_once(i8mm);
  }
}

void run_fixture(const Options& options) {
  if (!i8mm_available())
    throw std::runtime_error("I8MM is not exposed to this process");
  const Fixture fixture = read_fixture(options.input);
  const auto prepared = quantize_activations_once(fixture);
  if (options.correctness) {
    const auto dot_candidate =
        dotprod_prequantized(fixture, prepared, options.chunk_columns);
    const auto i8mm_candidate =
        i8mm_prequantized(fixture, prepared, options.chunk_columns);
    write_output(options.dot_output, dot_candidate);
    write_output(options.i8mm_output, i8mm_candidate);
    const auto reference = reference_affine(fixture);
    const auto dot_quality = error_metrics(dot_candidate, reference);
    const auto i8mm_quality = error_metrics(i8mm_candidate, reference);
    const auto method_delta = error_metrics(i8mm_candidate, dot_candidate);
    std::printf(
        "{\"mode\":\"correctness\",\"candidate_source_sha256\":\"%s\","
        "\"fixture\":{\"M\":%d,\"N\":%d,\"K\":%d,\"group_size\":%d},"
        "\"dot_control_vs_affine_reference\":{\"max_abs\":%.9g,"
        "\"max_relative\":%.9g,\"normalized_max_abs\":%.9g,"
        "\"rmse\":%.9g,\"cosine\":%.12g},"
        "\"i8mm_2x2_vs_affine_reference\":{\"max_abs\":%.9g,"
        "\"max_relative\":%.9g,\"normalized_max_abs\":%.9g,"
        "\"rmse\":%.9g,\"cosine\":%.12g},"
        "\"i8mm_2x2_vs_dot_control\":{\"max_abs\":%.9g,"
        "\"max_relative\":%.9g,\"normalized_max_abs\":%.9g,"
        "\"rmse\":%.9g,\"cosine\":%.12g}}\n",
        MLX_I8MM_EVIDENCE_SOURCE_SHA256,
        fixture.m,
        fixture.n,
        fixture.k,
        fixture.group_size,
        dot_quality.max_abs,
        dot_quality.max_rel,
        dot_quality.normalized_max_abs,
        dot_quality.rmse,
        dot_quality.cosine,
        i8mm_quality.max_abs,
        i8mm_quality.max_rel,
        i8mm_quality.normalized_max_abs,
        i8mm_quality.rmse,
        i8mm_quality.cosine,
        method_delta.max_abs,
        method_delta.max_rel,
        method_delta.normalized_max_abs,
        method_delta.rmse,
        method_delta.cosine);
    return;
  }

  for (int warmup = 0; warmup < options.bench_warmups; ++warmup) {
    double ignored_dot = 0.0;
    double ignored_i8mm = 0.0;
    const bool i8mm_first = (options.trial_index + warmup) % 2 != 0;
    run_candidate_pair(
        fixture,
        prepared,
        options.chunk_columns,
        i8mm_first,
        &ignored_dot,
        &ignored_i8mm);
    run_end_to_end_pair(
        fixture,
        options.chunk_columns,
        !i8mm_first,
        &ignored_dot,
        &ignored_i8mm);
  }

  TimingSamples samples;
  for (int iteration = 0; iteration < options.bench_iterations; ++iteration) {
    samples.preparation.push_back(time_once([&] {
      const auto timed = quantize_activations_once(fixture);
      benchmark_sink += timed.scales.front();
    }));
    const bool i8mm_first = (options.trial_index + iteration) % 2 != 0;
    double dot_ms = 0.0;
    double i8mm_ms = 0.0;
    run_candidate_pair(
        fixture,
        prepared,
        options.chunk_columns,
        i8mm_first,
        &dot_ms,
        &i8mm_ms);
    samples.dot_control.push_back(dot_ms);
    samples.i8mm.push_back(i8mm_ms);
    run_end_to_end_pair(
        fixture, options.chunk_columns, !i8mm_first, &dot_ms, &i8mm_ms);
    samples.dot_end_to_end.push_back(dot_ms);
    samples.i8mm_end_to_end.push_back(i8mm_ms);
    samples.repeated_preparation.push_back(time_once([&] {
      const auto timed =
          i8mm_requantized_per_chunk(fixture, options.chunk_columns);
      benchmark_sink += timed.back();
    }));
  }
  const int chunks =
      (fixture.n + options.chunk_columns - 1) / options.chunk_columns;

  // Timing-only mode intentionally does not materialize correctness outputs,
  // compute a scalar reference, or compute numerical metrics before timing.
  std::printf(
      "{\"mode\":\"timing_only\",\"candidate_source_sha256\":\"%s\","
      "\"fixture\":{\"M\":%d,\"N\":%d,\"K\":%d,\"group_size\":%d},"
      "\"timing_order\":\"alternating_dot_control_and_i8mm\","
      "\"bench_warmups\":%d,\"bench_iterations\":%d,\"trial_index\":%d,"
      "\"activation_preparation_ms\":%.9f,\"dot_control_ms\":%.9f,"
      "\"i8mm_2x2_ms\":%.9f,\"dot_control_end_to_end_ms\":%.9f,"
      "\"i8mm_2x2_end_to_end_ms\":%.9f,"
      "\"threaded_m_gt_1_repeated_preparation_control_ms\":%.9f,"
      "\"reused_activation_passes\":%d,\"repeated_activation_passes\":%d}\n",
      MLX_I8MM_EVIDENCE_SOURCE_SHA256,
      fixture.m,
      fixture.n,
      fixture.k,
      fixture.group_size,
      options.bench_warmups,
      options.bench_iterations,
      options.trial_index,
      median(samples.preparation),
      median(samples.dot_control),
      median(samples.i8mm),
      median(samples.dot_end_to_end),
      median(samples.i8mm_end_to_end),
      median(samples.repeated_preparation),
      fixture.m,
      fixture.m * chunks);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.self_test) {
      self_test();
    } else if (options.identity) {
      print_identity();
    } else {
      run_fixture(options);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "quantized_i8mm_evidence: %s\n", error.what());
    return 1;
  }
}
