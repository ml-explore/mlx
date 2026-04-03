#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace mx = mlx::core;
namespace fast = mlx::core::fast;

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

static double bench_once(const std::function<mx::array()>& fn, int iters = 20, int warmup = 5) {
  for (int i = 0; i < warmup; ++i) {
    auto y = fn();
    mx::eval(y);
  }

  double t0 = now_ms();
  for (int i = 0; i < iters; ++i) {
    auto y = fn();
    mx::eval(y);
  }
  double t1 = now_ms();

  return (t1 - t0) / iters;
}

static void run_case(int B, int H, int S, int D, float scale) {
  auto q_nope = mx::random::normal({B, H, D}, mx::float16);
  auto latent = mx::random::normal({B, S, D}, mx::float16);

  auto quant_result = mx::quantize(latent, 64, 4);
  auto k_packed = quant_result[0];
  auto k_scales = quant_result[1];
  auto k_biases = quant_result[2];

  // Candidate: our MLA primitive
  auto mla_fn = [&]() {
    return fast::mla_nope_scores(q_nope, k_packed, k_scales, k_biases, scale);
  };

  // Baseline: dequantize + matmul
  auto ref_fn = [&]() {
    auto latent_deq = mx::dequantize(k_packed, k_scales, k_biases, 64, 4);
    auto qf = mx::astype(q_nope, mx::float32);
    auto kf = mx::transpose(mx::astype(latent_deq, mx::float32), {0, 2, 1});
    return mx::multiply(mx::matmul(qf, kf), mx::array(scale));
  };

  // Correctness check
  auto out_mla = mla_fn();
  auto out_ref = ref_fn();
  mx::eval(out_mla);
  mx::eval(out_ref);

  auto diff = mx::abs(mx::astype(out_mla, mx::float32) - out_ref);
  float max_abs = mx::max(diff).item<float>();

  // Benchmark
  double mla_ms = bench_once(mla_fn);
  double ref_ms = bench_once(ref_fn);
  double speedup = ref_ms / mla_ms;

  std::cout
      << "B=" << B
      << " H=" << H
      << " S=" << std::setw(6) << S
      << " D=" << D
      << " | mla=" << std::fixed << std::setprecision(3) << mla_ms << "ms"
      << " | ref=" << ref_ms << "ms"
      << " | speedup=" << speedup << "x"
      << " | max_abs=" << std::scientific << max_abs
      << "\n";
}

int main() {
  constexpr int B = 1;
  constexpr int H = 32;
  constexpr int D = 256;
  constexpr float scale = 0.125f;

  std::vector<int> seqs = {1, 8, 16, 64, 256, 1024, 8192, 32768};

  std::cout << "=== MLA Nope Scores Benchmark ===\n";
  std::cout << "Candidate: fast::mla_nope_scores\n";
  std::cout << "Baseline : dequantize + matmul\n\n";

  for (int S : seqs) {
    run_case(B, H, S, D, scale);
  }

  return 0;
}
