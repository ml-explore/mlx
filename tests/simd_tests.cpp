#include "doctest/doctest.h"

#include "mlx/backend/cpu/simd/simd.h"

using namespace mlx::core;

#if defined(MLX_USE_ACCELERATE) && defined(__aarch64__) && \
    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("test float16 SIMD comparison masks") {
  using namespace mlx::core::simd;
  constexpr int size = 8;
  const float16_t values[] = {-2, 0, 1, -0.0f, 4, -1, 2, -4};
  const float16_t other[] = {0, 1, 1, -1, 0, -2, 3, -4};
  for (int offset = 0; offset < size; ++offset) {
    float16_t input[size];
    for (int lane = 0; lane < size; ++lane) {
      input[lane] = values[(lane + offset) % size];
    }
    auto x = load<float16_t, size>(input);
    auto y = load<float16_t, size>(other);
    auto inverted = !x;
    const Simd<bool, size> masks[] = {
        x == y,
        x != y,
        (x < y),
        x <= y,
        (x > y),
        x >= y,
        x == 0,
        x != 0,
        x < 0,
        x <= 0,
        x > 0,
        x >= 0,
        0 == x,
        0 != x,
        0 < x,
        0 <= x,
        0 > x,
        0 >= x};
    for (int lane = 0; lane < size; ++lane) {
      CAPTURE(offset);
      CAPTURE(lane);
      auto a = static_cast<float>(input[lane]);
      auto b = static_cast<float>(other[lane]);
      const bool expected[] = {
          a == b,
          a != b,
          (a < b),
          a <= b,
          (a > b),
          a >= b,
          a == 0,
          a != 0,
          a < 0,
          a <= 0,
          a > 0,
          a >= 0,
          0 == a,
          0 != a,
          0 < a,
          0 <= a,
          0 > a,
          0 >= a};
      CHECK_EQ(inverted.value[lane] != 0, a == 0);
      for (int op = 0; op < 18; ++op) {
        CAPTURE(op);
        CHECK_EQ(masks[op].value[lane] != 0, expected[op]);
      }
    }
  }
}
#endif
