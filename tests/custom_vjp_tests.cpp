// Copyright © 2023-2024 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test simple custom vjp") {
  auto one = array(1.0);
  auto x = array(2.0);
  auto y = array(3.0);

  auto fn = [](const std::vector<array>& inputs) {
    return std::vector<array>{inputs[0] * inputs[1], inputs[0] + inputs[1]};
  };
  auto transformed_fn = custom_vjp(
      fn,
      [&](const std::vector<array>&,
          const std::vector<array>&,
          const std::vector<array>&) { return std::vector<array>{one, one}; });

  auto [z, g] = vjp(fn, {x, y}, {one, one});
  CHECK_EQ(z[0].item<float>(), 6.0f);
  CHECK_EQ(z[1].item<float>(), 5.0f);
  CHECK_EQ(g[0].item<float>(), 4.0f);
  CHECK_EQ(g[1].item<float>(), 3.0f);

  std::tie(z, g) = vjp(transformed_fn, {x, y}, {one, one});
  CHECK_EQ(z[0].item<float>(), 6.0f);
  CHECK_EQ(z[1].item<float>(), 5.0f);
  CHECK_EQ(g[0].item<float>(), 1.0f);
  CHECK_EQ(g[1].item<float>(), 1.0f);
}

TEST_CASE("test checkpointing") {
  auto one = array(1.0);
  auto x = array(2.0);
  auto y = array(3.0);

  int cnt = 0;
  auto fn = [&cnt](const std::vector<array>& inputs) {
    cnt++;
    auto x = inputs[0] * inputs[1];
    auto y = inputs[0] + inputs[1];
    return std::vector<array>{square(x + y)};
  };
  auto checkpointed_fn = checkpoint(fn);

  auto [z, g] = vjp(checkpointed_fn, {x, y}, {one});
  CHECK_EQ(z[0].item<float>(), 121.0f);
  CHECK_EQ(g[0].item<float>(), 88.0f);
  CHECK_EQ(g[1].item<float>(), 66.0f);
  CHECK_EQ(cnt, 2);
}

class NonDifferentiableOp : public Primitive {
 public:
  explicit NonDifferentiableOp(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    outputs[0].copy_shared_buffer(inputs[0]);
  }
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {}
  DEFINE_NAME(NonDifferentiableOp);
};

TEST_CASE("test custom transforms with non-differentiable primitive") {
  auto fn = [](const std::vector<array>& inputs) {
    auto out = array(
        inputs[0].shape(),
        inputs[0].dtype(),
        std::make_shared<NonDifferentiableOp>(default_stream(default_device())),
        {inputs[0]});
    return std::vector<array>{out};
  };

  auto transformed_fn = custom_function(
      fn,
      /* vjp */
      [](const std::vector<array>&,
         const std::vector<array>& cotans,
         const std::vector<array>&) {
        return std::vector<array>{cotans[0] * 2.0f};
      },
      /* jvp */
      [](const std::vector<array>&,
         const std::vector<array>& tangents,
         const std::vector<int>&) {
        return std::vector<array>{tangents[0] * 2.0f};
      },
      /* vmap */
      [](const std::vector<array>& inputs, const std::vector<int>& in_axes) {
        return std::make_pair(std::vector<array>{inputs[0] * 2.0f}, in_axes);
      });

  auto x = array(3.0f);

  // VJP works
  auto [z, g] = vjp(transformed_fn, {x}, {array(1.0f)});
  CHECK_EQ(z[0].item<float>(), 3.0f);
  CHECK_EQ(g[0].item<float>(), 2.0f);

  // JVP
  auto [out_jvp, tangents] = jvp(transformed_fn, {x}, {array(1.0f)});
  CHECK_EQ(out_jvp[0].item<float>(), 3.0f);
  CHECK_EQ(tangents[0].item<float>(), 2.0f);

  // VMAP
  auto vmap_fn = vmap(transformed_fn, {0}, {0});
  auto v_out = vmap_fn({array({1.0f, 2.0f})});
  CHECK_EQ(v_out[0].shape(), Shape{2});
  CHECK(array_equal(v_out[0], array({2.0f, 4.0f})).item<bool>());
}
