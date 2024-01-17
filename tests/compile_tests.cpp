// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

std::vector<array> simple_fun(const std::vector<array>& inputs) {
  return std::vector<array>{inputs[0] + inputs[1]};
}

TEST_CASE("test simple compile") {
  auto compfn = compile(simple_fun);
  auto out = compfn({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);

  out = compfn({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);

  // Change the shapes
  out = compfn({array({1.0f, 2.0f}), array(2.0f)})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  out = compfn({array(2.0f), array({1.0f, 2.0f})})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  // Change the types
  out = compfn({array(2, int32), array({1.0f, 2.0f})})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  out = compfn({array(2.0f), array({1, 2}, int32)})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());
}

std::vector<array> grad_fun(const std::vector<array>& inputs) {
  auto loss = [](std::vector<array> ins) { return exp(ins[0] + ins[1]); };
  return grad(loss, {0, 1})(inputs);
}

TEST_CASE("test compile with grad") {
  auto x = array(1.0f);
  auto y = array(1.0f);
  auto grads_expected = grad_fun({x, y});
  auto grads_compile = compile(grad_fun)({x, y});
  CHECK_EQ(grads_compile[0].item<float>(), grads_expected[0].item<float>());
  CHECK_EQ(grads_compile[1].item<float>(), grads_expected[1].item<float>());
}

TEST_CASE("test compile inputs with primitive") {
  auto [k1, k2] = random::split(random::key(0));
  auto x = random::uniform({5, 5}, k1);
  auto y = random::uniform({5, 5}, k2);
  auto expected = simple_fun({x, y})[0];

  x = random::uniform({5, 5}, k1);
  y = random::uniform({5, 5}, k2);
  auto out = compile(simple_fun)({x, y})[0];
  CHECK(array_equal(expected, out).item<bool>());

  // Same thing twice
  out = compile(simple_fun)({x, y})[0];
  CHECK(array_equal(expected, out).item<bool>());
}

std::vector<array> fun_creats_array(const std::vector<array>& inputs) {
  return {inputs[0] + array(1.0f)};
}

TEST_CASE("test compile with created array") {
  auto cfun = compile(fun_creats_array);
  auto out = cfun({array(2.0f)});
  CHECK_EQ(out[0].item<float>(), 3.0f);

  // Try again
  out = cfun({array(2.0f)});
  CHECK_EQ(out[0].item<float>(), 3.0f);
}

std::vector<array> inner_fun(const std::vector<array>& inputs) {
  return {array(2) * inputs[0]};
}

std::vector<array> outer_fun(const std::vector<array>& inputs) {
  auto x = inputs[0] + inputs[1];
  auto y = compile(inner_fun)({x})[0];
  return {x + y};
}

TEST_CASE("test nested compile") {
  auto cfun = compile(outer_fun);
  auto out = cfun({array(1), array(2)})[0];
  CHECK_EQ(out.item<int>(), 9);

  // Try again
  out = cfun({array(1), array(2)})[0];
  CHECK_EQ(out.item<int>(), 9);
}

TEST_CASE("test enable and disable compile") {
  CHECK_THROWS(compile(nullptr));
  disable_compiler();
  compile(nullptr);
  enable_compiler();
  CHECK_THROWS(compile(nullptr));
}
