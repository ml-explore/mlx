// Copyright © 2023-2024 Apple Inc.

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
  disable_compile();
  compile(nullptr);
  enable_compile();
  CHECK_THROWS(compile(nullptr));
}

auto add_scalars(const std::vector<array>&) {
  auto a = array(-1.0f);
  auto b = array(-1.0f);
  return std::vector<array>{abs(a), abs(b)};
};

auto max_scalars(const std::vector<array>&) {
  auto a = array({-1.0f, 2.0f});
  auto b = maximum(a, array(0.0f));
  auto c = maximum(-a, array(0.0f));
  auto d = b + c;
  return std::vector<array>{b, c, d};
};

TEST_CASE("test simplify scalars") {
  {
    auto cfun = compile(add_scalars);
    auto out = cfun({});
    auto c = out[0];
    auto d = out[1];
    CHECK(c.inputs()[0].id() == d.inputs()[0].id());
  }

  {
    auto a = array({-1.0f, 2.0f});
    auto out = compile(max_scalars)({a});
    auto b = out[0];
    auto c = out[1];
    auto d = out[2];
    CHECK(b.inputs()[1].id() == c.inputs()[1].id());
  }
}

auto exp_two(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{exp(a) + exp(a)};
};

TEST_CASE("test simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = compile(exp_two)({a})[0];
  CHECK(b.inputs()[0].id() == b.inputs()[1].id());
}

auto add_diff(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{cos(a) + sin(a)};
};

TEST_CASE("test no simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = compile(add_diff)({a})[0];
  CHECK(b.inputs()[0].id() != b.inputs()[1].id());
}

auto multi_one(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(2.0);
  auto c = divmod(a, b);
  auto d = divmod(a, b);
  auto e = c[0] + d[0];
  auto f = c[1] + d[1];
  return std::vector<array>{e, f};
}

auto multi_two(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(1.0);
  auto c = divmod(a, b);
  return std::vector<array>{c};
}

auto multi_three(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(2.0);
  auto c = divmod(a, b);
  auto d = divmod(a, b);
  auto e = stack({c[0], c[1], d[0], d[1]});
  return std::vector<array>{e};
}

TEST_CASE("test simplify multi output") {
  {
    auto out = compile(multi_one)({});
    auto e = out[0];
    auto f = out[1];
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[1].id());
    CHECK_EQ(f.inputs()[0].id(), f.inputs()[1].id());
  }

  {
    auto c = compile(multi_two)({});
    CHECK_EQ(c[0].inputs()[0].id(), c[0].inputs()[1].id());
    CHECK_EQ(c[0].inputs()[0].id(), c[1].inputs()[0].id());
    CHECK_EQ(c[1].inputs()[0].id(), c[1].inputs()[1].id());
  }

  // Make sure the output order of multi-output primitives
  // is respected in simplification
  {
    auto e = compile(multi_three)({})[0];
    CHECK(array_equal(e, array({0.0f, 1.0f, 0.0f, 1.0f})).item<bool>());
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[2].id());
    CHECK_EQ(e.inputs()[1].id(), e.inputs()[3].id());
  }
}
