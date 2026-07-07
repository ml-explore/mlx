// Copyright © 2023-2024 Apple Inc.

// Required for using M_SQRT2 in MSVC.
#define _USE_MATH_DEFINES

#include "doctest/doctest.h"

#include <cmath>
#include <limits>

#include "mlx/compile_impl.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

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
  CHECK(allclose(grads_compile[0], grads_expected[0]).item<bool>());
  CHECK(allclose(grads_compile[1], grads_expected[1]).item<bool>());
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

std::vector<array> nan_const_fun(const std::vector<array>& inputs) {
  auto nan = array(std::numeric_limits<float>::quiet_NaN());
  return {where(greater(inputs[0], array(0.0f)), inputs[0], nan)};
}

std::vector<array> neg_inf_const_fun(const std::vector<array>& inputs) {
  auto inf = array(-std::numeric_limits<float>::infinity());
  return {where(greater(inputs[0], array(0.0f)), inputs[0], inf)};
}

TEST_CASE("test compile with non-finite constants") {
  // Regression test: baking a non-finite scalar constant (NaN / infinity) into
  // a fused compiled kernel used to stream a bare token (e.g. `nan`) into the
  // generated kernel source, which is not a valid identifier and broke
  // compilation (notably on the Metal backend).
  auto out = compile(nan_const_fun)({array({1.0f, -1.0f})})[0];
  eval(out);
  CHECK_EQ(out.shape(), Shape{2});
  CHECK_EQ(out.data<float>()[0], 1.0f);
  CHECK(std::isnan(out.data<float>()[1]));

  out = compile(neg_inf_const_fun)({array({1.0f, -1.0f})})[0];
  eval(out);
  CHECK_EQ(out.data<float>()[0], 1.0f);
  CHECK(std::isinf(out.data<float>()[1]));
  CHECK_LT(out.data<float>()[1], 0.0f);
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
  set_compile_mode(CompileMode::no_fuse);
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
  set_compile_mode(CompileMode::enabled);
}

auto exp_two(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{exp(a) + exp(a)};
};

TEST_CASE("test simplify") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto b = compile(exp_two)({a})[0];
  CHECK(b.inputs()[0].id() == b.inputs()[1].id());
  set_compile_mode(CompileMode::enabled);
}

TEST_CASE("test simplify noops") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto fun = [](const std::vector<array>& inputs) -> std::vector<array> {
    return {copy(stop_gradient(exp(stop_gradient(inputs[0]))))};
  };
  auto b = compile(fun)({a})[0];
  CHECK(b.inputs()[0].id() == a.id());
  set_compile_mode(CompileMode::enabled);
}

auto add_diff(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{cos(a) + sin(a)};
};

TEST_CASE("test no simplify") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto b = compile(add_diff)({a})[0];
  CHECK(b.inputs()[0].id() != b.inputs()[1].id());
  set_compile_mode(CompileMode::enabled);
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
  return divmod(a, b);
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
  set_compile_mode(CompileMode::no_fuse);
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
    CHECK_EQ(e.inputs().size(), 4);
    CHECK_EQ(e.inputs().at(0).id(), e.inputs().at(2).id());
    CHECK_EQ(e.inputs().at(1).id(), e.inputs().at(3).id());
    CHECK(array_equal(e, array({0.0f, 1.0f, 0.0f, 1.0f})).item<bool>());
  }
  set_compile_mode(CompileMode::enabled);
}

// No fusion
auto unary_fused_0(const std::vector<array>& inputs) {
  return std::vector<array>{exp(inputs[0])};
}

// All compilable
auto unary_fused_1(const std::vector<array>& inputs) {
  return std::vector<array>{abs(negative(exp(inputs[0])))};
}

auto unary_fused_1_copy(const std::vector<array>& inputs) {
  return std::vector<array>{abs(negative(exp(inputs[0])))};
}

auto unary_fused_1_diff(const std::vector<array>& inputs) {
  return std::vector<array>{abs(exp(negative(inputs[0])))};
}

// Output into un-compilable primitive
auto unary_fused_2(const std::vector<array>& inputs) {
  return std::vector<array>{sum(abs(negative(exp(inputs[0]))), true)};
}

// Input from un-compilable primitive
auto unary_fused_3(const std::vector<array>& inputs) {
  return std::vector<array>{exp(abs(negative(sum(inputs[0], true))))};
}

TEST_CASE("test compile unary fused") {
  // NB: some of these tests are brittle and may need to be
  // updated if we change compile conditions
  {
    auto cfun = compile(unary_fused_0);
    auto x = array(2.0);
    auto out = cfun({x})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Exp));
    CHECK_EQ(out.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(unary_fused_1);
    auto x = array(2.0);
    auto out = cfun({x})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());

    auto expected_out = unary_fused_1({array(2.0)})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }

  {
    auto cfun = compile(unary_fused_2);
    auto x = array({1.0, 2.0});
    auto out = cfun({x});
    CHECK_EQ(out.size(), 1);

    auto& p = out[0].primitive();
    // NB: this test is brittle, will need to update
    // it if we change compile conditions
    CHECK_EQ(typeid(p), typeid(Reduce));
    auto cout = out[0].inputs()[0];
    auto& cp = cout.primitive();
    CHECK_EQ(typeid(cp), typeid(Compiled));
    CHECK_EQ(cout.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(unary_fused_3);
    auto x = array({1.0, 2.0});
    auto out = cfun({x});

    auto& p = out[0].primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    auto sout = out[0].inputs()[0];
    CHECK_EQ(out[0].inputs().size(), 1);
    auto& sp = sout.primitive();
    CHECK_EQ(typeid(sp), typeid(Reduce));
    CHECK_EQ(sout.inputs()[0].id(), x.id());
  }

  // Is equivalent works
  {
    auto out1 = compile(unary_fused_1)({array(1.0)});
    auto out2 = compile(unary_fused_1_copy)({array(1.0)});
    CHECK(out1[0].primitive().is_equivalent(out2[0].primitive()));
    auto out3 = compile(unary_fused_1_diff)({array(1.0)});
    CHECK(!out1[0].primitive().is_equivalent(out3[0].primitive()));
  }
}

// All compilable
auto binary_fused_0(const std::vector<array>& inputs) {
  return std::vector<array>{inputs[0] + inputs[1]};
}

// Binary into unary
auto binary_fused_1(const std::vector<array>& inputs) {
  return std::vector<array>{abs(inputs[0] + inputs[1])};
}

// Binary into binary
auto binary_fused_2(const std::vector<array>& inputs) {
  auto x = inputs[0] + inputs[1];
  return std::vector<array>{x + inputs[0]};
}

// Binary into unary into un-compilable
auto binary_fused_3(const std::vector<array>& inputs) {
  return std::vector<array>{sum(abs(inputs[0] + inputs[1]), true)};
}

TEST_CASE("test compile binary fused") {
  {
    auto cfun = compile(binary_fused_0);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Add));
    CHECK_EQ(out.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(binary_fused_1);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());
    CHECK_EQ(out.inputs()[1].id(), y.id());

    auto expected_out = binary_fused_1({x, y})[0];
    CHECK_EQ(out.item<float>(), expected_out.item<float>());
  }

  {
    auto cfun = compile(binary_fused_2);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());
    CHECK_EQ(out.inputs()[1].id(), y.id());
  }

  {
    auto cfun = compile(binary_fused_3);
    auto x = array({1.0, 2.0});
    auto y = array({1.0, 2.0});
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Reduce));

    auto cout = out.inputs()[0];
    auto& cp = cout.primitive();
    CHECK_EQ(typeid(cp), typeid(Compiled));
    CHECK_EQ(cout.inputs()[0].id(), x.id());
    CHECK_EQ(cout.inputs()[1].id(), y.id());
  }
}

auto gelu_1(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto out = x * (1.0f + erf(x / M_SQRT2)) / 2.0f;
  return std::vector<array>{out};
}

TEST_CASE("test compile gelu") {
  {
    auto cfun = compile(gelu_1);
    auto x = array(1.0);
    auto out = cfun({x})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs().size(), 4);
    for (auto& in : out.inputs()) {
      CHECK(in.inputs().empty());
    }
    auto expected_out = gelu_1({x})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }

  {
    auto cfun = compile(gelu_1);
    auto x = array({1.0, 0.5});
    auto out = cfun({x})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs().size(), 4);
    for (auto& in : out.inputs()) {
      CHECK(in.inputs().empty());
    }

    auto expected_out = gelu_1({x})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }
}

// Uncompilable input outside fused tape
auto unary_with_two_outputs(const std::vector<array>& inputs) {
  auto x = exp(inputs[0]);
  return std::vector<array>{exp(x), sum(x, true)};
}

auto uncompilable_inputs(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  return std::vector<array>{x * abs(exp(y)), sum(x, true)};
}

auto uncompilable_inputs_order_matters(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  return std::vector<array>{x / abs(exp(y)), sum(x, true)};
}

TEST_CASE("test compile tape with outside parents") {
  {
    auto cfun = compile(unary_with_two_outputs);
    auto x = array({2.0, 2.0});
    auto out = cfun({x});

    auto& p1 = out[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Exp));
    auto& p2 = out[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
  }

  {
    auto cfun = compile(uncompilable_inputs);
    auto x = array({2.0, 2.0});
    auto y = array({1.6, 0.6});
    auto outs = cfun({x, y});

    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
    CHECK_EQ(outs[0].inputs().size(), 2);

    auto expected_outs = uncompilable_inputs({x, y});
    CHECK(allclose(outs[0], expected_outs[0]).item<bool>());
    CHECK(allclose(outs[1], expected_outs[1]).item<bool>());
  }

  {
    auto cfun = compile(uncompilable_inputs_order_matters);
    auto x = array({2.0, 2.0});
    auto y = array({1.6, 0.6});
    auto outs = cfun({x, y});

    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
    CHECK_EQ(outs[0].inputs().size(), 2);

    auto expected_outs = uncompilable_inputs_order_matters({x, y});
    CHECK(allclose(outs[0], expected_outs[0]).item<bool>());
    CHECK(allclose(outs[1], expected_outs[1]).item<bool>());
  }
}

auto compile_across_streams(const std::vector<array>& inputs) {
  auto s2 = new_stream(default_device());
  auto x = exp(abs(inputs[0]));
  auto y = exp(abs(x, s2), s2);
  return std::vector<array>{y};
}

TEST_CASE("test compile across streams") {
  auto cfun = compile(compile_across_streams);
  auto x = array({2.0f});
  auto out = cfun({x})[0];
  auto& p1 = out.primitive();
  CHECK_EQ(typeid(p1), typeid(Compiled));
  CHECK_EQ(out.inputs().size(), 1);
  auto child = out.inputs()[0];
  auto& p2 = child.primitive();
  CHECK_EQ(typeid(p2), typeid(Compiled));
  CHECK_EQ(child.inputs()[0].id(), x.id());
}

auto unary_compile_outputs(const std::vector<array>& inputs) {
  auto x = abs(inputs[0]);
  auto y = square(x);
  return std::vector<array>{x, y};
}

auto binary_compile_outputs(const std::vector<array>& inputs) {
  auto x = inputs[0];
  auto y = inputs[1];
  x = x + y;
  y = x + y;
  return std::vector<array>{x, y};
}

TEST_CASE("test compile internal output") {
  {
    auto cfun = compile(unary_compile_outputs);
    auto x = array({3, -2});
    auto outs = cfun({x});
    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Compiled));
    CHECK_EQ(outs[0].siblings()[0].id(), outs[1].id());
    auto expected_outs = unary_compile_outputs({x});
    CHECK(array_equal(outs[0], expected_outs[0]).item<bool>());
    CHECK(array_equal(outs[1], expected_outs[1]).item<bool>());
  }

  {
    auto cfun = compile(binary_compile_outputs);
    auto x = array({3, -2});
    auto y = array({1, -1});
    auto outs = cfun({x, y});
    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Compiled));
    auto expected_outs = binary_compile_outputs({x, y});
    CHECK(array_equal(outs[0], expected_outs[0]).item<bool>());
    CHECK(array_equal(outs[1], expected_outs[1]).item<bool>());
  }
}

auto deep_unary_compile(const std::vector<array>& inputs) {
  auto x = inputs[0];
  for (int i = 0; i < 10; ++i) {
    x = cos(sin(x));
  }
  return std::vector<array>{x};
}

TEST_CASE("test compile deep graph") {
  auto cfun = compile(deep_unary_compile);
  auto x = array({3.0f, -2.0f});
  auto out = cfun({x})[0];
  auto expected_out = deep_unary_compile({x})[0];
  CHECK(allclose(out, expected_out).item<bool>());
}

auto repeat_input_to_compiled(const std::vector<array>& inputs) {
  auto x = abs(exp(inputs[0]));
  auto y = abs(exp(sum(x)));
  return std::vector<array>{x + y};
}

TEST_CASE("test compile repeat input") {
  auto cfun = compile(repeat_input_to_compiled);
  auto x = array({3.0f, -2.0f});
  auto out = cfun({x})[0];
  auto expected_out = repeat_input_to_compiled({x})[0];
  CHECK(allclose(out, expected_out).item<bool>());
}

auto compile_unary_inner(const std::vector<array>& inputs) {
  auto x = inputs[0];
  return std::vector<array>{exp(exp(x))};
}

auto compile_unary_outer(const std::vector<array>& inputs) {
  auto cfun = compile(compile_unary_inner);
  return cfun(cfun(inputs));
}

TEST_CASE("test compile compiled function") {
  auto cfun = compile(compile_unary_outer);
  auto x = array({1.0f});
  auto out = cfun({x})[0];
  auto& p = out.primitive();
  CHECK_EQ(typeid(p), typeid(Compiled));
  CHECK_EQ(out.inputs()[0].id(), x.id());
}

auto grad_unary_compiled(const std::vector<array>& inputs) {
  auto gradfn = value_and_grad(compile(compile_unary_inner));
  auto [out, grad] = gradfn(inputs);
  return std::vector{out[0], grad[0]};
}

TEST_CASE("test transform compiled function") {
  auto cfun = compile(grad_unary_compiled);
  auto x = array(1.0f);
  auto outs = cfun({x});
  auto& p = outs[0].primitive();
  CHECK_EQ(typeid(p), typeid(Compiled));
  CHECK_EQ(outs[0].siblings()[0].id(), outs[1].id());
  CHECK(!outs[0].inputs()[0].has_primitive());
  CHECK(!outs[0].inputs()[1].has_primitive());
}

TEST_CASE("test fusion kernel reuse") {
  auto cfun = compile(gelu_1);
  auto x = array({2.0f, -2.0f});
  auto y = cfun({x})[0];
  auto p = std::dynamic_pointer_cast<Compiled>(y.primitive_ptr());
  eval(y);

  std::string lib_name = p->lib_name();
  CHECK(!lib_name.empty());

  x = astype(reshape(arange(10), {2, 5}), float32);
  auto z = cfun({x})[0];
  auto pz = std::dynamic_pointer_cast<Compiled>(z.primitive_ptr());
  eval(z);

  std::string lib_name_z = pz->lib_name();
  CHECK(!lib_name_z.empty());

  CHECK_EQ(lib_name, lib_name_z);
}

auto add3(const std::vector<array>& xs) {
  return std::vector<array>{xs[0] + xs[0] + xs[0]};
}

TEST_CASE("test fusion types") {
  auto cfun = compile(add3);
  auto x = array({2.0f, -2.0f});
  auto y = cfun({x})[0];
  auto p = std::dynamic_pointer_cast<Compiled>(y.primitive_ptr());
  eval(y);

  std::string lib_name = p->lib_name();
  CHECK(!lib_name.empty());

  x = array({2, -2}, int32);
  auto z = cfun({x})[0];
  auto pz = std::dynamic_pointer_cast<Compiled>(z.primitive_ptr());
  eval(z);

  std::string lib_name_z = pz->lib_name();
  CHECK(!lib_name_z.empty());
}

auto compile_shapeless_not_ok(const std::vector<array>& inputs) {
  auto x = reshape(inputs[0], {2, 2});
  return std::vector<array>{x};
}

auto compile_shapeless_ok(const std::vector<array>& inputs) {
  auto x = inputs[0] + array({2});
  return std::vector<array>{x};
}

TEST_CASE("test shapeless compile") {
  {
    auto cfun = compile(compile_shapeless_not_ok, /* shapeless */ true);
    cfun({array({1, 2, 3, 4})});
    CHECK_THROWS(cfun({array({1, 2, 3, 4, 5})}));
  }

  {
    auto cfun = compile(compile_shapeless_ok, /* shapeless */ true);
    auto out = cfun({array({1, 2})})[0];
    auto out2 = cfun({array({1, 2, 3, 4})})[0];

    // Not making a new constant array since no recompile,
    // hence the ids should be the same
    CHECK_EQ(out.inputs()[1].id(), out2.inputs()[1].id());
    CHECK(array_equal(out2, array({3, 4, 5, 6})).item<bool>());

    // Recompile since type changes
    out2 = cfun({array({1.0, 2.0})})[0];
    CHECK_NE(out.inputs()[1].id(), out2.inputs()[1].id());

    // Recompile since ndim changes
    out2 = cfun({array({1.0, 2.0}, {1, 2})})[0];
    CHECK_NE(out.inputs()[1].id(), out2.inputs()[1].id());
  }
}

auto compile_broadcast_add(const std::vector<array>& inputs) {
  auto b = zeros({8, 8});
  return std::vector<array>{inputs[0] + b};
}

TEST_CASE("test compile strides") {
  {
    auto cfun = compile(compile_broadcast_add);
    auto a = zeros({1, 8, 8});
    auto out = cfun({a})[0];
    eval(out);
    CHECK_EQ(out.strides().size(), 3);
  }
}

TEST_CASE("test compile change streams") {
  auto cfun = compile(simple_fun);
  auto out = cfun({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.primitive().stream(), default_stream(default_device()));

  auto s = new_stream(default_device());
  StreamContext sctx(s);
  out = cfun({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.primitive().stream(), s);
}

TEST_CASE("test compile lambda") {
  auto fun = [](const std::vector<array>& inputs) {
    return std::vector<array>{abs(inputs[0])};
  };

  auto out = compile(fun)({array(-1)});
  CHECK_EQ(out[0].item<int>(), 1);

  decltype(compile(nullptr)) c_local_fun;
  {
    auto local_fun = [](const std::vector<array>& inputs) {
      return std::vector<array>{abs(inputs[0])};
    };
    c_local_fun = compile(local_fun);
  }

  // This is ok even though local_fun is out of scope
  out = c_local_fun({array(-1)});
  CHECK_EQ(out[0].item<int>(), 1);

  {
    int x = 2;
    auto local_fun = [x](const std::vector<array>& inputs) {
      return std::vector<array>{inputs[0] + x};
    };
    c_local_fun = compile(local_fun);
  }
  // Also ok even though local_fun is out of scope.
  out = c_local_fun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  int x = 2;
  auto fun_with_capture = [&x](const std::vector<array>& inputs) {
    return std::vector<array>{inputs[0] + x};
  };
  auto cfun = compile(fun_with_capture);
  out = cfun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  // Doesn't recompile
  x = 3;
  out = cfun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  // Recompiles
  auto cfun2 = compile(fun_with_capture);
  out = cfun2({array(0)});
  CHECK_EQ(out[0].item<int>(), 3);
}

TEST_CASE("test compile with no-ops") {
  auto fun = [](const std::vector<array>& inputs) {
    return std::vector<array>{abs(stop_gradient(abs(inputs[0])))};
  };
  auto in = array(1.0);
  auto out = compile(fun)({in})[0];
  CHECK_EQ(out.inputs()[0].id(), in.id());
}

TEST_CASE("test compile random bits") {
  auto fun = [](const std::vector<array>& inputs) {
    auto key = inputs[0];
    auto a = random::bits({32, 32}, 4, key);
    auto b = random::bits({32, 32}, 2, key);
    return std::vector<array>{a + b};
  };
  auto in = random::key(0);
  auto expected = fun({in})[0];
  auto out = compile(fun)({in})[0];
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test compile throwing first trace does not poison cache") {
  // A nullary function whose first trace raises. Without rolling the cache
  // entry back, the second call with matching (empty) inputs would hit a
  // half-filled entry, skip tracing, and silently return empty outputs.
  bool should_throw = true;
  auto fun = [&should_throw](const std::vector<array>& inputs) {
    if (should_throw) {
      throw std::runtime_error("trace failure");
    }
    return std::vector<array>{array(1.0f) + array(2.0f)};
  };

  auto cfun = compile(fun);

  // First call: the trace raises and must propagate.
  CHECK_THROWS_AS(cfun({}), std::runtime_error);

  // Second call: still raising. It must re-trace and raise again rather than
  // return an empty result from a poisoned cache entry.
  CHECK_THROWS_AS(cfun({}), std::runtime_error);

  // Once the function stops raising, a retry must trace cleanly and succeed.
  should_throw = false;
  auto out = cfun({});
  REQUIRE_EQ(out.size(), 1);
  CHECK_EQ(out[0].item<float>(), 3.0f);
}

auto matmul_bias_fun(const std::vector<array>& inputs) {
  return std::vector<array>{matmul(inputs[0], inputs[1]) + inputs[2]};
}

// copy for deterministic compile checking
auto matmul_bias_fun_copy(const std::vector<array>& inputs) {
  return std::vector<array>{matmul(inputs[0], inputs[1]) + inputs[2]};
}

// Compare matmul-fusion results with tolerance for AddMM reassociation:
// fused alpha/beta scaling happens inside GEMM, not as separate ops.
bool matmul_allclose(const array& x, const array& y) {
  return allclose(x, y, /* rtol = */ 1e-4, /* atol = */ 1e-5).item<bool>();
}

// Proves whether compile produced a CompiledMatmul by checking the root
// primitive, then verifies the fused graph matches the original function.
array check_matmul_fusion(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    const std::vector<array>& args,
    bool fused) {
  auto out = compile(fun)(args)[0];
  auto& p = out.primitive();
  CHECK_EQ(typeid(p) == typeid(CompiledMatmul), fused);
  CHECK(matmul_allclose(out, fun(args)[0]));
  return out;
}

TEST_CASE("test compile matmul epilogue fused") {
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto bias = random::uniform({16});
  eval(a, b, bias);

  // Proves matmul+bias fuses into one CompiledMatmul by checking the root
  // primitive, preserved {a,b,bias} inputs, and numerical parity.
  {
    auto out = compile(matmul_bias_fun)({a, b, bias})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(CompiledMatmul));
    REQUIRE_EQ(out.inputs().size(), 3);
    CHECK_EQ(out.inputs()[0].id(), a.id());
    CHECK_EQ(out.inputs()[1].id(), b.id());
    CHECK_EQ(out.inputs()[2].id(), bias.id());
    CHECK(matmul_allclose(out, matmul_bias_fun({a, b, bias})[0]));
  }

  // Proves equivalent matmul epilogues compare equal by compiling two
  // structurally identical functions and checking is_equivalent.
  CHECK(compile(matmul_bias_fun)({a, b, bias})[0].primitive().is_equivalent(
      compile(matmul_bias_fun_copy)({a, b, bias})[0].primitive()));
}

TEST_CASE("test compile matmul epilogue guards") {
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto bias = random::uniform({16});
  auto c = random::uniform({4, 16});
  eval(a, b, bias, c);

  // Proves shared matmul producers are not absorbed by checking a matmul with
  // two consumers remains materialized and both outputs match eager results.
  {
    auto fun = [](const std::vector<array>& ins) {
      auto x = matmul(ins[0], ins[1]);
      return std::vector<array>{x + ins[2], x * array(2.0f)};
    };
    auto outs = compile(fun)({a, b, bias});
    // Check structure before parity, since eval can detach the graph.
    for (int i = 0; i < 2; ++i) {
      auto& p = outs[i].primitive();
      CHECK_NE(typeid(p), typeid(CompiledMatmul));
    }
    auto& mp = outs[0].inputs()[0].primitive();
    CHECK_EQ(typeid(mp), typeid(Matmul));
    auto expected = fun({a, b, bias});
    for (int i = 0; i < 2; ++i) {
      CHECK(matmul_allclose(outs[i], expected[i]));
    }
  }

  // Proves a self-referential activation can be an epilogue by compiling
  // x*sigmoid(x) after matmul and checking it fuses with correct values.
  check_matmul_fusion(
      [](const std::vector<array>& ins) {
        auto x = matmul(ins[0], ins[1]);
        return std::vector<array>{x * sigmoid(x)};
      },
      {a, b},
      true);

  // Proves duplicate references block matmul fusion by using x+x and checking
  // the matmul stays as a separate producer.
  {
    auto fun = [](const std::vector<array>& ins) {
      auto x = matmul(ins[0], ins[1]);
      return std::vector<array>{x + x};
    };
    auto out = compile(fun)({a, b})[0];
    // Check structure before parity, since eval can detach the graph.
    auto& p = out.primitive();
    CHECK_NE(typeid(p), typeid(CompiledMatmul));
    auto& mp = out.inputs()[0].primitive();
    CHECK_EQ(typeid(mp), typeid(Matmul));
    CHECK(matmul_allclose(out, fun({a, b})[0]));
  }

  // Proves cross-stream consumers are rejected by putting Add on a new stream
  // and checking it remains above a separate Matmul.
  {
    auto fun = [](const std::vector<array>& ins) {
      auto s2 = new_stream(default_device());
      auto x = matmul(ins[0], ins[1]);
      return std::vector<array>{add(x, ins[2], s2)};
    };
    auto out = compile(fun)({a, b, c})[0];
    // Check structure before parity, since eval can detach the graph.
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Add));
    auto& mp = out.inputs()[0].primitive();
    CHECK_EQ(typeid(mp), typeid(Matmul));
    CHECK(matmul_allclose(out, fun({a, b, c})[0]));
  }

  // Proves Broadcast consumers are excluded by broadcasting matmul output and
  // checking the Broadcast remains above a separate Matmul.
  {
    auto fun = [](const std::vector<array>& ins) {
      return std::vector<array>{
          broadcast_to(matmul(ins[0], ins[1]), {3, 4, 16})};
    };
    auto out = compile(fun)({a, b})[0];
    auto& p = out.primitive();
    CHECK_NE(typeid(p), typeid(CompiledMatmul));
    auto& mp = out.inputs()[0].primitive();
    CHECK_EQ(typeid(mp), typeid(Matmul));
    CHECK(matmul_allclose(out, fun({a, b})[0]));
  }

  // Proves non-fusable consumers are rejected by reducing matmul output and
  // checking Reduce remains above a separate Matmul.
  {
    auto fun = [](const std::vector<array>& ins) {
      return std::vector<array>{sum(matmul(ins[0], ins[1]), true)};
    };
    auto out = compile(fun)({a, b})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Reduce));
    auto& mp = out.inputs()[0].primitive();
    CHECK_EQ(typeid(mp), typeid(Matmul));
    CHECK(matmul_allclose(out, fun({a, b})[0]));
  }

  // Proves global-output matmuls are not absorbed by returning x and x+bias,
  // then checking x remains a Matmul and values match eager.
  {
    auto fun = [](const std::vector<array>& ins) {
      auto x = matmul(ins[0], ins[1]);
      return std::vector<array>{x, x + ins[2]};
    };
    auto outs = compile(fun)({a, b, c});
    auto& p0 = outs[0].primitive();
    CHECK_EQ(typeid(p0), typeid(Matmul));
    auto& p1 = outs[1].primitive();
    CHECK_NE(typeid(p1), typeid(CompiledMatmul));
    auto expected = fun({a, b, c});
    for (int i = 0; i < 2; ++i) {
      CHECK(matmul_allclose(outs[i], expected[i]));
    }
  }

  // Proves no_fuse disables both normal fusion and matmul epilogue fusion by
  // checking matmul+bias stays as Add over Matmul.
  {
    detail::compile_clear_cache();
    set_compile_mode(CompileMode::no_fuse);
    auto out = compile(matmul_bias_fun)({a, b, c})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Add));
    CHECK(matmul_allclose(out, matmul_bias_fun({a, b, c})[0]));
    set_compile_mode(CompileMode::enabled);
    detail::compile_clear_cache();
  }
}

TEST_CASE("test compile matmul epilogue shapes") {
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto bias = random::uniform({16});
  eval(a, b, bias);

  // Proves batched matmul+bias preserves broadcast semantics by fusing
  // {2,4,8}@{2,8,16}+{16} and checking parity.
  auto a3 = random::uniform({2, 4, 8});
  auto b3 = random::uniform({2, 8, 16});
  eval(a3, b3);
  check_matmul_fusion(matmul_bias_fun, {a3, b3, bias}, true);

  // Proves lone elementwise consumers become epilogues by fusing matmul*c
  // with a runtime matrix operand and checking parity.
  auto c = random::uniform({4, 16});
  eval(c);
  check_matmul_fusion(
      [](const std::vector<array>& ins) {
        return std::vector<array>{matmul(ins[0], ins[1]) * ins[2]};
      },
      {a, b, c},
      true);

  // Proves vector-matmul epilogues stay correct by compiling v@b+bias and
  // checking parity without depending on reshape-heavy graph structure.
  {
    auto v = random::uniform({8});
    eval(v);
    auto fun = [](const std::vector<array>& ins) {
      return std::vector<array>{matmul(ins[0], ins[1]) + ins[2]};
    };
    auto out = compile(fun)({v, b, bias})[0];
    CHECK(matmul_allclose(out, fun({v, b, bias})[0]));
  }

  // Proves shapeless fallback epilogues compute new shapes by reusing
  // matmul+bias+ReLU with different input rows and checking parity.
  {
    auto relu_bias = [](const std::vector<array>& ins) {
      auto y = matmul(ins[0], ins[1]) + ins[2];
      return std::vector<array>{maximum(y, array(0.0f))};
    };
    auto cfun = compile(+relu_bias, /* shapeless = */ true);
    auto out = cfun({a, b, bias})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(CompiledMatmul));
    auto a2 = random::uniform({7, 8});
    eval(a2);
    for (auto& lhs : {a, a2}) {
      CHECK(matmul_allclose(
          cfun({lhs, b, bias})[0], relu_bias({lhs, b, bias})[0]));
    }
  }
}

TEST_CASE("test compile matmul epilogue chained") {
  // Proves chained matmuls each absorb their own epilogue by checking the
  // final CompiledMatmul consumes another CompiledMatmul and matches eager.
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto w = random::uniform({16, 8});
  auto bias = random::uniform({8});
  eval(a, b, w, bias);

  auto fun = [](const std::vector<array>& ins) {
    auto h = maximum(matmul(ins[0], ins[1]), array(0.0f));
    return std::vector<array>{matmul(h, ins[2]) + ins[3]};
  };
  auto out = compile(fun)({a, b, w, bias})[0];
  // Structure first: eval detaches the graph
  auto& p = out.primitive();
  CHECK_EQ(typeid(p), typeid(CompiledMatmul));
  auto& hp = out.inputs()[0].primitive();
  CHECK_EQ(typeid(hp), typeid(CompiledMatmul));
  CHECK(matmul_allclose(out, fun({a, b, w, bias})[0]));
}

TEST_CASE("test compiled matmul addmm dispatch") {
  // Proves matches_addmm drives backend dispatch by inspecting real
  // CompiledMatmul nodes for AddMM eligibility and extracted parameters.
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto c = random::uniform({4, 16});
  eval(a, b, c);

  float alpha, beta;
  int c_index;

  // Proves AddMM-shaped epilogues report alpha, beta, and c_index correctly
  // by matching them against the compiled node's runtime inputs.
  {
    auto fun = [](const std::vector<array>& ins) {
      return std::vector<array>{
          array(0.5f) * matmul(ins[0], ins[1]) + array(-1.5f) * ins[2]};
    };
    auto out = compile(fun)({a, b, c})[0];
    auto p = std::dynamic_pointer_cast<CompiledMatmul>(out.primitive_ptr());
    REQUIRE(p);
    CHECK(p->matches_addmm(alpha, beta, c_index));
    CHECK_EQ(alpha, 0.5f);
    CHECK_EQ(beta, -1.5f);
    CHECK_EQ(out.inputs()[c_index].id(), c.id());
    CHECK(matmul_allclose(out, fun({a, b, c})[0]));
  }

  // Proves non-AddMM fused epilogues reject AddMM dispatch by adding ReLU and
  // checking matches_addmm returns false while values still match.
  {
    auto fun = [](const std::vector<array>& ins) {
      auto y = matmul(ins[0], ins[1]) + ins[2];
      return std::vector<array>{maximum(y, array(0.0f))};
    };
    auto out = compile(fun)({a, b, c})[0];
    auto p = std::dynamic_pointer_cast<CompiledMatmul>(out.primitive_ptr());
    REQUIRE(p);
    CHECK(!p->matches_addmm(alpha, beta, c_index));
    CHECK(matmul_allclose(out, fun({a, b, c})[0]));
  }
}

TEST_CASE("test compiled matmul fallback") {
  // Proves the fallback evaluator and equivalence checks by hand-building a
  // CompiledMatmul with a Multiply epilogue and comparing variants.
  auto s = default_stream(default_device());
  auto a = random::uniform({4, 8});
  auto b = random::uniform({8, 16});
  auto c = random::uniform({4, 16});
  eval(a, b, c);

  // Build a one-op epilogue over placeholder accumulator/c inputs so the test
  // can vary epilogue and producer primitives directly.
  array acc(Shape{4, 16}, float32, nullptr, {});
  array c_in(Shape{4, 16}, float32, nullptr, {});
  auto make_epilogue = [&](std::shared_ptr<Primitive> op) {
    array r(
        Shape{4, 16}, float32, std::move(op), std::vector<array>{acc, c_in});
    return std::make_shared<Compiled>(
        s,
        std::vector<array>{acc, c_in},
        std::vector<array>{r},
        std::vector<array>{r},
        std::unordered_set<uintptr_t>{});
  };

  array out(
      Shape{4, 16},
      float32,
      std::make_shared<CompiledMatmul>(
          s,
          std::make_shared<Matmul>(s),
          make_epilogue(std::make_shared<Multiply>(s))),
      std::vector<array>{a, b, c});

  // Proves epilogue structure participates in equivalence by comparing
  // Multiply and Add epilogues before eval detaches the graph.
  CompiledMatmul with_add(
      s, std::make_shared<Matmul>(s), make_epilogue(std::make_shared<Add>(s)));
  CHECK(!out.primitive().is_equivalent(with_add));

  // Proves producer primitive type participates in equivalence by comparing
  // Matmul and AddMM producers with the same Multiply epilogue.
  CompiledMatmul with_addmm_producer(
      s,
      std::make_shared<AddMM>(s, 1.0f, 1.0f),
      make_epilogue(std::make_shared<Multiply>(s)));
  CHECK(!out.primitive().is_equivalent(with_addmm_producer));

  // Proves identical producer and epilogue structures compare equivalent by
  // constructing a matching CompiledMatmul.
  CompiledMatmul same(
      s,
      std::make_shared<Matmul>(s),
      make_epilogue(std::make_shared<Multiply>(s)));
  CHECK(out.primitive().is_equivalent(same));

  CHECK(matmul_allclose(out, matmul(a, b) * c));
}
