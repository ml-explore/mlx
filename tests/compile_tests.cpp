// Copyright © 2023-2024 Apple Inc.

#include "doctest/doctest.h"

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
    CHECK(array_equal(e, array({0.0f, 1.0f, 0.0f, 1.0f})).item<bool>());
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[2].id());
    CHECK_EQ(e.inputs()[1].id(), e.inputs()[3].id());
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

auto compile_accross_streams(const std::vector<array>& inputs) {
  auto s2 = new_stream(default_device());
  auto x = exp(abs(inputs[0]));
  auto y = exp(abs(x, s2), s2);
  return std::vector<array>{y};
}

TEST_CASE("test compile accross streams") {
  auto cfun = compile(compile_accross_streams);
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
    CHECK_THROWS(cfun({array({1, 2, 3, 4})}));
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
