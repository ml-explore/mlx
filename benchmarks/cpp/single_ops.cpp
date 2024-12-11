// Copyright © 2023 Apple Inc.

#include "mlx/mlx.h"
#include "time_utils.h"

namespace mx = mlx::core;

void time_creation_ops() {
  int M = 2000;
  int N = 500;
  auto shape = {M, N};
  auto full_fp32 = [&]() { return mx::full(shape, 3.3f); };
  TIME(full_fp32);
  auto zeros_fp32 = [&]() { return mx::zeros(shape, mx::float32); };
  TIME(zeros_fp32);
  auto ones_fp32 = [&]() { return mx::ones(shape, mx::float32); };
  TIME(ones_fp32);

  auto arange_fp32 = [&]() { return mx::arange(0.0, 10.0, 1e-4); };
  TIME(arange_fp32);
}

void time_type_conversions() {
  int M = 2000;
  int N = 500;
  auto shape = {M, N};
  auto device = mx::default_device();

  auto a = mx::zeros(shape, mx::float32);
  mx::eval(a);
  TIMEM("mx::float32 to mx::int32", mx::astype, a, mx::int32, device);
  TIMEM("mx::float32 to mx::uint32", mx::astype, a, mx::uint32, device);

  a = mx::zeros(shape, mx::int32);
  mx::eval(a);
  TIMEM("mx::int32 to mx::float32", mx::astype, a, mx::float32, device);

  a = mx::zeros(shape, mx::bool_);
  mx::eval(a);
  TIMEM("bool to mx::float32", mx::astype, a, mx::float32, device);
  TIMEM("bool to mx::int32", mx::astype, a, mx::int32, device);
  TIMEM("bool to mx::uint32", mx::astype, a, mx::uint32, device);
}

void time_random_generation() {
  int M = 2000;
  int N = 500;

  auto uniform = [&]() { return mx::random::uniform({M, N}, mx::float32); };
  TIME(uniform);
  auto normal = [&]() { return mx::random::normal({M, N}, mx::float32); };
  TIME(normal);
}

void time_unary_ops() {
  int M = 2000;
  int N = 500;
  auto device = mx::default_device();

  auto a = mx::random::normal({M, N});
  mx::eval(a);
  TIME(mlx::core::abs, a, device);
  TIME(mx::negative, a, device);
  TIME(mx::sign, a, device);
  TIME(mx::square, a, device);
  TIME(mlx::core::sqrt, a, device);
  TIME(mx::rsqrt, a, device);
  TIME(mlx::core::exp, a, device);

  a = mx::random::uniform({M, N});
  TIME(mlx::core::log, a, device);
}

void time_binary_ops() {
  int M = 1000, N = 100, K = 10;
  auto condition = mx::random::randint(0, 2, {M, N, K});
  auto a = mx::random::uniform({M, N, K});
  auto b = mx::random::uniform({M, N, K});
  auto device = mx::default_device();
  mx::eval(a, b);

  TIME(mx::add, a, b, device);
  TIME(mx::subtract, a, b, device);
  TIME(mx::multiply, a, b, device);
  TIME(mx::divide, a, b, device);
  TIME(mx::maximum, a, b, device);
  TIME(mx::minimum, a, b, device);
  TIME(mx::where, condition, a, b, device);

  condition = mx::array({true});
  b = mx::random::uniform({1});
  mx::eval(b);
  TIMEM("scalar", mx::add, a, b, device);
  TIMEM("vector-scalar", mx::subtract, a, b, device);
  TIMEM("scalar-vector", mx::subtract, b, a, device);
  TIMEM("scalar", mx::multiply, a, b, device);
  TIMEM("vector-scalar", mx::divide, a, b, device);
  TIMEM("scalar-vector", mx::divide, b, a, device);
  TIMEM("scalar-vector", mx::where, condition, a, b, device);

  condition = mx::broadcast_to(mx::array({true}), {1000, 100});
  a = mx::broadcast_to(mx::random::uniform({1}), {1000, 100});
  b = mx::broadcast_to(mx::random::uniform({1}), {1000, 100});
  mx::eval(a, b);
  TIMEM("scalar-scalar broadcast", mx::add, a, b, device);
  TIMEM("scalar-scalar broadcast", mx::subtract, a, b, device);
  TIMEM("scalar-scalar broadcast", mx::multiply, a, b, device);
  TIMEM("scalar-scalar broadcast", mx::divide, a, b, device);
  TIMEM("scalar-scalar broadcast", mx::where, condition, a, b, device);
}

void time_strided_ops() {
  int M = 50, N = 50, O = 50, P = 50;
  auto a = mx::random::uniform({M, N, O, P});
  auto b = mx::random::uniform({M, N, O, P});
  auto device = mx::default_device();
  mx::eval(a, b);
  TIMEM("non-strided", mx::add, a, b, device);
  a = mx::transpose(a, {1, 0, 2, 3});
  b = mx::transpose(b, {3, 2, 0, 1});
  mx::eval(a, b);
  TIMEM("strided", mx::add, a, b, device);
}

void time_comparisons() {
  int M = 1000, N = 100, K = 10;
  auto a = mx::random::uniform({M, N, K});
  auto b = mx::random::uniform({M, N, K});
  auto device = mx::default_device();
  mx::eval(a, b);
  TIME(mx::equal, a, b, device);
  TIME(mx::greater, a, b, device);
  TIME(mx::greater_equal, a, b, device);
  TIME(mx::less, a, b, device);
  TIME(mx::less_equal, a, b, device);
}

void time_matvec() {
  int M = 2000, N = 200;
  auto a = mx::random::uniform({M, N});
  auto b = mx::random::uniform({N});
  auto c = mx::random::uniform({M});
  mx::eval(a, b, c);
  auto matvec = [&]() { return mx::matmul(a, b); };
  TIME(matvec);

  auto matvec_transpose = [&]() { return mx::matmul(mx::transpose(a), c); };
  TIME(matvec_transpose);
}

void time_matmul() {
  int M = 1000, N = 1000, K = 1000;
  auto a = mx::random::uniform({M, K});
  auto b = mx::random::uniform({K, N});
  auto device = mx::default_device();
  mx::eval(a, b);
  TIME(mx::matmul, a, b, device);

  auto transpose_matmul = [&]() { return mx::matmul(mx::transpose(a), b); };
  TIME(transpose_matmul);
}

void time_reductions() {
  auto a = mx::random::normal({10000, 1000});
  mx::eval(a);
  auto sum_all = [&a]() { return mx::sum(a, false); };
  TIME(sum_all);

  auto sum_along_0 = [&a]() { return mx::sum(a, 0, false); };
  TIME(sum_along_0);

  auto sum_along_1 = [&a]() { return mx::sum(a, 1, false); };
  TIME(sum_along_1);

  auto prod_all = [&a]() { return mx::prod(a, false); };
  TIME(prod_all);

  auto all_true = [&a]() { return mx::all(a, false); };
  TIME(all_true);

  auto all_along_0 = [&a]() { return mx::all(a, 0, false); };
  TIME(all_along_0);

  auto all_along_1 = [&a]() { return mx::all(a, 1, false); };
  TIME(all_along_1);

  auto any_true = [&a]() { return mx::any(a, false); };
  TIME(any_true);

  auto argmin_along_0 = [&a]() { return mx::argmin(a, 0, false); };
  TIME(argmin_along_0);

  auto argmin_along_1 = [&a]() { return mx::argmin(a, 1, false); };
  TIME(argmin_along_1);
}

void time_gather_scatter() {
  auto a = mx::random::normal({1000, 768});
  mx::eval(a);
  auto indices = mx::random::randint(0, 1000, {256});
  mx::eval(indices);

  auto embedding_lookup = [&a, &indices]() { return mx::take(a, indices, 0); };
  TIME(embedding_lookup);

  indices = mx::random::randint(0, 768 * 1000, {256 * 768});
  mx::eval(indices);

  auto single_element_lookup = [&a, &indices]() {
    return mx::take(a, indices);
  };
  TIME(single_element_lookup);

  indices = mx::random::randint(0, 1000, {256});
  auto updates = mx::random::normal({256, 1, 768});
  mx::eval(indices, updates);

  auto embedding_update = [&a, &indices, &updates]() {
    return scatter(a, indices, updates, 0);
  };
  TIME(embedding_update);

  auto embedding_add = [&a, &indices, &updates]() {
    return scatter_add(a, indices, updates, 0);
  };
  TIME(embedding_add);

  a = mx::reshape(a, {-1});
  indices = mx::random::randint(0, 768 * 1000, {768 * 256});
  updates = mx::random::normal({256 * 768, 1});
  mx::eval(a, indices, updates);

  auto single_element_update = [&a, &indices, &updates]() {
    return scatter(a, indices, updates, 0);
  };
  TIME(single_element_update);

  auto single_element_add = [&a, &indices, &updates]() {
    return scatter_add(a, indices, updates, 0);
  };
  TIME(single_element_add);
}

void time_divmod() {
  auto a = mx::random::normal({1000});
  auto b = mx::random::normal({1000});
  mx::eval({a, b});

  auto divmod_fused = [&a, &b]() { return mx::divmod(a, b); };
  TIME(divmod_fused);

  auto divmod_separate = [&a, &b]() {
    return std::vector<mx::array>{mx::floor_divide(a, b), mx::remainder(a, b)};
  };
  TIME(divmod_separate);
}

int main() {
  std::cout << "Benchmarks for " << mx::default_device() << std::endl;
  time_creation_ops();
  time_type_conversions();
  time_unary_ops();
  time_binary_ops();
  time_strided_ops();
  time_random_generation();
  time_comparisons();
  time_matvec();
  time_matmul();
  time_reductions();
  time_gather_scatter();
  time_divmod();
}
