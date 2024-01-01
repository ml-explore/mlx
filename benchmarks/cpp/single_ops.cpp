// Copyright © 2023 Apple Inc.

#include "mlx/mlx.h"
#include "time_utils.h"

using namespace mlx::core;

void time_creation_ops() {
  int M = 2000;
  int N = 500;
  auto shape = {M, N};
  auto full_fp32 = [&]() { return full(shape, 3.3f); };
  TIME(full_fp32);
  auto zeros_fp32 = [&]() { return zeros(shape, float32); };
  TIME(zeros_fp32);
  auto ones_fp32 = [&]() { return ones(shape, float32); };
  TIME(ones_fp32);

  auto arange_fp32 = [&]() { return arange(0.0, 10.0, 1e-4); };
  TIME(arange_fp32);
}

void time_type_conversions() {
  int M = 2000;
  int N = 500;
  auto shape = {M, N};
  auto device = default_device();

  auto a = zeros(shape, float32);
  eval(a);
  TIMEM("float32 to int32", astype, a, int32, device);
  TIMEM("float32 to uint32", astype, a, uint32, device);

  a = zeros(shape, int32);
  eval(a);
  TIMEM("int32 to float32", astype, a, float32, device);

  a = zeros(shape, bool_);
  eval(a);
  TIMEM("bool to float32", astype, a, float32, device);
  TIMEM("bool to int32", astype, a, int32, device);
  TIMEM("bool to uint32", astype, a, uint32, device);
}

void time_random_generation() {
  int M = 2000;
  int N = 500;

  auto uniform = [&]() { return random::uniform({M, N}, float32); };
  TIME(uniform);
  auto normal = [&]() { return random::normal({M, N}, float32); };
  TIME(normal);
}

void time_unary_ops() {
  int M = 2000;
  int N = 500;
  auto device = default_device();

  auto a = random::normal({M, N});
  eval(a);
  TIME(mlx::core::abs, a, device);
  TIME(negative, a, device);
  TIME(sign, a, device);
  TIME(square, a, device);
  TIME(mlx::core::sqrt, a, device);
  TIME(rsqrt, a, device);
  TIME(mlx::core::exp, a, device);

  a = random::uniform({M, N});
  TIME(mlx::core::log, a, device);
}

void time_binary_ops() {
  int M = 1000, N = 100, K = 10;
  auto a = random::uniform({M, N, K});
  auto b = random::uniform({M, N, K});
  auto device = default_device();
  eval(a, b);

  TIME(add, a, b, device);
  TIME(subtract, a, b, device);
  TIME(multiply, a, b, device);
  TIME(divide, a, b, device);
  TIME(maximum, a, b, device);
  TIME(minimum, a, b, device);

  b = random::uniform({1});
  eval(b);
  TIMEM("scalar", add, a, b, device);
  TIMEM("vector-scalar", subtract, a, b, device);
  TIMEM("scalar-vector", subtract, b, a, device);
  TIMEM("scalar", multiply, a, b, device);
  TIMEM("vector-scalar", divide, a, b, device);
  TIMEM("scalar-vector", divide, b, a, device);

  a = broadcast_to(random::uniform({1}), {1000, 100});
  b = broadcast_to(random::uniform({1}), {1000, 100});
  eval(a, b);
  TIMEM("scalar-scalar broadcast", add, a, b, device);
  TIMEM("scalar-scalar broadcast", subtract, a, b, device);
  TIMEM("scalar-scalar broadcast", multiply, a, b, device);
  TIMEM("scalar-scalar broadcast", divide, a, b, device);
}

void time_strided_ops() {
  int M = 50, N = 50, O = 50, P = 50;
  auto a = random::uniform({M, N, O, P});
  auto b = random::uniform({M, N, O, P});
  auto device = default_device();
  eval(a, b);
  TIMEM("non-strided", add, a, b, device);
  a = transpose(a, {1, 0, 2, 3});
  b = transpose(b, {3, 2, 0, 1});
  eval(a, b);
  TIMEM("strided", add, a, b, device);
}

void time_comparisons() {
  int M = 1000, N = 100, K = 10;
  auto a = random::uniform({M, N, K});
  auto b = random::uniform({M, N, K});
  auto device = default_device();
  eval(a, b);
  TIME(equal, a, b, device);
  TIME(greater, a, b, device);
  TIME(greater_equal, a, b, device);
  TIME(less, a, b, device);
  TIME(less_equal, a, b, device);
}

void time_matvec() {
  int M = 2000, N = 200;
  auto a = random::uniform({M, N});
  auto b = random::uniform({N});
  auto c = random::uniform({M});
  eval(a, b, c);
  auto matvec = [&]() { return matmul(a, b); };
  TIME(matvec);

  auto matvec_transpose = [&]() { return matmul(transpose(a), c); };
  TIME(matvec_transpose);
}

void time_matmul() {
  int M = 1000, N = 1000, K = 1000;
  auto a = random::uniform({M, K});
  auto b = random::uniform({K, N});
  auto device = default_device();
  eval(a, b);
  TIME(matmul, a, b, device);

  auto transpose_matmul = [&]() { return matmul(transpose(a), b); };
  TIME(transpose_matmul);
}

void time_reductions() {
  auto a = random::normal({10000, 1000});
  eval(a);
  auto sum_all = [&a]() { return sum(a, false); };
  TIME(sum_all);

  auto sum_along_0 = [&a]() { return sum(a, 0, false); };
  TIME(sum_along_0);

  auto sum_along_1 = [&a]() { return sum(a, 1, false); };
  TIME(sum_along_1);

  auto prod_all = [&a]() { return prod(a, false); };
  TIME(prod_all);

  auto all_true = [&a]() { return all(a, false); };
  TIME(all_true);

  auto all_along_0 = [&a]() { return all(a, 0, false); };
  TIME(all_along_0);

  auto all_along_1 = [&a]() { return all(a, 1, false); };
  TIME(all_along_1);

  auto any_true = [&a]() { return any(a, false); };
  TIME(any_true);

  auto argmin_along_0 = [&a]() { return argmin(a, 0, false); };
  TIME(argmin_along_0);

  auto argmin_along_1 = [&a]() { return argmin(a, 1, false); };
  TIME(argmin_along_1);
}

void time_gather_scatter() {
  auto a = random::normal({1000, 768});
  eval(a);
  auto indices = random::randint(0, 1000, {256});
  eval(indices);

  auto embedding_lookup = [&a, &indices]() { return take(a, indices, 0); };
  TIME(embedding_lookup);

  indices = random::randint(0, 768 * 1000, {256 * 768});
  eval(indices);

  auto single_element_lookup = [&a, &indices]() { return take(a, indices); };
  TIME(single_element_lookup);

  indices = random::randint(0, 1000, {256});
  auto updates = random::normal({256, 1, 768});
  eval(indices, updates);

  auto embedding_update = [&a, &indices, &updates]() {
    return scatter(a, indices, updates, 0);
  };
  TIME(embedding_update);

  auto embedding_add = [&a, &indices, &updates]() {
    return scatter_add(a, indices, updates, 0);
  };
  TIME(embedding_add);

  a = reshape(a, {-1});
  indices = random::randint(0, 768 * 1000, {768 * 256});
  updates = random::normal({256 * 768, 1});
  eval(a, indices, updates);

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
  auto a = random::normal({1000});
  auto b = random::normal({1000});
  eval({a, b});

  auto divmod_fused = [&a, &b]() { return divmod(a, b); };
  TIME(divmod_fused);

  auto divmod_separate = [&a, &b]() {
    return std::vector<array>{floor_divide(a, b), remainder(a, b)};
  };
  TIME(divmod_separate);
}

int main() {
  std::cout << "Benchmarks for " << default_device() << std::endl;
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
