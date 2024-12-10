// Copyright © 2023 Apple Inc.

#include <iostream>
#include <sstream>

#include "mlx/mlx.h"
#include "time_utils.h"

namespace mx = mlx::core;

void time_irregular_binary_ops_1D() {
  auto device = mx::default_device();
  int size = 1000000;
  int step = 2;
  auto a = mx::random::uniform({size});
  auto b = mx::random::uniform({size});
  mx::eval(a, b);
  a = slice(a, {0}, {size}, {step});
  b = slice(b, {0}, {size}, {step});
  TIMEM("1D strided", mx::add, a, b, device);
}

void time_irregular_binary_ops_2D() {
  auto device = mx::default_device();
  int size = 2048;
  auto a = mx::random::uniform({size, size});
  auto b = mx::random::uniform({size, size});
  mx::eval(a, b);
  TIMEM("2D regular", mx::add, a, b, device);

  b = mx::transpose(b);
  mx::eval(b);
  TIMEM("2D mx::transpose", mx::add, a, b, device);

  b = mx::random::uniform({size});
  mx::eval(b);
  TIMEM("2D broadcast dim 0", mx::add, a, b, device);

  b = mx::reshape(b, {size, 1});
  mx::eval(b);
  TIMEM("2D broadcast dim 1", mx::add, a, b, device);
}

void time_irregular_binary_ops_3D() {
  auto device = mx::default_device();
  int d0 = 32;
  int d1 = 512;
  int d2 = 512;
  auto a = mx::random::uniform({d0, d1, d2});
  auto b = mx::random::uniform({d0, d1, d2});
  TIMEM("3D regular", mx::add, a, b, device);

  b = mx::transpose(b, {0, 2, 1});
  TIMEM("3D mx::transpose", mx::add, a, b, device);

  b = mx::random::uniform({d1, d2});
  TIMEM("3D broadcast dim 0", mx::add, a, b, device);

  b = mx::random::uniform({d0, 1, d2});
  TIMEM("3D broadcast dim 1", mx::add, a, b, device);

  b = mx::random::uniform({d0, d1, 1});
  TIMEM("3D broadcast dim 2", mx::add, a, b, device);

  b = mx::random::uniform({d2});
  TIMEM("3D broadcast dims 0, 1", mx::add, a, b, device);

  b = mx::random::uniform({d1, 1});
  TIMEM("3D broadcast dims 0, 2", mx::add, a, b, device);

  b = mx::random::uniform({d0, 1, 1});
  TIMEM("3D broadcast dims 1, 2", mx::add, a, b, device);
}

void time_irregular_binary_ops_4D() {
  auto device = mx::default_device();
  std::vector<int> shape = {8, 8, 512, 512};
  auto a = mx::random::uniform(shape);
  auto b = mx::random::uniform(shape);

  TIMEM("4D regular", mx::add, a, b, device);

  b = mx::transpose(b, {0, 1, 3, 2});
  TIMEM("4D mx::transpose", mx::add, a, b, device);

  std::string om = "4D broadcast dims ";
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = 1;
    b = mx::random::uniform(shape);
    std::ostringstream msg;
    msg << om << i;
    TIMEM(msg.str(), mx::add, a, b, device);

    for (int j = i + 1; j < shape.size(); ++j) {
      shape[j] = 1;
      std::ostringstream msg;
      msg << om << i << ", " << j;
      b = mx::random::uniform(shape);
      TIMEM(msg.str(), mx::add, a, b, device);
      shape[j] = a.shape(j);

      for (int k = j + 1; k < shape.size(); ++k) {
        shape[k] = 1;
        std::ostringstream msg;
        msg << om << i << ", " << j << ", " << k;
        b = mx::random::uniform(shape);
        TIMEM(msg.str(), mx::add, a, b, device);
        shape[k] = a.shape(k);
      }
    }
    shape[i] = a.shape(i);
  }
}

void time_irregular_reshape() {
  auto device = mx::default_device();
  std::vector<int> shape;
  auto reshape_fn = [&shape, device](const mx::array& a) {
    return mx::reshape(a, shape, device);
  };

  int size = 64;
  int d = 2 * size;

  auto a = mx::random::uniform({d, d, d});

  shape = {8 * size, size, size};
  TIMEM("3D contiguous", reshape_fn, a);

  a = mx::transpose(a);
  shape = {8 * size, size, size};
  TIMEM("3D mx::transpose", reshape_fn, a);

  a = mx::transpose(a, {1, 2, 0});
  shape = {8 * size, size, size};
  TIMEM("3D mx::transpose dims 1 2", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d, d}), {d, d, d});
  TIMEM("3D broadcast dim 0", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d, 1, d}), {d, d, d});
  TIMEM("3D broadcast dim 1", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d, d, 1}), {d, d, d});
  TIMEM("3D broadcast dim 2", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d}), {d, d, d});
  TIMEM("3D broadcast dims 0, 1", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d, 1}), {d, d, d});
  TIMEM("3D broadcast dims 0, 2", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({d, 1, 1}), {d, d, d});
  TIMEM("3D broadcast dims 1, 2", reshape_fn, a);

  a = mx::broadcast_to(mx::random::uniform({1, 1, 1}), {d, d, d});
  TIMEM("3D broadcast dims 1, 2, 3", reshape_fn, a);
}

void time_irregular_astype_1D() {
  auto device = mx::default_device();
  int size = 1000000;
  int step = 2;
  auto a = mx::random::uniform({size});
  a = slice(a, {0}, {size}, {step});
  TIMEM("1D strided", mx::astype, a, mx::int32, device);
}

void time_irregular_astype_2D() {
  auto device = mx::default_device();
  int size = 2048;
  std::vector<int> shape = {size, size};

  auto a = mx::random::uniform(shape);
  TIMEM("2D regular", mx::astype, a, mx::int32, device);

  a = mx::transpose(a);
  TIMEM("2D mx::transpose", mx::astype, a, mx::int32, device);

  a = mx::broadcast_to(mx::random::uniform({size}), shape);
  TIMEM("2D broadcast dim 0", mx::astype, a, mx::int32, device);

  a = mx::broadcast_to(mx::random::uniform({size, 1}), shape);
  TIMEM("2D broadcast dim 1", mx::astype, a, mx::int32, device);
}

int main(int argc, char** argv) {
  if (argc > 1) {
    bool use_gpu = !strcmp(argv[1], "gpu");
    set_default_device(use_gpu ? mx::Device::gpu : mx::Device::cpu);
  }
  std::cout << "Benchmarks for " << mx::default_device() << std::endl;
  time_irregular_binary_ops_1D();
  time_irregular_binary_ops_2D();
  time_irregular_binary_ops_3D();
  time_irregular_binary_ops_4D();
  time_irregular_reshape();
  time_irregular_astype_1D();
  time_irregular_astype_2D();
}
