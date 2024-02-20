// Copyright © 2023 Apple Inc.

#include <climits>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test array basics") {
  // Scalar
  array x(1.0);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.ndim(), 0);
  CHECK_EQ(x.shape(), std::vector<int>{});
  CHECK_THROWS_AS(x.shape(0), std::out_of_range);
  CHECK_THROWS_AS(x.shape(-1), std::out_of_range);
  CHECK_EQ(x.strides(), std::vector<size_t>{});
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), sizeof(float));
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.item<float>(), 1.0);

  // Scalar with specified type
  x = array(1, float32);
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.item<float>(), 1.0);

  // Scalar with specified type
  x = array(1, bool_);
  CHECK_EQ(x.dtype(), bool_);
  CHECK_EQ(x.itemsize(), sizeof(bool));
  CHECK_EQ(x.nbytes(), sizeof(bool));
  CHECK_EQ(x.item<bool>(), true);

  // Check shaped arrays
  x = array({1.0});
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.ndim(), 1);
  CHECK_EQ(x.shape(), std::vector<int>{1});
  CHECK_EQ(x.shape(0), 1);
  CHECK_EQ(x.shape(-1), 1);
  CHECK_THROWS_AS(x.shape(1), std::out_of_range);
  CHECK_THROWS_AS(x.shape(-2), std::out_of_range);
  CHECK_EQ(x.strides(), std::vector<size_t>{1});
  CHECK_EQ(x.item<float>(), 1.0);

  // Check empty array
  x = array({});
  CHECK_EQ(x.size(), 0);
  CHECK_EQ(x.dtype(), float32);
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), 0);
  CHECK_THROWS_AS(x.item<float>(), std::invalid_argument);

  x = array({1.0, 1.0});
  CHECK_EQ(x.size(), 2);
  CHECK_EQ(x.shape(), std::vector<int>{2});
  CHECK_EQ(x.itemsize(), sizeof(float));
  CHECK_EQ(x.nbytes(), x.itemsize() * x.size());

  // Accessing item in non-scalar array throws
  CHECK_THROWS_AS(x.item<float>(), std::invalid_argument);

  x = array({1.0, 1.0, 1.0}, {1, 3});
  CHECK(x.size() == 3);
  CHECK(x.shape() == std::vector<int>{1, 3});
  CHECK(x.strides() == std::vector<size_t>{3, 1});

  // Test wrong size/shapes throw:
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {4}), std::invalid_argument);
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {1, 4}), std::invalid_argument);
  CHECK_THROWS_AS(array({1.0, 1.0, 1.0}, {1, 2}), std::invalid_argument);

  // Test array ids work as expected
  x = array(1.0);
  auto y = x;
  CHECK_EQ(y.id(), x.id());
  array z(2.0);
  CHECK_NE(z.id(), x.id());
  z = x;
  CHECK_EQ(z.id(), x.id());

  // Array creation from pointer
  float data[] = {0.0, 1.0, 2.0, 3.0};
  x = array(data, {4});
  CHECK_EQ(x.dtype(), float32);
  CHECK(array_equal(x, array({0.0, 1.0, 2.0, 3.0})).item<bool>());

  // Array creation from vectors
  {
    std::vector<int> data = {0, 1, 2, 3};
    x = array(data.begin(), {4});
    CHECK_EQ(x.dtype(), int32);
    CHECK(array_equal(x, array({0, 1, 2, 3})).item<bool>());
  }

  {
    std::vector<bool> data = {false, true, false, true};
    x = array(data.begin(), {4});
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, true, false, true})).item<bool>());
  }
}

TEST_CASE("test array types") {
#define basic_dtype_test(T, mlx_type) \
  T val = 42;                         \
  array x(val);                       \
  CHECK_EQ(x.dtype(), mlx_type);      \
  CHECK_EQ(x.item<T>(), val);         \
  x = array({val, val});              \
  CHECK_EQ(x.dtype(), mlx_type);

  // bool_
  {
    array x(true);
    CHECK_EQ(x.dtype(), bool_);
    CHECK_EQ(x.item<bool>(), true);

    x = array({true, false});
    CHECK_EQ(x.dtype(), bool_);

    x = array({true, false}, float32);
    CHECK_EQ(x.dtype(), float32);
    CHECK(array_equal(x, array({1.0f, 0.0f})).item<bool>());
  }

  // uint8
  { basic_dtype_test(uint8_t, uint8); }

  // uint16
  { basic_dtype_test(uint16_t, uint16); }

  // uint32
  { basic_dtype_test(uint32_t, uint32); }

  // uint64
  { basic_dtype_test(uint64_t, uint64); }

  // int8
  { basic_dtype_test(int8_t, int8); }

  // int16
  { basic_dtype_test(int16_t, int16); }

  // int32
  { basic_dtype_test(int32_t, int32); }

  // int64
  { basic_dtype_test(int64_t, int64); }

  // float16
  { basic_dtype_test(float16_t, float16); }

  // float32
  { basic_dtype_test(float, float32); }

  // bfloat16
  { basic_dtype_test(bfloat16_t, bfloat16); }

  // uint32
  {
    uint32_t val = UINT_MAX;
    array x(val);
    CHECK_EQ(x.dtype(), uint32);
    CHECK_EQ(x.item<uint32_t>(), val);

    x = array({1u, 2u});
    CHECK_EQ(x.dtype(), uint32);
  }

  // int32
  {
    array x(-1);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(x.item<int>(), -1);

    x = array({-1, 2});
    CHECK_EQ(x.dtype(), int32);

    std::vector<int> data{0, 1, 2};
    x = array(data.data(), {static_cast<int>(data.size())}, bool_);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, true, true})).item<bool>());
  }

  // int64
  {
    int64_t val = static_cast<int64_t>(INT_MIN) - 1;
    array x(val);
    CHECK_EQ(x.dtype(), int64);
    CHECK_EQ(x.item<int64_t>(), val);

    x = array({val, val});
    CHECK_EQ(x.dtype(), int64);
  }

  // float32
  {
    array x(3.14f);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 3.14f);

    x = array(1.25);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 1.25f);

    x = array({1.0f, 2.0f});
    CHECK_EQ(x.dtype(), float32);

    x = array({1.0, 2.0});
    CHECK_EQ(x.dtype(), float32);

    std::vector<double> data{1.0, 2.0, 4.0};
    x = array(data.data(), {static_cast<int>(data.size())});
    CHECK_EQ(x.dtype(), float32);
    CHECK(array_equal(x, array({1.0f, 2.0f, 4.0f})).item<bool>());
  }

  // complex64
  {
    CHECK_EQ(sizeof(complex64_t), sizeof(std::complex<float>));

    complex64_t v = {1.0f, 1.0f};
    array x(v);
    CHECK_EQ(x.dtype(), complex64);
    CHECK_EQ(x.item<complex64_t>(), v);

    array y(std::complex<float>{1.0f, 1.0f});
    CHECK_EQ(x.dtype(), complex64);
    CHECK_EQ(x.item<complex64_t>(), v);
  }

#undef basic_dtype_test

#define basic_dtype_str_test(s, dtype)         \
  CHECK_EQ(s, dtype_to_array_protocol(dtype)); \
  CHECK_EQ(dtype_from_array_protocol(s), dtype);

  // To and from str
  {
    basic_dtype_str_test("|b1", bool_);
    basic_dtype_str_test("|u1", uint8);
    basic_dtype_str_test("<u2", uint16);
    basic_dtype_str_test("<u4", uint32);
    basic_dtype_str_test("<u8", uint64);
    basic_dtype_str_test("|i1", int8);
    basic_dtype_str_test("<i2", int16);
    basic_dtype_str_test("<i4", int32);
    basic_dtype_str_test("<i8", int64);
    basic_dtype_str_test("<f2", float16);
    basic_dtype_str_test("<f4", float32);
    basic_dtype_str_test("<V2", bfloat16);
    basic_dtype_str_test("<c8", complex64);
  }

#undef basic_dtype_str_test
}

TEST_CASE("test array metadata") {
  array x(1.0f);
  CHECK_EQ(x.data_size(), 1);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = array({1.0f}, {1, 1, 1});
  CHECK_EQ(x.data_size(), 1);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = array({1.0f, 1.0f}, {1, 2});
  CHECK_EQ(x.data_size(), 2);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = zeros({1, 1, 4});
  eval(x);
  CHECK_EQ(x.data_size(), 4);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, true);

  x = zeros({2, 4});
  eval(x);
  CHECK_EQ(x.data_size(), 8);
  CHECK_EQ(x.flags().contiguous, true);
  CHECK_EQ(x.flags().row_contiguous, true);
  CHECK_EQ(x.flags().col_contiguous, false);

  x = array(1.0f);
  auto y = broadcast_to(x, {1, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = broadcast_to(x, {2, 8, 10});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = broadcast_to(x, {1, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = broadcast_to(zeros({4, 2, 1}), {4, 2, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array(1.0f);
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x, {0, 1, 2});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({1, 1, 1});
  y = transpose(x, {1, 2, 0});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({4, 1});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 3, 4});
  y = transpose(x);
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = transpose(x, {0, 2, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = transpose(transpose(x, {0, 2, 1}), {0, 2, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 24);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = array(1.0f);
  y = reshape(x, {1, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 4});
  y = reshape(x, {8});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = reshape(x, {8, 1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = reshape(x, {1, 8, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 8);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({12});
  y = reshape(x, {2, 3, 2});
  eval(y);
  CHECK_EQ(y.data_size(), 12);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = array(1.0f);
  y = slice(x, {}, {});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f});
  y = slice(x, {-10}, {10}, {10});
  eval(y);
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 3);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 3);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {0, 3}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 0);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({1.0f, 2.0f, 3.0f}, {1, 3});
  y = slice(x, {0, 0}, {1, 2}, {2, 3});
  eval(y);
  CHECK_EQ(y.shape(), std::vector<int>{1, 1});
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({0.0f, 1.0f, 2.0f, 3.0f}, {1, 4});
  y = slice(x, {0, 0}, {1, 4}, {1, 2});
  eval(y);
  CHECK_EQ(y.shape(), std::vector<int>{1, 2});
  CHECK_EQ(y.flags().contiguous, false);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = broadcast_to(array(1.0f), {4, 10});
  y = slice(x, {0, 0}, {4, 10}, {2, 2});
  eval(y);
  CHECK_EQ(y.shape(), std::vector<int>{2, 5});
  CHECK_EQ(y.data_size(), 1);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, false);

  x = broadcast_to(array({1.0f, 2.0f}), {4, 2});
  y = slice(x, {0, 0}, {1, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  y = slice(x, {1, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 2);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
  y = slice(x, {0, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, true);
  CHECK_EQ(y.flags().col_contiguous, false);

  y = slice(transpose(x), {0, 0}, {2, 2}, {1, 1});
  eval(y);
  CHECK_EQ(y.data_size(), 4);
  CHECK_EQ(y.flags().contiguous, true);
  CHECK_EQ(y.flags().row_contiguous, false);
  CHECK_EQ(y.flags().col_contiguous, true);

  x = ones({2, 4});
  auto out = split(x, 2);
  eval(out);
  for (auto y : out) {
    CHECK_EQ(y.data_size(), 4);
    CHECK_EQ(y.flags().contiguous, true);
    CHECK_EQ(y.flags().row_contiguous, true);
    CHECK_EQ(y.flags().col_contiguous, true);
  }
  out = split(x, 4, 1);
  eval(out);
  for (auto y : out) {
    CHECK_EQ(y.flags().contiguous, false);
    CHECK_EQ(y.flags().row_contiguous, false);
    CHECK_EQ(y.flags().col_contiguous, false);
  }
}

TEST_CASE("test array iteration") {
  // Dim 0 arrays
  auto arr = array(1);
  CHECK_THROWS(arr.begin());

  // Iterated arrays are read only
  CHECK(std::is_const_v<decltype(*arr.begin())>);

  arr = array({1, 2, 3, 4, 5});
  int i = 0;
  for (auto a : arr) {
    i++;
    CHECK_EQ(a.item<int>(), i);
  }
  CHECK_EQ(i, 5);

  arr = array({1, 2, 3, 4}, {2, 2});
  CHECK(array_equal(*arr.begin(), array({1, 2})).item<bool>());
  CHECK(array_equal(*(arr.begin() + 1), array({3, 4})).item<bool>());
  CHECK_EQ(arr.begin() + 2, arr.end());
}

TEST_CASE("test array shared buffer") {
  std::vector<int> shape = {2, 2};
  int n_elem = shape[0] * shape[1];

  allocator::Buffer buf_b = allocator::malloc(n_elem * sizeof(float));
  void* buf_b_ptr = buf_b.raw_ptr();
  float* float_buf_b = (float*)buf_b_ptr;

  for (int i = 0; i < n_elem; i++) {
    float_buf_b[i] = 2.;
  }

  CHECK_EQ(float_buf_b[0], ((float*)buf_b_ptr)[0]);

  auto deleter = [float_buf_b](allocator::Buffer buf) {
    CHECK_EQ(float_buf_b, (float*)buf.raw_ptr());
    CHECK_EQ(float_buf_b[0], ((float*)buf.raw_ptr())[0]);
    allocator::free(buf);
  };

  array a = ones(shape, float32);
  array b = array(buf_b, shape, float32, deleter);

  eval(a + b);
}

TEST_CASE("test make empty array") {
  auto a = array({});
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), float32);

  a = array({}, int32);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), int32);

  a = array({}, float32);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), float32);

  a = array({}, bool_);
  CHECK_EQ(a.size(), 0);
  CHECK_EQ(a.dtype(), bool_);
}
