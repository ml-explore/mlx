// Copyright © 2026 Apple Inc.

#include <vector>

#include "doctest/doctest.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test metal all reduce min max dispatch") {
  auto s = new_stream(Device::gpu);
  auto& d = metal::device(s.device);
  auto& encoder = metal::get_command_encoder(s);
  array out_min({}, float32, nullptr, {});
  array out_max({}, float32, nullptr, {});
  out_min.set_data(allocator::malloc(out_min.nbytes()));
  out_max.set_data(allocator::malloc(out_max.nbytes()));

  for (int size :
       {1,
        31,
        33,
        REDUCE_N_READS * 1024,
        REDUCE_N_READS * 1024 + 1,
        100003,
        (1 << 24) + 1}) {
    std::vector<float> values(size, 2.0f);
    values[0] = -7.0f;
    values[size - 1] = 11.0f;
    auto in = array(values.data(), {size});
    eval(in);
    all_reduce_min_max_dispatch(in, out_min, out_max, encoder, d, s);
    encoder.synchronize();
    CHECK(out_min.data<float>()[0] == (size == 1 ? 11.0f : -7.0f));
    CHECK(out_max.data<float>()[0] == 11.0f);
  }

  array empty({0}, float32, nullptr, {});
  auto wrong_dtype = array(1);
  auto in = array({1.0f, 2.0f});
  auto unallocated = array({}, float32, nullptr, {});
  CHECK_THROWS_AS(
      all_reduce_min_max_dispatch(empty, out_min, out_max, encoder, d, s),
      std::invalid_argument);
  CHECK_THROWS_AS(
      all_reduce_min_max_dispatch(wrong_dtype, out_min, out_max, encoder, d, s),
      std::invalid_argument);
  CHECK_THROWS_AS(
      all_reduce_min_max_dispatch(in, unallocated, out_max, encoder, d, s),
      std::invalid_argument);
  CHECK_THROWS_AS(
      all_reduce_min_max_dispatch(in, out_min, out_min, encoder, d, s),
      std::invalid_argument);
}
