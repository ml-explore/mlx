// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/config.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test command buffer counts new memory, not inputs") {
  auto s = default_stream(Device::gpu);
  // Only the 1 MB limit can end the buffer. With int8, elements equal bytes.
  auto limits = config::scoped_update(
      {{"MLX_MAX_MB_PER_BUFFER", 1}, {"MLX_MAX_OPS_PER_BUFFER", 1000}});
  auto x = zeros({2 << 20}, int8);
  eval(x);
  synchronize(s);

  auto& encoder = metal::get_command_encoder(s);
  // Reading an existing array adds no new memory.
  encoder.set_input_array(x, 0);
  CHECK_FALSE(encoder.needs_commit());

  // A new temporary of the same size does.
  array temp(x.shape(), int8, nullptr, {});
  temp.set_data(allocator::malloc(temp.nbytes()));
  encoder.add_temporary(temp);
  CHECK(encoder.needs_commit());
  synchronize(s);
}
