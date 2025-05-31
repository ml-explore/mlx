// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/paged_attention_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::paged_attention {

void PagedAttention::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& q = inputs[0];
  auto& k_cache = inputs[1];
  auto& v_cache = inputs[2];
  auto& block_tables = inputs[3];
  auto& context_lens = inputs[4];
  const auto alibi_slopes =
      inputs.size() == 6 ? std::optional{inputs[5]} : std::nullopt;
  return;
}
} // namespace mlx::core::paged_attention