// Copyright © 2024 Apple Inc.
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

bool GatedDeltaUpdate::use_fallback(Stream s) {
    // always run on GPU for now
    return false;
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {

    auto& s = stream();
    auto& d = metal::device(s.device);

    auto& q   = inputs[0];
    auto& k   = inputs[1];
    auto& v   = inputs[2];
    auto& g   = inputs[3];
    auto& beta = inputs[4];
    auto& h0  = inputs[5];

    auto& out = outputs[0];
    auto& hf  = outputs[1];

    // TODO: allocate outputs, dispatch Metal kernel
    throw std::runtime_error("NYI");
}

bool GatedDeltaUpdate::is_equivalent(const Primitive& other) const {
    const auto* p = dynamic_cast<const GatedDeltaUpdate*>(&other);
    if (p == nullptr) {
        return false;
    }
    // TODO: compare chunk_size and other state fields once added
    return true;
}

} // namespace mlx::core::fast
