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

namespace {

void gated_delta_update_forward_metal(
  const Stream& s,
  metal::Device& d){
			
	std::string base_name = "gated_delta_update_fwd_float_64_64";
  std::string hash_name = base_name;
  metal::MTLFCList func_consts = {};

  auto kernel = get_steel_gated_delta_forward_kernel(
        d,
        base_name,
        hash_name,
        func_consts);
}
    


}

bool GatedDeltaUpdate::use_fallback(Stream s) {
    // TODO: finish implementation
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

		out.set_data(allocator::malloc(out.nbytes()));
    hf.set_data(allocator::malloc(hf.nbytes()));

    gated_delta_update_forward_metal(s,d);
    // throw std::runtime_error("NYI");
}

bool GatedDeltaUpdate::is_equivalent(const Primitive& other) const {
    const auto* p = dynamic_cast<const GatedDeltaUpdate*>(&other);
    if (p == nullptr) {
        return false;
    }
    // TODO: finish implementation
    return true;
}

} // namespace mlx::core::fast
