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
    
    int B  = q.shape(0);
		int T  = q.shape(1);
		int Hk = q.shape(2);
		int Dk = q.shape(3);
		int Hv = v.shape(2);
		int Dv = v.shape(3);

		out.set_data(allocator::malloc(out.nbytes()));
    hf.set_data(allocator::malloc(hf.nbytes()));
    
		std::string base_name = "gated_delta_step_"
				+ get_type_string(q.dtype())          // "float"
				+ "_" + get_type_string(h0.dtype()) // "float"
				+ "_" + std::to_string(Dk)
				+ "_" + std::to_string(Dv)
				+ "_" + std::to_string(Hk)
				+ "_" + std::to_string(Hv);
		// e.g. "gated_delta_step_float_float_64_64_4_4"
		std::string hash_name = base_name;
		metal::MTLFCList func_consts = {};

		auto kernel = get_steel_gated_delta_forward_kernel(
					d,
					base_name,
					hash_name,
					func_consts);
    
    auto& compute_encoder = metal::get_command_encoder(s);

    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q,     0);
    compute_encoder.set_input_array(k,     1);
    compute_encoder.set_input_array(v,     2);
    compute_encoder.set_input_array(g,     3);
    compute_encoder.set_input_array(beta,  4);
    compute_encoder.set_input_array(h0,		 5);
    compute_encoder.set_bytes(T,           6);
    compute_encoder.set_output_array(out,  7);
    compute_encoder.set_output_array(hf,   8);

    // auto grid   = MTL::Size(1, 1, 1);
    // auto threads = MTL::Size(1, 1, 1);
    auto grid   = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
    compute_encoder.dispatch_threads(grid, threads);
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
