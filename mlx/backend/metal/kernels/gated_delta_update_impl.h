#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void make_wy(
    const device InT* k          					[[buffer(0)]],
    const device InT* v          					[[buffer(1)]],
    const device InT* g          					[[buffer(2)]],   // [B, T, Hv] or [B, T, Hv, Dk]
    const device InT* beta   						[[buffer(3)]],   // [B, Hv, Dv, Dk]
	device InT* W      								[[buffer(4)]],   // [B, T, Hv]
	device InT* U	      							[[buffer(5)]],   // [B, T, Hv]
	constant int& T              					[[buffer(6)]],
    uint3 thread_position_in_grid        	[[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup 	[[thread_position_in_threadgroup]],
    uint  thread_index_in_simdgroup      	[[thread_index_in_simdgroup]]
) {
	auto n        = thread_position_in_grid.z;
	auto n_chunks = T / C;
	auto chunk    = n % n_chunks;
	auto bh_idx   = n / n_chunks;
	auto b_idx    = bh_idx / Hv;
	auto hv_idx   = bh_idx % Hv;
	auto hk_idx   = hv_idx / (Hv / Hk);

	constexpr int n_per_dk = Dk / 32;
	constexpr int n_per_dv = Dv / 32;

	auto dk_idx        = thread_position_in_threadgroup.x;   
	auto simd_group_id = thread_position_in_threadgroup.y;   
	const int num_simdgroups = 4;

	int offset_t = chunk * C;
	auto g_ = g + b_idx * T * Hv + offset_t * Hv;
	auto k_ = k + b_idx * T * Hk * Dk + offset_t * Hk * Dk + hk_idx * Dk;
	auto v_ = v + b_idx * T * Hv * Dv + offset_t * Hv * Dv + hv_idx * Dv;
	auto beta_ = beta + b_idx * T * Hv + offset_t * Hv;
	auto W_ = W + b_idx * T * Hv * Dk + offset_t * Hv * Dk + hv_idx * Dk;
	auto U_ = U + b_idx * T * Hv * Dv + offset_t * Hv * Dv + hv_idx * Dv;

	// threadgroup memory
	threadgroup float K_tg[C][Dk];
	threadgroup float KKt[C][C];

	for (int i = simd_group_id; i < C; i += num_simdgroups) {
		for (int d = dk_idx; d < Dk; d += 32) {
			K_tg[i][d] = k_[i * Hk * Dk + d];
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// compute gamma (cumprod)
	float gamma[C];
	gamma[0] = g_[hv_idx];
	for (int i = 1; i < C; i++) {
		gamma[i] = gamma[i-1] * g_[i * Hv + hv_idx];
	}

	// compute KKt = K @ K.T
	unsigned int counter = 0;
	for (int i = 0; i < C; i++) {
		for (int j = 0; j <= i; j++) {
			if (counter % num_simdgroups == simd_group_id) {
				float sum = 0;
				for (int d = dk_idx; d < Dk; d += 32) {
					sum += K_tg[i][d] * K_tg[j][d];
				}
				sum = simd_sum(sum);
				if (thread_index_in_simdgroup == 0) {
					KKt[i][j] = sum;
				}
			}
			counter++;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	float u[C][n_per_dv];  
	float w[C][n_per_dk];

	// initialize
	for (int i = 0; i < C; i++) {
		float beta_i = beta_[i * Hv + hv_idx];
		for (int p = 0; p < n_per_dv; p++) {
			int dv = dk_idx * n_per_dv + p;
			u[i][p] = beta_i * v_[i * Hv * Dv + dv]; 
		}
		for (int p = 0; p < n_per_dk; p++) {
			int dk = dk_idx * n_per_dk + p;
			w[i][p] = beta_i * K_tg[i][dk];
		}
	}


	// forward substitution
	for (int i = 1; i < C; i++) {
		float beta_i = beta_[i * Hv + hv_idx];
		for (int j = 0; j < i; j++) {
			float a_U = beta_i * (gamma[i] / gamma[j]) * KKt[i][j];
			float a_W = beta_i * KKt[i][j];
			for (int p = 0; p < n_per_dv; p++) {
				u[i][p] -= a_U * u[j][p];
			}
			for (int p = 0; p < n_per_dk; p++) {
				w[i][p] -= a_W * w[j][p];
			}
		}
	}

	for (int i = 0; i < C; i++) {
		for (int p = 0; p < n_per_dv; p++) {
			int dv = dk_idx * n_per_dv + p;
			U_[i * Hv * Dv + dv] = u[i][p];   
		}
		for (int p = 0; p < n_per_dk; p++) {
			int dk = dk_idx * n_per_dk + p;
			W_[i * Hk * Dk + dk] = w[i][p];   
		}
	}
}

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_chunk(
    const device InT* q          					[[buffer(0)]],
    const device InT* k          					[[buffer(1)]],
    const device InT* W          					[[buffer(2)]],	 // [B, T, Hv, Dk]
    const device InT* U          					[[buffer(3)]],   // [B, T, Hv, Dv]
    const device StT* state_in   					[[buffer(4)]],   // [B, Hv, Dv, Dk]
	const device InT* g      						[[buffer(5)]],   // [B, T, Hv]
    device InT* y                					[[buffer(6)]],   // [B, T, Hv, Dv]
    device StT* state_out        					[[buffer(7)]],   // [B, Hv, Dv, Dk]
	constant int& T              					[[buffer(8)]],
    uint3 thread_position_in_grid        	[[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup 	[[thread_position_in_threadgroup]],
    uint  thread_index_in_simdgroup      	[[thread_index_in_simdgroup]]
) {
    auto n        = thread_position_in_grid.z;
	auto b_idx    = n / Hv;
	auto hv_idx   = n % Hv;
	auto hk_idx   = hv_idx / (Hv / Hk);
	constexpr int n_per_dk = Dk / 32;

	auto dk_idx = thread_position_in_threadgroup.x;   // simd group id
	auto dv_idx = thread_position_in_grid.y;

	// set up pointers
	// g: [B, T, Hv]
	auto g_ = g + b_idx * T * Hv;

	// q, k: [B, T, Hk, Dk]
	auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
	auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

	// W: [B, T, Hk, Dk], U: [B, T, Hv, Dv]
	auto W_ = W + b_idx * T * Hv * Dk + hv_idx * Dk;
	auto U_ = U + b_idx * T * Hv * Dv + hv_idx * Dv;

	// v, y: [B, T, Hv, Dv]
	y += b_idx * T * Hv * Dv + hv_idx * Dv;

	// state_in, state_out: [B, Hv, Dv, Dk]
	auto i_state = state_in + (n * Dv + dv_idx) * Dk;
	auto o_state = state_out + (n * Dv + dv_idx) * Dk;

	float state[n_per_dk];
	for (int i = 0; i < n_per_dk; ++i) {
		auto s_idx = n_per_dk * dk_idx + i;
		state[i] = static_cast<float>(i_state[s_idx]);
	}

	float gamma[C];
	float delta[C];

	for (int t = 0; t < T; t+=C) {
		gamma[0] = g_[hv_idx];
		for (int i = 1; i < C; i++) {
			gamma[i] = gamma[i-1] * g_[i * Hv + hv_idx];
		}
		float gamma_C = gamma[C-1];

		for (int c = 0; c < C; c++) {
			// W_left[c] @ S^T
			float ws = 0;
			float g_c = gamma[c];
			for (int i = 0; i < n_per_dk; i++) {
				int s_idx = dk_idx * n_per_dk + i;
				float w_left = g_c * W_[c * Hk * Dk + s_idx];
				ws += w_left * state[i];
			}
			ws = simd_sum(ws);   

			// delta = U - ws
			delta[c] = U_[c * Hv * Dv + dv_idx] - ws;
		}
		

		// Compute Output: O = Q_left S^T + Q K^T delta
		for (int c = 0; c < C; c++) {
			// Q_left[c] S^T
			float out_c = 0;
			float g_c = gamma[c];
			for (int i = 0; i < n_per_dk; i++) {
				int s_idx = dk_idx * n_per_dk + i;
				float q_left = g_c * q_[c * Hk * Dk + s_idx];
				out_c += q_left * state[i];
			}
			out_c = simd_sum(out_c);   

			// Q K^T delta
			for (int j = 0; j <= c; j++) {
				float qkt = 0;
				for (int i = 0; i < n_per_dk; i++) {
					int s_idx = dk_idx * n_per_dk + i;
					qkt += q_[c * Hk * Dk + s_idx] * k_[j * Hk * Dk + s_idx];
				}
				qkt = simd_sum(qkt); 
				out_c += (gamma[c] / gamma[j]) * qkt * delta[j];
			}

			if (thread_index_in_simdgroup == 0) {
				y[c * Hv * Dv + dv_idx] = out_c;
			}
		}

		// Update state: S = gamma_C * S + delta.T @ K
		// gamma_C * S
		for (int i = 0; i < n_per_dk; i++) {
		    state[i] *= gamma_C;  
		}

		// delta.T @ K
		for (int c = 0; c < C; c++) {
			float k_right = (gamma_C / gamma[c]);   
			for (int i = 0; i < n_per_dk; i++) {
				int s_idx = dk_idx * n_per_dk + i;
				state[i] += k_right * k_[c * Hk * Dk + s_idx] * delta[c];
			}
		}

		// update pointers
		q_ += C * Hk * Dk;
		k_ += C * Hk * Dk;
		U_ += C * Hv * Dv;
		W_ += C * Hv * Dk;
		y  += C * Hv * Dv;
		g_ += C * Hv;
	}
	for (int i = 0; i < n_per_dk; ++i) {
		auto s_idx = n_per_dk * dk_idx + i;
		o_state[s_idx] = static_cast<StT>(state[i]);
	}
}

/*
	auto grid   = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
 */
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_seq(
    const device InT* q          					[[buffer(0)]],
    const device InT* k          					[[buffer(1)]],
    const device InT* v          					[[buffer(2)]],
    const device InT* g          					[[buffer(3)]],   // [B, T, Hv] or [B, T, Hv, Dk]
    const device InT* beta      					[[buffer(4)]],   // [B, T, Hv]
    const device StT* state_in   					[[buffer(5)]],   // [B, Hv, Dv, Dk]
    constant int& T              					[[buffer(6)]],
    device InT* y                					[[buffer(7)]],   // [B, T, Hv, Dv]
    device StT* state_out        					[[buffer(8)]],   // [B, Hv, Dv, Dk]
    uint3 thread_position_in_grid        	[[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup 	[[thread_position_in_threadgroup]],
    uint  thread_index_in_simdgroup      	[[thread_index_in_simdgroup]]
) {
    // kernel implementation
    auto n = thread_position_in_grid.z;
	auto b_idx = n / Hv;
	auto hv_idx = n % Hv;
	auto hk_idx = hv_idx / (Hv / Hk);
	constexpr int n_per_t = Dk / 32;

	// q, k: [B, T, Hk, Dk]
	auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
	auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

	// v, y: [B, T, Hv, Dv]
	auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
	y += b_idx * T * Hv * Dv + hv_idx * Dv;

	auto dk_idx = thread_position_in_threadgroup.x;
	auto dv_idx = thread_position_in_grid.y;

	// state_in, state_out: [B, Hv, Dv, Dk]
	auto i_state = state_in + (n * Dv + dv_idx) * Dk;
	auto o_state = state_out + (n * Dv + dv_idx) * Dk;

	float state[n_per_t];
	for (int i = 0; i < n_per_t; ++i) {
		auto s_idx = n_per_t * dk_idx + i;
		state[i] = static_cast<float>(i_state[s_idx]);
	}

	// g: [B, T, Hv]
	auto g_ = g + b_idx * T * Hv;
	auto beta_ = beta + b_idx * T * Hv;

	for (int t = 0; t < T; ++t) {
		float kv_mem = 0.0f;
		for (int i = 0; i < n_per_t; ++i) {
			auto s_idx = n_per_t * dk_idx + i;
			state[i] = state[i] * g_[hv_idx];
			kv_mem += state[i] * k_[s_idx];
		}
		kv_mem = simd_sum(kv_mem);

		auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

		float out = 0.0f;
		for (int i = 0; i < n_per_t; ++i) {
			auto s_idx = n_per_t * dk_idx + i;
			state[i] = state[i] + k_[s_idx] * delta;
			out += state[i] * q_[s_idx];
		}
		out = simd_sum(out);
		if (thread_index_in_simdgroup == 0) {
			y[dv_idx] = static_cast<InT>(out);
		}
		// Increment data pointers to next time step
		q_ += Hk * Dk;
		k_ += Hk * Dk;
		v_ += Hv * Dv;
		y += Hv * Dv;
		g_ += Hv;
		beta_ += Hv;
	}
	for (int i = 0; i < n_per_t; ++i) {
		auto s_idx = n_per_t * dk_idx + i;
		o_state[s_idx] = static_cast<StT>(state[i]);
	}
}
