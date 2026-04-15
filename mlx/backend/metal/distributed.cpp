// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"
#include "mlx/types/half_types.h"

namespace mlx::core::distributed {

// Forward declare from moe.cpp
MTL::ComputePipelineState*
get_moe_kernel(metal::Device& d, const std::string& base_name, Dtype dtype);

void AllReduce::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllReduce::eval_gpu] has no GPU implementation.");
}

void AllGather::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllGather::eval_gpu] has no GPU implementation.");
}

void Send::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Send::eval_gpu] has no GPU implementation.");
}

void Recv::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Recv::eval_gpu] has no GPU implementation.");
}

void ReduceScatter::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[ReduceScatter::eval_gpu] has no GPU implementation.");
}

void AllToAll::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllToAll::eval_gpu] has no GPU implementation.");
}

// ---------------------------------------------------------------------------
// MoeDispatchExchange::eval_gpu
// ---------------------------------------------------------------------------
//
// Architecture:
//   CPU = O(N*top_k) routing only (route build, meta decode, exchange)
//   GPU = O(N*D) data movement (scatter, gather via Metal kernels)
//
// Flows:
//   ws==1: synchronize -> CPU route -> GPU zero-fill + dispatch_local
//   ws==2: synchronize -> CPU route -> GPU zero-fill + dispatch_local +
//          packet_gather -> mid-sync -> CPU exchange_v -> CPU decode ->
//          GPU packet_scatter
//   ws>2:  CPU fallback (create CPU-stream primitive)
// ---------------------------------------------------------------------------

void MoeDispatchExchange::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  try {
    // Fault injection (test-only)
    if (std::getenv("MLX_MOE_EP_FORCE_METAL_ERROR")) {
      throw std::runtime_error("[MoE EP] Forced Metal error for testing");
    }

    assert(inputs.size() == 2);
    assert(outputs.size() == 2);

    auto& s = stream();
    auto& d = metal::device(s.device);
    int world_size = group().size();

    // ws > 2: fallback to CPU eval
    if (world_size > 2) {
      MoeDispatchExchange cpu_fb(
          to_stream(std::monostate{}, Device::cpu),
          group(),
          num_experts_,
          capacity_,
          deterministic_,
          MoeBackend::Cpu);
      cpu_fb.eval_cpu(inputs, outputs);
      return;
    }

    // Ensure inputs are evaluated on GPU
    auto& tokens_in = inputs[0];
    auto& indices_in = inputs[1];

    int N = tokens_in.shape(0);
    int D_val = tokens_in.shape(1);
    int top_k = indices_in.shape(1);
    int num_experts = num_experts_;
    int capacity = capacity_;
    int experts_per_device = num_experts / std::max(world_size, 1);
    int cap_total = world_size * capacity;
    size_t elem_size = tokens_in.itemsize();
    Dtype dtype = tokens_in.dtype();

    // Allocate output arrays
    // outputs[0]: dispatched [E_local, cap_total, D]
    // outputs[1]: route_indices [N, top_k] int32
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));

    // Step 1: gpu::synchronize to flush upstream GPU ops so CPU can read
    // expert_indices via UMA
    gpu::synchronize(s);

    // CPU: read expert_indices via UMA
    const int32_t* idx_ptr = indices_in.data<int32_t>();
    int32_t* route_ptr = outputs[1].data<int32_t>();

    // Initialize route_indices to -1 (CPU, O(N*top_k))
    std::fill(route_ptr, route_ptr + N * top_k, int32_t(-1));

    // Zero-fill dispatched output via CPU memset (UMA, after synchronize)
    size_t out_nbytes =
        static_cast<size_t>(experts_per_device) * cap_total * D_val * elem_size;
    std::memset(outputs[0].data<void>(), 0, out_nbytes);

    // Expert count tracking for routing
    std::vector<int> expert_counts(num_experts, 0);

    // =========================================================================
    // ws == 1: local-only path
    // =========================================================================
    if (world_size == 1) {
      // Build slot_map and nk_indices for valid assignments
      std::vector<int32_t> slot_map_vec;
      std::vector<int32_t> nk_indices_vec;
      slot_map_vec.reserve(N * top_k);
      nk_indices_vec.reserve(N * top_k);

      // k-outer, n-inner for deterministic slot assignment
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_ptr[n * top_k + k];
          if (eid < 0 || eid >= num_experts)
            continue;
          int pos = expert_counts[eid]++;
          if (pos >= capacity)
            continue;

          int flat_idx = eid * cap_total + pos;
          route_ptr[n * top_k + k] = flat_idx;

          slot_map_vec.push_back(flat_idx);
          nk_indices_vec.push_back(n * top_k + k);
        }
      }

      int valid_count = static_cast<int>(slot_map_vec.size());
      if (valid_count == 0) {
        // No valid assignments — output is already zero-filled
        return;
      }

      // Build GPU metadata buffers (UMA zero-copy)
      array slot_map_buf({valid_count}, int32, nullptr, {});
      slot_map_buf.set_data(allocator::malloc(valid_count * sizeof(int32_t)));
      std::memcpy(
          slot_map_buf.data<int32_t>(),
          slot_map_vec.data(),
          valid_count * sizeof(int32_t));

      array nk_indices_buf({valid_count}, int32, nullptr, {});
      nk_indices_buf.set_data(allocator::malloc(valid_count * sizeof(int32_t)));
      std::memcpy(
          nk_indices_buf.data<int32_t>(),
          nk_indices_vec.data(),
          valid_count * sizeof(int32_t));

      // Launch moe_dispatch_local kernel
      auto kernel = get_moe_kernel(d, "moe_dispatch_local", dtype);
      auto& enc = d.get_command_encoder(s.index);
      enc.set_compute_pipeline_state(kernel);
      enc.set_input_array(tokens_in, 0);
      enc.set_output_array(outputs[0], 1);
      enc.set_input_array(slot_map_buf, 2);
      enc.set_input_array(nk_indices_buf, 3);
      enc.set_bytes(D_val, 4);
      enc.set_bytes(top_k, 5);

      int tx = std::min(D_val, 256);
      MTL::Size grid_dims = MTL::Size(D_val, valid_count, 1);
      MTL::Size group_dims = MTL::Size(tx, 1, 1);
      enc.dispatch_threads(grid_dims, group_dims);

      // Keep temporaries alive until command buffer commits
      d.add_temporary(slot_map_buf, s.index);
      d.add_temporary(nk_indices_buf, s.index);
      return;
    }

    // =========================================================================
    // ws == 2: v3 variable exchange protocol with Metal kernels
    // =========================================================================
    {
      int my_rank = group().rank();
      int peer = 1 - my_rank;
      Group grp = group();

      // Packet row layout: [header(16B) | payload(D*elem_size) | pad]
      size_t raw_row = 16 + static_cast<size_t>(D_val) * elem_size;
      int row_stride = static_cast<int>((raw_row + 15) & ~size_t(15));

      int max_send = N * top_k;
      int recv_cap = experts_per_device * capacity;

      // Build routing metadata on CPU
      // local_slot_map[i] = flat_idx for i-th valid local assignment
      // local_nk[i] = n*top_k+k for i-th valid local assignment
      // remote_tok_idx[j] = n (token index) for j-th remote assignment
      // remote_headers[j] = meta32 = (local_expert << 16) | (pos & 0xFFFF)
      std::vector<int32_t> local_slot_map;
      std::vector<int32_t> local_nk;
      std::vector<int32_t> remote_tok_idx;
      std::vector<uint32_t> remote_headers;

      local_slot_map.reserve(N * top_k);
      local_nk.reserve(N * top_k);
      remote_tok_idx.reserve(N * top_k);
      remote_headers.reserve(N * top_k);

      // k-outer, n-inner for deterministic slot assignment
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_ptr[n * top_k + k];
          if (eid < 0 || eid >= num_experts)
            continue;
          int dest_rank = eid / experts_per_device;
          int local_expert = eid % experts_per_device;
          int pos = expert_counts[eid]++;
          if (pos >= capacity)
            continue;

          int flat_idx = local_expert * cap_total + dest_rank * capacity + pos;
          route_ptr[n * top_k + k] = flat_idx;

          if (dest_rank == my_rank) {
            // LOCAL
            local_slot_map.push_back(flat_idx);
            local_nk.push_back(n * top_k + k);
          } else {
            // REMOTE — will be packed into packet via GPU kernel
            remote_tok_idx.push_back(n);
            uint32_t meta = (static_cast<uint32_t>(local_expert) << 16) |
                (static_cast<uint32_t>(pos) & 0xFFFF);
            remote_headers.push_back(meta);
          }
        }
      }

      int local_count = static_cast<int>(local_slot_map.size());
      int send_count = static_cast<int>(remote_tok_idx.size());

      // --- GPU Phase 1: local scatter via moe_dispatch_local ---
      if (local_count > 0) {
        array slot_map_buf({local_count}, int32, nullptr, {});
        slot_map_buf.set_data(allocator::malloc(local_count * sizeof(int32_t)));
        std::memcpy(
            slot_map_buf.data<int32_t>(),
            local_slot_map.data(),
            local_count * sizeof(int32_t));

        array nk_buf({local_count}, int32, nullptr, {});
        nk_buf.set_data(allocator::malloc(local_count * sizeof(int32_t)));
        std::memcpy(
            nk_buf.data<int32_t>(),
            local_nk.data(),
            local_count * sizeof(int32_t));

        auto kernel = get_moe_kernel(d, "moe_dispatch_local", dtype);
        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);
        enc.set_input_array(tokens_in, 0);
        enc.set_output_array(outputs[0], 1);
        enc.set_input_array(slot_map_buf, 2);
        enc.set_input_array(nk_buf, 3);
        enc.set_bytes(D_val, 4);
        enc.set_bytes(top_k, 5);

        int tx = std::min(D_val, 256);
        MTL::Size grid_dims = MTL::Size(D_val, local_count, 1);
        MTL::Size group_dims = MTL::Size(tx, 1, 1);
        enc.dispatch_threads(grid_dims, group_dims);

        d.add_temporary(slot_map_buf, s.index);
        d.add_temporary(nk_buf, s.index);
      }

      // --- GPU Phase 2: pack remote tokens into packets via moe_packet_gather
      // --- Allocate packet buffers
      size_t send_pkt_bytes =
          static_cast<size_t>(std::max(send_count, 1)) * row_stride;
      size_t recv_pkt_bytes =
          static_cast<size_t>(std::max(recv_cap, 1)) * row_stride;

      array send_pkt({static_cast<int>(send_pkt_bytes)}, uint8, nullptr, {});
      send_pkt.set_data(allocator::malloc(send_pkt_bytes));

      array recv_pkt({static_cast<int>(recv_pkt_bytes)}, uint8, nullptr, {});
      recv_pkt.set_data(allocator::malloc(recv_pkt_bytes));

      if (send_count > 0) {
        // Build src_idx (token row indices) and headers buffers
        array src_idx_buf({send_count}, int32, nullptr, {});
        src_idx_buf.set_data(allocator::malloc(send_count * sizeof(int32_t)));
        std::memcpy(
            src_idx_buf.data<int32_t>(),
            remote_tok_idx.data(),
            send_count * sizeof(int32_t));

        array headers_buf({send_count}, uint32, nullptr, {});
        headers_buf.set_data(allocator::malloc(send_count * sizeof(uint32_t)));
        std::memcpy(
            headers_buf.data<uint32_t>(),
            remote_headers.data(),
            send_count * sizeof(uint32_t));

        auto pkt_kernel = get_moe_kernel(d, "moe_packet_gather", dtype);
        auto& enc2 = d.get_command_encoder(s.index);
        enc2.set_compute_pipeline_state(pkt_kernel);
        enc2.set_input_array(tokens_in, 0);
        enc2.set_output_array(send_pkt, 1);
        enc2.set_input_array(src_idx_buf, 2);
        enc2.set_input_array(headers_buf, 3);
        enc2.set_bytes(D_val, 4);
        enc2.set_bytes(send_count, 5);
        enc2.set_bytes(row_stride, 6);

        int tx = std::min(D_val, 256);
        MTL::Size grid_dims = MTL::Size(D_val, send_count, 1);
        MTL::Size group_dims = MTL::Size(tx, 1, 1);
        enc2.dispatch_threads(grid_dims, group_dims);

        d.add_temporary(src_idx_buf, s.index);
        d.add_temporary(headers_buf, s.index);
      }

      // --- Mid-sync: flush GPU work so packet data is ready for RDMA ---
      gpu::synchronize(s);

      // --- CPU: v3 exchange ---
      array count_send({1}, int32, nullptr, {});
      count_send.set_data(allocator::malloc(sizeof(int32_t)));
      array count_recv({1}, int32, nullptr, {});
      count_recv.set_data(allocator::malloc(sizeof(int32_t)));

      auto* raw = grp.raw_group().get();
      int peer_count = raw->blocking_exchange_v(
          send_pkt,
          send_count,
          recv_pkt,
          recv_cap,
          row_stride,
          peer,
          detail::ExchangeTag::MoeDispatchCount,
          detail::ExchangeTag::MoeDispatchPayload,
          count_send,
          count_recv);

      // --- CPU: decode recv meta -> build recv_flat_idx ---
      if (peer_count > 0) {
        std::vector<int32_t> recv_flat_idx_vec(peer_count);
        auto* recv_pkt_ptr = recv_pkt.data<uint8_t>();

        for (int i = 0; i < peer_count; i++) {
          const uint8_t* row =
              recv_pkt_ptr + static_cast<size_t>(i) * row_stride;
          uint32_t meta;
          std::memcpy(&meta, row, 4);
          int local_expert = static_cast<int>(meta >> 16);
          int slot_pos = static_cast<int>(meta & 0xFFFF);

          if (local_expert < 0 || local_expert >= experts_per_device ||
              slot_pos < 0 || slot_pos >= capacity) {
            throw std::runtime_error(
                "[MoeDispatchExchange::eval_gpu] received out-of-bounds "
                "metadata: local_expert=" +
                std::to_string(local_expert) +
                " slot_pos=" + std::to_string(slot_pos));
          }
          recv_flat_idx_vec[i] =
              local_expert * cap_total + peer * capacity + slot_pos;
        }

        // Build GPU buffer for flat indices
        array flat_idx_buf({peer_count}, int32, nullptr, {});
        flat_idx_buf.set_data(allocator::malloc(peer_count * sizeof(int32_t)));
        std::memcpy(
            flat_idx_buf.data<int32_t>(),
            recv_flat_idx_vec.data(),
            peer_count * sizeof(int32_t));

        // --- GPU Phase 3: scatter recv packets into dispatched ---
        auto scatter_kernel = get_moe_kernel(d, "moe_packet_scatter", dtype);
        auto& enc3 = d.get_command_encoder(s.index);
        enc3.set_compute_pipeline_state(scatter_kernel);
        enc3.set_input_array(recv_pkt, 0);
        enc3.set_output_array(outputs[0], 1);
        enc3.set_input_array(flat_idx_buf, 2);
        enc3.set_bytes(D_val, 3);
        enc3.set_bytes(peer_count, 4);
        enc3.set_bytes(row_stride, 5);

        int tx = std::min(D_val, 256);
        MTL::Size grid_dims = MTL::Size(D_val, peer_count, 1);
        MTL::Size group_dims = MTL::Size(tx, 1, 1);
        enc3.dispatch_threads(grid_dims, group_dims);

        d.add_temporary(flat_idx_buf, s.index);
        d.add_temporary(recv_pkt, s.index);
      }
      // send_pkt, count_send, count_recv will be freed when going out of scope
      // (they are no longer referenced by any GPU command after synchronize)
    }

  } catch (const std::exception& e) {
    const char* fb_env = std::getenv("MLX_MOE_EP_FALLBACK_ON_ERROR");
    bool fallback_enabled = !fb_env || std::string(fb_env) != "0";

    if (!fallback_enabled) {
      throw; // rethrow for debug/CI
    }

    // Log warning (once)
    static std::once_flag warn_flag;
    std::call_once(warn_flag, [&]() {
      std::cerr << "[MoE EP] Metal eval_gpu failed: " << e.what()
                << ". Falling back to CPU.\n";
    });

    // Flush any partial GPU state before CPU fallback
    try {
      gpu::synchronize(stream());
    } catch (...) {
    }

    // CPU fallback: create new CPU-stream primitive
    MoeDispatchExchange cpu_prim(
        to_stream(std::monostate{}, Device::cpu),
        group(),
        num_experts_,
        capacity_,
        deterministic_,
        MoeBackend::Cpu);
    cpu_prim.eval_cpu(inputs, outputs);
  }
}

// ---------------------------------------------------------------------------
// MoeCombineExchange::eval_gpu
// ---------------------------------------------------------------------------
//
// Flows:
//   ws==1: synchronize -> CPU route -> GPU combine_weighted_sum
//   ws==2: synchronize -> CPU route -> CPU exchange_v (requests) ->
//          CPU decode -> GPU packet_gather (responses) -> mid-sync ->
//          CPU exchange_v (responses) -> CPU decode ->
//          GPU build unified_src -> GPU packet_scatter ->
//          GPU combine_weighted_sum
//   ws>2:  CPU fallback
// ---------------------------------------------------------------------------

void MoeCombineExchange::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  try {
    // Fault injection (test-only)
    if (std::getenv("MLX_MOE_EP_FORCE_METAL_ERROR")) {
      throw std::runtime_error("[MoE EP] Forced Metal error for testing");
    }

    assert(inputs.size() == 4);
    assert(outputs.size() == 1);

    auto& s = stream();
    auto& d = metal::device(s.device);
    int world_size = group().size();

    // ws > 2: fallback to CPU eval
    if (world_size > 2) {
      MoeCombineExchange cpu_fb(
          to_stream(std::monostate{}, Device::cpu),
          group(),
          num_experts_,
          capacity_,
          deterministic_,
          MoeBackend::Cpu);
      cpu_fb.eval_cpu(inputs, outputs);
      return;
    }

    // inputs: expert_outputs [E_local, cap_total, D],
    //         route_indices [N, top_k] int32,
    //         weights [N, top_k] float32,
    //         original_tokens [N, D]
    auto& expert_out = inputs[0];
    auto& route_idx = inputs[1];
    auto& weights_in = inputs[2];
    auto& orig_tok = inputs[3];

    int experts_per_device = expert_out.shape(0);
    int cap_total = expert_out.shape(1);
    int D_val = expert_out.shape(2);
    int N = orig_tok.shape(0);
    int top_k = route_idx.shape(1);
    int capacity = capacity_;
    size_t elem_size = expert_out.itemsize();
    Dtype dtype = expert_out.dtype();

    // Allocate output
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

    // Step 1: gpu::synchronize to flush upstream so CPU can read route_indices
    gpu::synchronize(s);

    const int32_t* ri_ptr = route_idx.data<int32_t>();
    const float* w_ptr = weights_in.data<float>();

    // =========================================================================
    // ws == 1: local-only path
    // =========================================================================
    if (world_size == 1) {
      // Build src_idx for the weighted_sum kernel:
      // src_idx[n*top_k+k] = flat_idx into expert_out, or -1
      // For ws=1, expert_out is directly indexed by route_idx flat values
      // since data_src == expert_out

      // src_idx is just the route_indices — already in the right format
      // We can pass route_idx directly as src_idx

      auto kernel = get_moe_kernel(d, "moe_combine_weighted_sum", dtype);
      auto& enc = d.get_command_encoder(s.index);
      enc.set_compute_pipeline_state(kernel);
      enc.set_input_array(expert_out, 0); // data_src
      enc.set_output_array(outputs[0], 1); // output
      enc.set_input_array(orig_tok, 2); // original
      enc.set_input_array(weights_in, 3); // weights
      enc.set_input_array(route_idx, 4); // src_idx
      enc.set_bytes(D_val, 5);
      enc.set_bytes(N, 6);
      enc.set_bytes(top_k, 7);

      int tx = std::min(D_val, 256);
      MTL::Size grid_dims = MTL::Size(D_val, N, 1);
      MTL::Size group_dims = MTL::Size(tx, 1, 1);
      enc.dispatch_threads(grid_dims, group_dims);

      return;
    }

    // =========================================================================
    // ws == 2: v3 combine protocol with Metal kernels
    // =========================================================================
    {
      int my_rank = group().rank();
      int peer = 1 - my_rank;
      Group grp = group();

      // Response row layout: [header(16B) | payload(D*elem_size) | pad]
      size_t raw_resp_row = 16 + static_cast<size_t>(D_val) * elem_size;
      int resp_stride = static_cast<int>((raw_resp_row + 15) & ~size_t(15));

      // Request row layout: [token_slot32(4B) | local_expert16(2B) | pos16(2B)]
      int req_stride = 8;

      int max_local_routes = N * top_k;
      int max_peer_routes = experts_per_device * capacity;

      // --------------- Step 2: CPU route analysis ---------------
      // Separate local vs remote routes

      // For local routes: we need src_idx entries pointing into expert_out
      // For remote routes: we need to send requests to peer

      // src_idx[n*top_k+k] will eventually be an index into unified_src
      // For ws==2, unified_src = expert_out (possibly copied) + scattered
      // responses

      // We track:
      // - local_entries: (nk_idx, flat_idx_in_expert_out) for local
      // accumulation
      // - remote_entries: (nk_idx, local_expert, pos) for remote requests
      struct LocalEntry {
        int nk_idx;
        int flat_idx;
      };
      struct RemoteEntry {
        int nk_idx;
        int local_expert;
        int pos;
      };

      std::vector<LocalEntry> local_entries;
      std::vector<RemoteEntry> remote_entries;
      local_entries.reserve(N * top_k);
      remote_entries.reserve(N * top_k);

      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int flat_idx = ri_ptr[n * top_k + k];
          if (flat_idx < 0)
            continue;

          int remainder = flat_idx % cap_total;
          int dest_rank = remainder / capacity;

          if (dest_rank == my_rank) {
            local_entries.push_back({n * top_k + k, flat_idx});
          } else {
            int local_expert_idx = flat_idx / cap_total;
            int pos = remainder % capacity;
            remote_entries.push_back({n * top_k + k, local_expert_idx, pos});
          }
        }
      }

      int req_send_count = static_cast<int>(remote_entries.size());

      // --------------- Step 3: CPU exchange requests ---------------
      // Allocate request buffers
      size_t req_send_bytes =
          static_cast<size_t>(std::max(req_send_count, 1)) * req_stride;
      size_t req_recv_bytes =
          static_cast<size_t>(std::max(max_peer_routes, 1)) * req_stride;

      array req_send({static_cast<int>(req_send_bytes)}, uint8, nullptr, {});
      req_send.set_data(allocator::malloc(req_send_bytes));
      array req_recv({static_cast<int>(req_recv_bytes)}, uint8, nullptr, {});
      req_recv.set_data(allocator::malloc(req_recv_bytes));

      // Pack requests
      auto* req_send_ptr = req_send.data<uint8_t>();
      for (int i = 0; i < req_send_count; i++) {
        auto& re = remote_entries[i];
        uint32_t token_slot = static_cast<uint32_t>(re.nk_idx);
        uint16_t le16 = static_cast<uint16_t>(re.local_expert);
        uint16_t pos16 = static_cast<uint16_t>(re.pos);

        uint8_t* row = req_send_ptr + static_cast<size_t>(i) * req_stride;
        std::memcpy(row, &token_slot, 4);
        std::memcpy(row + 4, &le16, 2);
        std::memcpy(row + 6, &pos16, 2);
      }

      // Count exchange arrays
      array count_send({1}, int32, nullptr, {});
      count_send.set_data(allocator::malloc(sizeof(int32_t)));
      array count_recv({1}, int32, nullptr, {});
      count_recv.set_data(allocator::malloc(sizeof(int32_t)));

      auto* raw = grp.raw_group().get();

      int peer_req_count = raw->blocking_exchange_v(
          req_send,
          req_send_count,
          req_recv,
          max_peer_routes,
          req_stride,
          peer,
          detail::ExchangeTag::MoeCombineReqCount,
          detail::ExchangeTag::MoeCombineReqPayload,
          count_send,
          count_recv);

      // --------------- Step 4: decode peer requests, build response metadata
      // --- For each received request, we need to:
      //   - look up the flat_idx in expert_out
      //   - record the token_slot for the response header
      std::vector<int32_t> eo_flat_idx_vec(peer_req_count);
      std::vector<uint32_t> resp_headers_vec(peer_req_count);

      auto* req_recv_ptr = req_recv.data<uint8_t>();
      for (int i = 0; i < peer_req_count; i++) {
        const uint8_t* req_row =
            req_recv_ptr + static_cast<size_t>(i) * req_stride;
        uint32_t token_slot;
        uint16_t le16, pos16;
        std::memcpy(&token_slot, req_row, 4);
        std::memcpy(&le16, req_row + 4, 2);
        std::memcpy(&pos16, req_row + 6, 2);

        int local_expert = static_cast<int>(le16);
        int slot_pos = static_cast<int>(pos16);

        if (local_expert < 0 || local_expert >= experts_per_device ||
            slot_pos < 0 || slot_pos >= capacity) {
          throw std::runtime_error(
              "[MoeCombineExchange::eval_gpu] out-of-bounds request: "
              "local_expert=" +
              std::to_string(local_expert) +
              " pos=" + std::to_string(slot_pos));
        }

        // expert_out flat index: expert_out[local_expert,
        // peer*capacity+slot_pos]
        int eo_flat = local_expert * cap_total + peer * capacity + slot_pos;
        eo_flat_idx_vec[i] = eo_flat;
        resp_headers_vec[i] = token_slot;
      }

      // --------------- Step 5: GPU packet_gather (pack responses)
      // ---------------
      size_t resp_send_bytes =
          static_cast<size_t>(std::max(peer_req_count, 1)) * resp_stride;
      size_t resp_recv_bytes =
          static_cast<size_t>(std::max(req_send_count, 1)) * resp_stride;

      array resp_send({static_cast<int>(resp_send_bytes)}, uint8, nullptr, {});
      resp_send.set_data(allocator::malloc(resp_send_bytes));
      array resp_recv({static_cast<int>(resp_recv_bytes)}, uint8, nullptr, {});
      resp_recv.set_data(allocator::malloc(resp_recv_bytes));

      if (peer_req_count > 0) {
        // Build GPU metadata buffers
        array eo_flat_idx_buf({peer_req_count}, int32, nullptr, {});
        eo_flat_idx_buf.set_data(
            allocator::malloc(peer_req_count * sizeof(int32_t)));
        std::memcpy(
            eo_flat_idx_buf.data<int32_t>(),
            eo_flat_idx_vec.data(),
            peer_req_count * sizeof(int32_t));

        array resp_hdr_buf({peer_req_count}, uint32, nullptr, {});
        resp_hdr_buf.set_data(
            allocator::malloc(peer_req_count * sizeof(uint32_t)));
        std::memcpy(
            resp_hdr_buf.data<uint32_t>(),
            resp_headers_vec.data(),
            peer_req_count * sizeof(uint32_t));

        auto pkt_kernel = get_moe_kernel(d, "moe_packet_gather", dtype);
        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(pkt_kernel);
        enc.set_input_array(expert_out, 0); // source
        enc.set_output_array(resp_send, 1); // packet
        enc.set_input_array(eo_flat_idx_buf, 2); // src_idx
        enc.set_input_array(resp_hdr_buf, 3); // headers
        enc.set_bytes(D_val, 4);
        enc.set_bytes(peer_req_count, 5);
        enc.set_bytes(resp_stride, 6);

        int tx = std::min(D_val, 256);
        MTL::Size grid_dims = MTL::Size(D_val, peer_req_count, 1);
        MTL::Size group_dims = MTL::Size(tx, 1, 1);
        enc.dispatch_threads(grid_dims, group_dims);

        d.add_temporary(eo_flat_idx_buf, s.index);
        d.add_temporary(resp_hdr_buf, s.index);
      }

      // --------------- Mid-sync: flush GPU packet_gather ---------------
      gpu::synchronize(s);

      // --------------- Step 6: CPU exchange responses ---------------
      int peer_res_count = raw->blocking_exchange_v(
          resp_send,
          peer_req_count,
          resp_recv,
          req_send_count,
          resp_stride,
          peer,
          detail::ExchangeTag::MoeCombineResCount,
          detail::ExchangeTag::MoeCombineResPayload,
          count_send,
          count_recv);

      // --------------- Step 7: decode recv responses -> build scatter indices
      // --- resp_recv contains [token_slot32(in 16B header) | payload] We need
      // to:
      //   1. Build unified_src workspace that contains all data rows
      //      (expert_out rows for local + received response rows for remote)
      //   2. Build src_idx[N*top_k] mapping (n,k) -> row in unified_src
      //   3. Run moe_combine_weighted_sum

      // unified_src layout:
      //   Rows 0..E_local*cap_total-1  = expert_out (for local lookups)
      //   Rows E_local*cap_total..      = received response payloads
      int eo_total_rows = experts_per_device * cap_total;
      int unified_total_rows = eo_total_rows + peer_res_count;

      // Allocate unified_src
      size_t unified_nbytes =
          static_cast<size_t>(unified_total_rows) * D_val * elem_size;
      array unified_src({unified_total_rows, D_val}, dtype, nullptr, {});
      unified_src.set_data(allocator::malloc(unified_nbytes));

      // Copy expert_out into unified_src base region
      // After synchronize, CPU can safely memcpy from expert_out (UMA)
      std::memcpy(
          unified_src.data<void>(),
          expert_out.data<void>(),
          static_cast<size_t>(eo_total_rows) * D_val * elem_size);

      // Build src_idx on CPU
      // Initialize all to -1
      std::vector<int32_t> src_idx_vec(N * top_k, -1);

      // Local entries: src_idx -> flat_idx in expert_out = row in unified_src
      for (auto& le : local_entries) {
        src_idx_vec[le.nk_idx] = le.flat_idx;
      }

      // Decode responses and scatter payloads into unified_src
      if (peer_res_count > 0) {
        auto* resp_recv_ptr = resp_recv.data<uint8_t>();

        // Build a map from token_slot -> response index for the scatter
        std::vector<int32_t> resp_flat_idx_vec(peer_res_count);

        for (int i = 0; i < peer_res_count; i++) {
          const uint8_t* resp_row =
              resp_recv_ptr + static_cast<size_t>(i) * resp_stride;
          uint32_t token_slot;
          std::memcpy(&token_slot, resp_row, 4);

          if (token_slot >= static_cast<uint32_t>(N * top_k)) {
            throw std::runtime_error(
                "[MoeCombineExchange::eval_gpu] invalid token_slot=" +
                std::to_string(token_slot));
          }

          // This response goes to row eo_total_rows + i in unified_src
          int unified_row = eo_total_rows + i;
          src_idx_vec[static_cast<int>(token_slot)] = unified_row;

          // Target row in unified_src for packet_scatter
          resp_flat_idx_vec[i] = unified_row;
        }

        // GPU: scatter received response payloads into unified_src
        array resp_flat_buf({peer_res_count}, int32, nullptr, {});
        resp_flat_buf.set_data(
            allocator::malloc(peer_res_count * sizeof(int32_t)));
        std::memcpy(
            resp_flat_buf.data<int32_t>(),
            resp_flat_idx_vec.data(),
            peer_res_count * sizeof(int32_t));

        auto scatter_kernel = get_moe_kernel(d, "moe_packet_scatter", dtype);
        auto& enc_scatter = d.get_command_encoder(s.index);
        enc_scatter.set_compute_pipeline_state(scatter_kernel);
        enc_scatter.set_input_array(resp_recv, 0);
        enc_scatter.set_output_array(unified_src, 1);
        enc_scatter.set_input_array(resp_flat_buf, 2);
        enc_scatter.set_bytes(D_val, 3);
        enc_scatter.set_bytes(peer_res_count, 4);
        enc_scatter.set_bytes(resp_stride, 5);

        int tx = std::min(D_val, 256);
        MTL::Size grid_dims = MTL::Size(D_val, peer_res_count, 1);
        MTL::Size group_dims = MTL::Size(tx, 1, 1);
        enc_scatter.dispatch_threads(grid_dims, group_dims);

        d.add_temporary(resp_flat_buf, s.index);
        d.add_temporary(resp_recv, s.index);
      }

      // --------------- Step 8: GPU combine_weighted_sum ---------------
      // Build src_idx GPU buffer
      array src_idx_buf({N * top_k}, int32, nullptr, {});
      src_idx_buf.set_data(allocator::malloc(N * top_k * sizeof(int32_t)));
      std::memcpy(
          src_idx_buf.data<int32_t>(),
          src_idx_vec.data(),
          N * top_k * sizeof(int32_t));

      auto ws_kernel = get_moe_kernel(d, "moe_combine_weighted_sum", dtype);
      auto& enc_ws = d.get_command_encoder(s.index);
      enc_ws.set_compute_pipeline_state(ws_kernel);
      enc_ws.set_input_array(unified_src, 0); // data_src
      enc_ws.set_output_array(outputs[0], 1); // output
      enc_ws.set_input_array(orig_tok, 2); // original
      enc_ws.set_input_array(weights_in, 3); // weights
      enc_ws.set_input_array(src_idx_buf, 4); // src_idx
      enc_ws.set_bytes(D_val, 5);
      enc_ws.set_bytes(N, 6);
      enc_ws.set_bytes(top_k, 7);

      int tx = std::min(D_val, 256);
      MTL::Size grid_dims = MTL::Size(D_val, N, 1);
      MTL::Size group_dims = MTL::Size(tx, 1, 1);
      enc_ws.dispatch_threads(grid_dims, group_dims);

      // Keep temporaries alive
      d.add_temporary(unified_src, s.index);
      d.add_temporary(src_idx_buf, s.index);
    }

  } catch (const std::exception& e) {
    const char* fb_env = std::getenv("MLX_MOE_EP_FALLBACK_ON_ERROR");
    bool fallback_enabled = !fb_env || std::string(fb_env) != "0";

    if (!fallback_enabled) {
      throw; // rethrow for debug/CI
    }

    // Log warning (once)
    static std::once_flag combine_warn_flag;
    std::call_once(combine_warn_flag, [&]() {
      std::cerr << "[MoE EP] Metal eval_gpu failed: " << e.what()
                << ". Falling back to CPU.\n";
    });

    // Flush any partial GPU state before CPU fallback
    try {
      gpu::synchronize(stream());
    } catch (...) {
    }

    // CPU fallback: create new CPU-stream primitive
    MoeCombineExchange cpu_prim(
        to_stream(std::monostate{}, Device::cpu),
        group(),
        num_experts_,
        capacity_,
        deterministic_,
        MoeBackend::Cpu);
    cpu_prim.eval_cpu(inputs, outputs);
  }
}

} // namespace mlx::core::distributed
