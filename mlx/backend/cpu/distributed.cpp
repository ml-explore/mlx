// Copyright © 2024 Apple Inc.

#include <cassert>
#include <cstring>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/primitives.h"
#include "mlx/types/half_types.h"

namespace mlx::core::distributed {

std::pair<array, bool> ensure_row_contiguous(const array& arr, Stream stream) {
  if (arr.flags().row_contiguous) {
    return {arr, false};
  } else {
    return {contiguous_copy_cpu(arr, stream), true};
  }
};

void AllReduce::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto donate_or_copy = [s = stream()](const array& in, array& out) {
    if (in.flags().row_contiguous) {
      if (in.is_donatable()) {
        out.copy_shared_buffer(in);
      } else {
        out.set_data(allocator::malloc(out.nbytes()));
      }
      return in;
    } else {
      array arr_copy = contiguous_copy_cpu(in, s);
      out.copy_shared_buffer(arr_copy);
      return arr_copy;
    }
  };

  auto in = donate_or_copy(inputs[0], outputs[0]);
  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), in, outputs[0], stream());
      break;
    case Max:
      distributed::detail::all_max(group(), in, outputs[0], stream());
      break;
    case Min:
      distributed::detail::all_min(group(), in, outputs[0], stream());
      break;
    default:
      throw std::runtime_error(
          "Only all reduce sum, min and max are supported for now");
  }
}

void AllGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::all_gather(group(), in, outputs[0], stream());
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

void Send::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  distributed::detail::send(group(), in, dst_, stream());
  outputs[0].copy_shared_buffer(inputs[0]);
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

void Recv::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::recv(group(), outputs[0], src_, stream());
}

void ReduceScatter::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[ReduceScatter] Not implemented yet.");
}

void AllToAll::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::all_to_all(group(), in, outputs[0], stream());
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

// Helper: make a row-contiguous array that shares the buffer of `parent`,
// starting at byte offset `byte_offset`, with shape `shape`.
// The caller must ensure `parent` stays alive while the returned array is used.
static array make_subview(
    const array& parent,
    const Shape& shape,
    size_t byte_offset,
    Dtype dtype) {
  // Compute strides for row-contiguous layout
  Strides strides(shape.size());
  if (!shape.empty()) {
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  size_t num_elems = 1;
  for (auto s : shape)
    num_elems *= s;
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = (shape.size() <= 1 || num_elems <= 1);

  // byte_offset in terms of elements
  int64_t elem_offset = static_cast<int64_t>(byte_offset / size_of(dtype));
  array view(shape, dtype, nullptr, {});
  view.copy_shared_buffer(parent, strides, flags, num_elems, elem_offset);
  return view;
}

void MoeDispatchExchange::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  assert(outputs.size() == 2);

  auto [tokens_in, tok_copied] = ensure_row_contiguous(inputs[0], stream());
  auto [indices_in, idx_copied] = ensure_row_contiguous(inputs[1], stream());

  // Allocate outputs before dispatch so callers can depend on their pointers
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));

  int N = tokens_in.shape(0);
  int D = tokens_in.shape(1);
  int top_k = indices_in.shape(1);
  int world_size = group().size();
  int num_experts = num_experts_;
  int capacity = capacity_;
  Group grp = group();
  size_t elem_size = tokens_in.itemsize();
  Dtype dtype = tokens_in.dtype();

  // Capture raw pointers; arrays kept alive via add_temporary below
  const void* tok_raw = tokens_in.data<void>();
  const int32_t* idx_raw = indices_in.data<int32_t>();
  void* out0_raw = outputs[0].data<void>();
  int32_t* out1_raw = outputs[1].data<int32_t>();

  auto& enc = cpu::get_command_encoder(stream());

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([tok_raw,
                idx_raw,
                out0_raw,
                out1_raw,
                N,
                D,
                top_k,
                world_size,
                num_experts,
                capacity,
                elem_size,
                dtype,
                grp]() mutable {
    int experts_per_device = num_experts / world_size;
    int cap_total = world_size * capacity; // total capacity slots per expert

    // Initialize route_indices to -1
    int32_t* route_ptr = out1_raw;
    std::fill(route_ptr, route_ptr + N * top_k, int32_t(-1));

    // Zero-initialize the output dispatched buffer
    // Output shape: [experts_per_device, cap_total, D]
    size_t out_nbytes = (size_t)experts_per_device * cap_total * D * elem_size;
    std::memset(out0_raw, 0, out_nbytes);

    const auto* tok_bytes = static_cast<const uint8_t*>(tok_raw);
    auto* out_bytes = static_cast<uint8_t*>(out0_raw);
    std::vector<int> expert_counts(num_experts, 0);

    // world_size == 1: local-only path (no send/recv)
    // New route_idx layout: flat_idx = local_expert * cap_total + dest_rank *
    // capacity + pos For world_size=1: dest_rank=0, cap_total=capacity
    //   => flat_idx = local_expert * capacity + pos  (same formula as old for
    //   ws=1)
    if (world_size == 1) {
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
          if (eid < 0 || eid >= num_experts)
            continue;
          int pos = expert_counts[eid]++;
          if (pos < capacity) {
            // dest_rank=0, local_expert=eid, cap_total=capacity
            int flat_idx = eid * cap_total + 0 * capacity + pos;
            route_ptr[n * top_k + k] = flat_idx;
            std::memcpy(
                out_bytes + flat_idx * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          }
          // else: route stays -1
        }
      }
      return;
    }

    // world_size == 2: v3 variable exchange protocol
    if (world_size == 2) {
      int my_rank = grp.rank();
      int peer = 1 - my_rank;

      // Packet row layout: [meta32(4B) | payload(D*elem_size) | pad]
      // meta32 = (local_expert << 16) | (pos & 0xFFFF)
      size_t raw_row = 4 + D * elem_size;
      int row_stride =
          static_cast<int>((raw_row + 15) & ~size_t(15)); // align to 16

      int max_send = N * top_k;
      int recv_cap = experts_per_device *
          capacity; // peer can fill at most capacity per expert

      // Allocate packet buffers
      size_t send_pkt_bytes =
          static_cast<size_t>(std::max(max_send, 1)) * row_stride;
      size_t recv_pkt_bytes =
          static_cast<size_t>(std::max(recv_cap, 1)) * row_stride;

      array send_pkt({static_cast<int>(send_pkt_bytes)}, uint8, nullptr, {});
      send_pkt.set_data(allocator::malloc(send_pkt_bytes));
      auto* send_pkt_ptr = send_pkt.data<uint8_t>();

      array recv_pkt({static_cast<int>(recv_pkt_bytes)}, uint8, nullptr, {});
      recv_pkt.set_data(allocator::malloc(recv_pkt_bytes));

      // Count exchange arrays
      array count_send({1}, int32, nullptr, {});
      count_send.set_data(allocator::malloc(sizeof(int32_t)));
      array count_recv({1}, int32, nullptr, {});
      count_recv.set_data(allocator::malloc(sizeof(int32_t)));

      int send_count = 0;

      // Dispatch: k-outer, n-inner deterministic loop
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
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
            // LOCAL: directly scatter into output
            std::memcpy(
                out_bytes + flat_idx * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          } else {
            // REMOTE: pack into send packet
            uint8_t* row =
                send_pkt_ptr + static_cast<size_t>(send_count) * row_stride;
            uint32_t meta = (static_cast<uint32_t>(local_expert) << 16) |
                (static_cast<uint32_t>(pos) & 0xFFFF);
            std::memcpy(row, &meta, 4);
            std::memcpy(row + 4, tok_bytes + n * D * elem_size, D * elem_size);
            send_count++;
          }
        }
      }

      // Exchange packets
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

      // Scatter received remote tokens into output
      auto* recv_pkt_ptr = recv_pkt.data<uint8_t>();
      for (int i = 0; i < peer_count; i++) {
        const uint8_t* row = recv_pkt_ptr + static_cast<size_t>(i) * row_stride;
        uint32_t meta;
        std::memcpy(&meta, row, 4);
        int local_expert = static_cast<int>(meta >> 16);
        int slot_pos = static_cast<int>(meta & 0xFFFF);
        if (local_expert < 0 || local_expert >= experts_per_device ||
            slot_pos < 0 || slot_pos >= capacity) {
          throw std::runtime_error(
              "[MoeDispatchExchange] received out-of-bounds metadata: "
              "local_expert=" +
              std::to_string(local_expert) +
              " slot_pos=" + std::to_string(slot_pos));
        }
        int recv_flat_idx =
            local_expert * cap_total + peer * capacity + slot_pos;
        std::memcpy(
            out_bytes + recv_flat_idx * D * elem_size, row + 4, D * elem_size);
      }
      return;
    }

    // world_size > 2: fallback to existing fixed all_to_all
    {
      int slots_per_device = experts_per_device * capacity;
      int total_slots = world_size * slots_per_device;
      size_t send_nbytes = (size_t)total_slots * D * elem_size;

      // Allocate send buffer: [total_slots, D] (layout: [W, E, C, D])
      array send_arr(Shape{total_slots, D}, dtype, nullptr, {});
      send_arr.set_data(allocator::malloc(send_nbytes));
      std::memset(send_arr.data<void>(), 0, send_nbytes);

      auto* send_bytes = static_cast<uint8_t*>(send_arr.data<void>());

      // Dispatch: k-outer, n-inner for deterministic slot assignment
      // Use NEW route_idx layout: flat_idx = local_expert * cap_total +
      // dest_rank * capacity + pos The send buffer uses old layout [W, E, C, D]
      // for all_to_all compatibility
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
          if (eid < 0 || eid >= num_experts)
            continue;
          int pos = expert_counts[eid]++;
          if (pos < capacity) {
            int dest_rank = eid / experts_per_device;
            int local_expert = eid % experts_per_device;
            // Old send buffer layout for all_to_all: [W, E, C, D]
            int send_flat =
                dest_rank * slots_per_device + local_expert * capacity + pos;
            // New route_idx layout
            int new_flat_idx =
                local_expert * cap_total + dest_rank * capacity + pos;
            route_ptr[n * top_k + k] = new_flat_idx;
            std::memcpy(
                send_bytes + send_flat * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          }
          // else: route stays -1
        }
      }

      // Allocate recv buffer
      array recv_arr(Shape{total_slots, D}, dtype, nullptr, {});
      recv_arr.set_data(allocator::malloc(send_nbytes));

      // All-to-all exchange using blocking API
      grp.raw_group()->blocking_all_to_all(send_arr, recv_arr);

      // recv_arr layout: [world_size, experts_per_device, capacity, D]
      // output layout:   [experts_per_device, world_size * capacity, D]
      // out[e, w*capacity+c, d] = recv[w, e, c, d]
      const auto* recv_bytes =
          static_cast<const uint8_t*>(recv_arr.data<void>());

      for (int w = 0; w < world_size; w++) {
        for (int e = 0; e < experts_per_device; e++) {
          for (int c = 0; c < capacity; c++) {
            int recv_row = w * slots_per_device + e * capacity + c;
            int out_row = e * cap_total + w * capacity + c;
            std::memcpy(
                out_bytes + out_row * D * elem_size,
                recv_bytes + recv_row * D * elem_size,
                D * elem_size);
          }
        }
      }
      // send_arr and recv_arr go out of scope here; their allocator memory
      // is freed via the array destructor.
    }
  });

  // Keep input arrays alive until the dispatched lambda has executed
  enc.add_temporary(tokens_in);
  enc.add_temporary(indices_in);
}

void MoeCombineExchange::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  assert(outputs.size() == 1);

  // inputs: expert_outputs [E_local, cap_total, D],
  //         route_indices [N, top_k] int32,
  //         weights [N, top_k] float32,
  //         original_tokens [N, D]
  auto [expert_out, eo_copied] = ensure_row_contiguous(inputs[0], stream());
  auto [route_idx, ri_copied] = ensure_row_contiguous(inputs[1], stream());
  auto [weights_in, w_copied] = ensure_row_contiguous(inputs[2], stream());
  auto [orig_tok, ot_copied] = ensure_row_contiguous(inputs[3], stream());

  int experts_per_device = expert_out.shape(0);
  int cap_total = expert_out.shape(1);
  int D = expert_out.shape(2);
  int N = orig_tok.shape(0);
  int top_k = route_idx.shape(1);
  int world_size = group().size();
  int capacity = capacity_;
  Group grp = group();
  size_t elem_size = expert_out.itemsize();
  Dtype dtype = expert_out.dtype();

  // Allocate output before dispatch
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

  // Capture raw pointers; arrays kept alive via add_temporary below
  const void* eo_raw = expert_out.data<void>();
  const int32_t* ri_raw = route_idx.data<int32_t>();
  const float* w_raw = weights_in.data<float>();
  const void* orig_raw = orig_tok.data<void>();
  void* out0_raw = outputs[0].data<void>();

  auto& enc = cpu::get_command_encoder(stream());

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([eo_raw,
                ri_raw,
                w_raw,
                orig_raw,
                out0_raw,
                experts_per_device,
                cap_total,
                D,
                N,
                top_k,
                world_size,
                capacity,
                elem_size,
                dtype,
                grp]() mutable {
    // world_size == 1: local-only path (no send/recv)
    // route_idx flat_idx = local_expert * cap_total + 0 * capacity + pos
    //                    = local_expert * capacity + pos  (cap_total ==
    //                    capacity for ws=1)
    // expert_outputs is indexed directly by flat_idx
    if (world_size == 1) {
      switch (dtype) {
        case float32: {
          const auto* eo_f = static_cast<const float*>(eo_raw);
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);
          for (int n = 0; n < N; n++) {
            float* dst = out_f + n * D;
            std::fill(dst, dst + D, 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const float* src = eo_f + flat_idx * D;
                for (int d = 0; d < D; d++)
                  dst[d] += w * src[d];
              }
            }
            if (!has_valid) {
              std::memcpy(dst, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          const auto* eo_h = static_cast<const float16_t*>(eo_raw);
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);
          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const float16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            float16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++)
                dst[d] = float16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          const auto* eo_h = static_cast<const bfloat16_t*>(eo_raw);
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);
          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const bfloat16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            bfloat16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++)
                dst[d] = bfloat16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error(
              "[MoeCombineExchange] Unsupported dtype. Use float32, float16, or bfloat16.");
      }
      return;
    }

    // world_size == 2: v3 combine protocol
    if (world_size == 2) {
      int my_rank = grp.rank();
      int peer = 1 - my_rank;

      const auto* eo_bytes = static_cast<const uint8_t*>(eo_raw);

      // Response row layout: [token_slot32(4B) | payload(D*elem_size) | pad]
      size_t raw_resp_row = 4 + D * elem_size;
      int resp_stride = static_cast<int>((raw_resp_row + 15) & ~size_t(15));

      // Request row layout: [token_slot32(4B) | local_expert16(2B) | pos16(2B)]
      int req_stride = 8;

      int max_local_routes = N * top_k; // max requests WE send
      int max_peer_routes =
          experts_per_device * capacity; // max requests PEER can send

      // Allocate request buffers
      size_t req_send_bytes =
          static_cast<size_t>(std::max(max_local_routes, 1)) * req_stride;
      size_t req_recv_bytes =
          static_cast<size_t>(std::max(max_peer_routes, 1)) * req_stride;
      array req_send({static_cast<int>(req_send_bytes)}, uint8, nullptr, {});
      req_send.set_data(allocator::malloc(req_send_bytes));
      array req_recv({static_cast<int>(req_recv_bytes)}, uint8, nullptr, {});
      req_recv.set_data(allocator::malloc(req_recv_bytes));

      // Allocate response buffers — responses bounded by received requests /
      // sent requests
      size_t resp_send_bytes =
          static_cast<size_t>(std::max(max_peer_routes, 1)) * resp_stride;
      size_t resp_recv_bytes =
          static_cast<size_t>(std::max(max_local_routes, 1)) * resp_stride;
      array resp_send({static_cast<int>(resp_send_bytes)}, uint8, nullptr, {});
      resp_send.set_data(allocator::malloc(resp_send_bytes));
      array resp_recv({static_cast<int>(resp_recv_bytes)}, uint8, nullptr, {});
      resp_recv.set_data(allocator::malloc(resp_recv_bytes));

      // Count exchange arrays (reused for both request and response exchanges)
      array count_send({1}, int32, nullptr, {});
      count_send.set_data(allocator::malloc(sizeof(int32_t)));
      array count_recv({1}, int32, nullptr, {});
      count_recv.set_data(allocator::malloc(sizeof(int32_t)));

      // Lambda for weighted accumulate (handles all dtypes)
      auto weighted_add = [&](void* dst_raw,
                              const void* src_raw,
                              float w,
                              int D) {
        switch (dtype) {
          case float32: {
            auto* dst = static_cast<float*>(dst_raw);
            const auto* src = static_cast<const float*>(src_raw);
            for (int d = 0; d < D; d++)
              dst[d] += w * src[d];
            break;
          }
          case float16: {
            auto* dst = static_cast<float*>(dst_raw); // accumulate in float32
            const auto* src = static_cast<const float16_t*>(src_raw);
            for (int d = 0; d < D; d++)
              dst[d] += w * static_cast<float>(src[d]);
            break;
          }
          case bfloat16: {
            auto* dst = static_cast<float*>(dst_raw); // accumulate in float32
            const auto* src = static_cast<const bfloat16_t*>(src_raw);
            for (int d = 0; d < D; d++)
              dst[d] += w * static_cast<float>(src[d]);
            break;
          }
          default:
            throw std::runtime_error("[MoeCombineExchange] Unsupported dtype");
        }
      };

      // Accumulation buffer (always float32 for precision)
      std::vector<float> accum(static_cast<size_t>(N) * D, 0.0f);
      std::vector<bool> has_valid(N, false);

      int req_send_count = 0;
      auto* req_send_ptr = req_send.data<uint8_t>();

      // Step 1: Process all routes, accumulate local, pack remote requests
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int flat_idx = ri_raw[n * top_k + k];
          if (flat_idx < 0)
            continue;

          int remainder = flat_idx % cap_total;
          int dest_rank = remainder / capacity;
          float w = w_raw[n * top_k + k];

          if (dest_rank == my_rank) {
            // LOCAL: accumulate directly
            has_valid[n] = true;
            weighted_add(
                accum.data() + static_cast<size_t>(n) * D,
                eo_bytes + static_cast<size_t>(flat_idx) * D * elem_size,
                w,
                D);
          } else {
            // REMOTE: pack request
            int local_expert_idx = flat_idx / cap_total;
            int pos = remainder % capacity;
            uint32_t token_slot = static_cast<uint32_t>(n * top_k + k);
            uint16_t le16 = static_cast<uint16_t>(local_expert_idx);
            uint16_t pos16 = static_cast<uint16_t>(pos);

            uint8_t* row =
                req_send_ptr + static_cast<size_t>(req_send_count) * req_stride;
            std::memcpy(row, &token_slot, 4);
            std::memcpy(row + 4, &le16, 2);
            std::memcpy(row + 6, &pos16, 2);
            req_send_count++;
            has_valid[n] = true;
          }
        }
      }

      auto* raw = grp.raw_group().get();

      // Step 2: Exchange requests
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

      // Step 3: Build responses from received requests
      auto* req_recv_ptr = req_recv.data<uint8_t>();
      auto* resp_send_ptr = resp_send.data<uint8_t>();

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
              "[MoeCombineExchange] out-of-bounds request: local_expert=" +
              std::to_string(local_expert) +
              " pos=" + std::to_string(slot_pos));
        }

        // Lookup: expert_outputs at peer's slot
        int eo_flat = local_expert * cap_total + peer * capacity + slot_pos;

        // Pack response: [token_slot | payload]
        uint8_t* resp_row =
            resp_send_ptr + static_cast<size_t>(i) * resp_stride;
        std::memcpy(resp_row, &token_slot, 4);
        std::memcpy(
            resp_row + 4,
            eo_bytes + static_cast<size_t>(eo_flat) * D * elem_size,
            D * elem_size);
      }

      // Step 4: Exchange responses
      int peer_res_count = raw->blocking_exchange_v(
          resp_send,
          peer_req_count,
          resp_recv,
          max_local_routes,
          resp_stride,
          peer,
          detail::ExchangeTag::MoeCombineResCount,
          detail::ExchangeTag::MoeCombineResPayload,
          count_send,
          count_recv);

      // Step 5: Process responses — accumulate into output
      auto* resp_recv_ptr = resp_recv.data<uint8_t>();
      for (int i = 0; i < peer_res_count; i++) {
        const uint8_t* resp_row =
            resp_recv_ptr + static_cast<size_t>(i) * resp_stride;
        uint32_t token_slot;
        std::memcpy(&token_slot, resp_row, 4);

        if (token_slot >= static_cast<uint32_t>(N * top_k)) {
          throw std::runtime_error(
              "[MoeCombineExchange] invalid token_slot=" +
              std::to_string(token_slot));
        }

        int n = static_cast<int>(token_slot) / top_k;
        int k = static_cast<int>(token_slot) % top_k;
        float w = w_raw[n * top_k + k];

        weighted_add(
            accum.data() + static_cast<size_t>(n) * D, resp_row + 4, w, D);
      }

      // Step 6: Write output
      switch (dtype) {
        case float32: {
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);
          for (int n = 0; n < N; n++) {
            if (has_valid[n]) {
              std::memcpy(
                  out_f + n * D,
                  accum.data() + static_cast<size_t>(n) * D,
                  D * sizeof(float));
            } else {
              std::memcpy(out_f + n * D, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);
          for (int n = 0; n < N; n++) {
            if (has_valid[n]) {
              for (int d = 0; d < D; d++) {
                out_h[n * D + d] =
                    float16_t(accum[static_cast<size_t>(n) * D + d]);
              }
            } else {
              std::memcpy(out_h + n * D, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);
          for (int n = 0; n < N; n++) {
            if (has_valid[n]) {
              for (int d = 0; d < D; d++) {
                out_h[n * D + d] =
                    bfloat16_t(accum[static_cast<size_t>(n) * D + d]);
              }
            } else {
              std::memcpy(
                  out_h + n * D, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error("[MoeCombineExchange] Unsupported dtype");
      }
      return;
    }

    // world_size > 2: fallback to existing all_to_all
    {
      int total_slots = experts_per_device * cap_total; // E * W * C
      size_t total_bytes = (size_t)total_slots * D * elem_size;
      int slots_per_device = experts_per_device * capacity;

      // Reverse transpose: [E, W*C, D] -> [W, E, C, D]
      // send_arr[w, e, c, d] = expert_out[e, w*capacity+c, d]
      array send_arr(Shape{total_slots, D}, dtype, nullptr, {});
      send_arr.set_data(allocator::malloc(total_bytes));

      auto* send_bytes = static_cast<uint8_t*>(send_arr.data<void>());
      const auto* eo_bytes = static_cast<const uint8_t*>(eo_raw);

      for (int e = 0; e < experts_per_device; e++) {
        for (int w = 0; w < world_size; w++) {
          for (int c = 0; c < capacity; c++) {
            int eo_row = e * cap_total + w * capacity + c;
            int send_row = w * slots_per_device + e * capacity + c;
            std::memcpy(
                send_bytes + send_row * D * elem_size,
                eo_bytes + eo_row * D * elem_size,
                D * elem_size);
          }
        }
      }

      array recv_arr(Shape{total_slots, D}, dtype, nullptr, {});
      recv_arr.set_data(allocator::malloc(total_bytes));

      grp.raw_group()->blocking_all_to_all(send_arr, recv_arr);

      // recv_arr: [total_slots, D] flat, layout: [W, E, C, D]
      // route_idx uses NEW layout: flat_idx = local_expert * cap_total +
      // dest_rank * cap + pos Map new route_idx -> old recv_row:
      //   local_expert = flat_idx / cap_total
      //   dest_rank    = (flat_idx % cap_total) / capacity
      //   pos          = (flat_idx % cap_total) % capacity
      //   recv_row     = dest_rank * slots_per_device + local_expert * capacity
      //   + pos

      switch (dtype) {
        case float32: {
          const auto* recv_f = static_cast<const float*>(recv_arr.data<void>());
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);

          for (int n = 0; n < N; n++) {
            float* dst = out_f + n * D;
            std::fill(dst, dst + D, 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                    local_expert * capacity + pos;
                const float* src = recv_f + recv_row * D;
                for (int d = 0; d < D; d++)
                  dst[d] += w * src[d];
              }
            }
            if (!has_valid) {
              std::memcpy(dst, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          const auto* recv_h =
              static_cast<const float16_t*>(recv_arr.data<void>());
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);

          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                    local_expert * capacity + pos;
                const float16_t* src = recv_h + recv_row * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            float16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) {
                dst[d] = float16_t(accum[d]);
              }
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          const auto* recv_h =
              static_cast<const bfloat16_t*>(recv_arr.data<void>());
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);

          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                    local_expert * capacity + pos;
                const bfloat16_t* src = recv_h + recv_row * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            bfloat16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) {
                dst[d] = bfloat16_t(accum[d]);
              }
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error(
              "[MoeCombineExchange] Unsupported dtype. Use float32, float16, or bfloat16.");
      }
      // send_arr and recv_arr go out of scope here; their allocator memory
      // is freed via the array destructor.
    }
  });

  // Keep input arrays alive until the dispatched lambda has executed
  enc.add_temporary(expert_out);
  enc.add_temporary(route_idx);
  enc.add_temporary(weights_in);
  enc.add_temporary(orig_tok);
}
} // namespace mlx::core::distributed
