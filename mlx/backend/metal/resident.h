// Copyright © 2024 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <Metal/Metal.hpp>

namespace mlx::core::metal {

// Keeps allocations GPU-resident, up to the wired limit set by
// `set_wired_limit` (0 by default, i.e. nothing is wired unless asked for).
//
// Within that budget the allocations are distributed over several size-capped
// MTL::ResidencySets rather than one large one. macOS makes residency decisions
// per residency set, so when a set loses residency under GPU memory pressure
// only the allocations in that set have to be made resident again. Capping the
// size of each set bounds the cost of one such event. Every set holds a
// standing requestResidency() and is attached to every command queue.
//
// MLX_RESIDENCY_SET_MAX_PCT (env::residency_set_max_pct) sets the per-set cap.
// The total wired budget is unaffected by it: that is still `set_wired_limit`.
class ResidencySets {
 public:
  ResidencySets(MTL::Device* d);
  ~ResidencySets();

  ResidencySets(const ResidencySets&) = delete;
  ResidencySets& operator=(const ResidencySets&) = delete;

  // Called with the allocator's mutex held.
  void insert(MTL::Allocation* buf);
  void erase(MTL::Allocation* buf);

  void resize(size_t size);

  bool enabled() const {
    return enabled_;
  }

  // Attaches the sets this queue has not seen yet. Called from the encoder
  // thread before every command-buffer commit, because Metal locks a command
  // buffer's residency at commit time.
  void attach_new_sets(MTL::CommandQueue* q, uint64_t& attached);

  // Total bytes currently wired across all sets.
  size_t wired_size() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return total_wired_;
  }
  size_t num_sets() const {
    return num_sets_.load(std::memory_order_acquire);
  }

  // Testing only: sets the per-set cap for subsequent inserts, bypassing the
  // size floor so tests can reach the multi-set paths cheaply. 0 selects the
  // single-set layout.
  void set_max_bytes_per_set(size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    max_bytes_per_set_ = bytes;
  }
  size_t max_bytes_per_set() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return max_bytes_per_set_;
  }

 private:
  // A set id of kNoSet means the allocation is tracked but is not in any
  // set, because it did not fit in the wired limit.
  static constexpr uint32_t kNoSet = UINT32_MAX;
  // A command queue accepts a limited number of residency sets and every set
  // is attached to every queue, so the number of sets is capped. Once the cap
  // is reached allocations go to the emptiest set, which then grows past
  // max_bytes_per_set_. The limit is from the Metal feature set tables:
  // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
  static constexpr uint32_t kMaxSets = 32;

  struct Set {
    NS::SharedPtr<MTL::ResidencySet> set;
    size_t size{0};
  };

  // The set an allocation lives in and the size it was inserted with. Sizes
  // are recorded rather than read back from the allocation at erase, so the
  // running totals cannot drift.
  struct Placement {
    uint32_t set_id;
    size_t bytes;
  };

  // The following all require mtx_. add_set_locked reports whether the driver
  // gave us a set; the new set is the last one. add_to_set_locked returns the
  // set it used. Neither it nor remove_from_set_locked commits, so a bulk
  // resize costs one commit per set instead of one per allocation.
  bool add_set_locked(NS::Error** error = nullptr);
  uint32_t choose_set_locked(size_t bytes);
  uint32_t add_to_set_locked(const MTL::Allocation* buf, Placement& at);
  void remove_from_set_locked(const MTL::Allocation* buf, Placement& at);

  MTL::Device* device_{nullptr};
  bool enabled_{false};

  // Per-set cap in bytes (0 for a single set) and the total wired budget.
  size_t max_bytes_per_set_{0};
  size_t capacity_{0};
  size_t total_wired_{0};

  // Sets indexed by id. Append-only, so ids stay valid and dense.
  std::vector<Set> sets_;
  // Every tracked allocation, wired or not.
  std::unordered_map<const MTL::Allocation*, Placement> buf_to_set_;

  // sets_.size(), published so a queue can check for new sets without
  // taking the lock.
  std::atomic<uint64_t> num_sets_{0};
  mutable std::mutex mtx_;
};

} // namespace mlx::core::metal
