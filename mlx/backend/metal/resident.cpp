// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/utils.h"

namespace mlx::core::metal {

ResidencySets::ResidencySets(MTL::Device* d) {
  if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
    return; // enabled_ stays false, so everything below is a no-op
  }
  if (__builtin_available(macOS 15, iOS 18, *)) {
    device_ = d;
    enabled_ = true;
    int pct = env::residency_set_max_pct();
    if (pct <= 0 || pct >= 100) {
      max_bytes_per_set_ = 0; // a single set holds everything
    } else {
      size_t ws = static_cast<size_t>(d->recommendedMaxWorkingSetSize());
      // Floor the cap so a small working set or a small percentage cannot ask
      // for an absurd number of sets. 64 MiB comfortably exceeds the
      // allocator's heap.
      max_bytes_per_set_ = std::max<size_t>(
          (ws / 100) * static_cast<size_t>(pct), size_t(64) << 20);
    }
    std::lock_guard<std::mutex> lk(mtx_);
    // Set 0 always exists and is the fallback when a later set cannot be
    // made, so failing to create it is fatal.
    NS::Error* error = nullptr;
    if (!add_set_locked(&error)) {
      std::ostringstream msg;
      msg << "[metal::Device] Unable to construct residency set.\n";
      if (error) {
        msg << error->localizedDescription()->utf8String() << "\n";
      }
      throw std::runtime_error(msg.str());
    }
  }
}

ResidencySets::~ResidencySets() = default;

bool ResidencySets::add_set_locked(NS::Error** error_out) {
  NS::SharedPtr<MTL::ResidencySet> set;
  if (__builtin_available(macOS 15, iOS 18, *)) {
    auto pool = new_scoped_memory_pool();
    auto desc = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
    NS::Error* error = nullptr;
    set = NS::TransferPtr(device_->newResidencySet(desc, &error));
    if (set) {
      // A standing request, so allocations added to this set later are
      // covered without requesting residency again on every insert.
      set->requestResidency();
    } else if (error_out) {
      *error_out = error;
    }
  }
  if (!set) {
    return false;
  }
  sets_.push_back(Set{std::move(set), 0});
  num_sets_.store(sets_.size(), std::memory_order_release);
  if (env::residency_debug()) {
    fprintf(
        stderr,
        "[residency] created residency set %zu (max_bytes_per_set=%zu MB)\n",
        sets_.size() - 1,
        max_bytes_per_set_ >> 20);
  }
  return true;
}

uint32_t ResidencySets::choose_set_locked(size_t bytes) {
  if (max_bytes_per_set_ == 0) {
    return 0;
  }
  const uint32_t none = static_cast<uint32_t>(sets_.size());
  uint32_t empty = none;
  uint32_t emptiest = 0;
  for (uint32_t i = 0; i < sets_.size(); ++i) {
    const size_t size = sets_[i].size;
    if (size + bytes <= max_bytes_per_set_) {
      return i; // first fit
    }
    // Only reached by an allocation larger than the cap, which needs a set of
    // its own. Reusing an emptied one keeps the set count from growing.
    if (i != 0 && size == 0 && empty == none) {
      empty = i;
    }
    if (size < sets_[emptiest].size) {
      emptiest = i;
    }
  }
  if (empty != none) {
    return empty;
  }
  if (sets_.size() < kMaxSets && add_set_locked()) {
    return static_cast<uint32_t>(sets_.size() - 1);
  }
  // At the cap, or the driver would not make another set. Filling the emptiest
  // set keeps them evenly sized, so each one still holds a bounded fraction of
  // the wired bytes.
  return emptiest;
}

uint32_t ResidencySets::add_to_set_locked(
    const MTL::Allocation* buf,
    Placement& at) {
  uint32_t idx = choose_set_locked(at.bytes);
  auto& s = sets_[idx];
  s.set->addAllocation(buf);
  s.size += at.bytes;
  total_wired_ += at.bytes;
  at.set_id = idx;
  return idx;
}

void ResidencySets::remove_from_set_locked(
    const MTL::Allocation* buf,
    Placement& at) {
  auto& s = sets_[at.set_id];
  s.set->removeAllocation(buf);
  // Subtract the size recorded at insert; allocatedSize() is never read back.
  s.size -= at.bytes;
  total_wired_ -= at.bytes;
  at.set_id = kNoSet;
}

void ResidencySets::insert(MTL::Allocation* buf) {
  if (!enabled_) {
    return;
  }
  const size_t bytes = buf->allocatedSize();
  std::lock_guard<std::mutex> lk(mtx_);

  auto [it, inserted] = buf_to_set_.try_emplace(buf, Placement{kNoSet, bytes});
  if (!inserted) {
    assert(false && "allocation is already tracked");
    return;
  }
  // Stay within the wired limit. The excess is tracked but left out of any
  // set, and is added to one by resize() if the limit is raised later.
  if (total_wired_ + bytes > capacity_) {
    return;
  }
  uint32_t idx = add_to_set_locked(buf, it->second);
  sets_[idx].set->commit();
}

void ResidencySets::erase(MTL::Allocation* buf) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = buf_to_set_.find(buf);
  if (it == buf_to_set_.end()) {
    assert(false && "erasing an allocation that was never inserted");
    return;
  }
  if (it->second.set_id != kNoSet) {
    const uint32_t idx = it->second.set_id;
    remove_from_set_locked(buf, it->second);
    sets_[idx].set->commit();
  }
  buf_to_set_.erase(it);
}

void ResidencySets::resize(size_t size) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lk(mtx_);
  if (capacity_ == size) {
    return;
  }
  capacity_ = size;

  auto pool = new_scoped_memory_pool();
  std::vector<bool> touched(sets_.size(), false);
  // The loops below only mutate map values, never insert or erase, so the
  // iterators stay valid across the whole walk.
  if (total_wired_ < capacity_) {
    // The budget grew: add allocations that now fit.
    for (auto& [buf, at] : buf_to_set_) {
      if (at.set_id != kNoSet || total_wired_ + at.bytes > capacity_) {
        continue;
      }
      uint32_t idx = add_to_set_locked(buf, at);
      if (idx >= touched.size()) {
        touched.resize(sets_.size(), false); // a new set was made
      }
      touched[idx] = true;
    }
  } else {
    // The budget shrank: remove allocations until we are back under it.
    for (auto& [buf, at] : buf_to_set_) {
      if (total_wired_ <= capacity_) {
        break;
      }
      if (at.set_id == kNoSet) {
        continue;
      }
      touched[at.set_id] = true;
      remove_from_set_locked(buf, at);
    }
  }
  for (size_t i = 0; i < touched.size(); ++i) {
    if (touched[i]) {
      sets_[i].set->commit();
    }
  }
}

void ResidencySets::attach_new_sets(MTL::CommandQueue* q, uint64_t& attached) {
  // Lock-free when there is nothing new, which is the common case. It also
  // covers the disabled case, where the count stays 0.
  if (num_sets_.load(std::memory_order_acquire) == attached) {
    return;
  }
  // Attach the new sets and record how far we got under a single lock hold,
  // so a set created in between cannot be missed.
  std::lock_guard<std::mutex> lk(mtx_);
  std::vector<const MTL::ResidencySet*> sets;
  sets.reserve(sets_.size() - attached);
  for (uint64_t id = attached; id < sets_.size(); ++id) {
    sets.push_back(sets_[id].set.get());
  }
  q->addResidencySets(sets.data(), sets.size());
  attached = sets_.size();
}

} // namespace mlx::core::metal
