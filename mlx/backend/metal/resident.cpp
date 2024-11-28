// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/resident.h"
#include "mlx/backend/metal/metal_impl.h"

namespace mlx::core::metal {

ResidencySet::ResidencySet(MTL::Device* d) {
  if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
    return;
  } else if (__builtin_available(macOS 15, iOS 18, *)) {
    auto pool = new_scoped_memory_pool();
    auto desc = MTL::ResidencySetDescriptor::alloc()->init();
    NS::Error* error;
    wired_set_ = d->newResidencySet(desc, &error);
    desc->release();
    if (!wired_set_) {
      std::ostringstream msg;
      msg << "[metal::Device] Unable to construct residency set.\n";
      if (error) {
        msg << error->localizedDescription()->utf8String() << "\n";
      }
      throw std::runtime_error(msg.str());
    }
  }
}

void ResidencySet::insert(MTL::Allocation* buf) {
  if (!wired_set_) {
    return;
  }
  if (wired_set_->allocatedSize() + buf->allocatedSize() <= capacity_) {
    wired_set_->addAllocation(buf);
    wired_set_->commit();
    wired_set_->requestResidency();
  } else {
    unwired_set_.insert(buf);
  }
}

void ResidencySet::erase(MTL::Allocation* buf) {
  if (!wired_set_) {
    return;
  }
  if (auto it = unwired_set_.find(buf); it != unwired_set_.end()) {
    unwired_set_.erase(it);
  } else {
    wired_set_->removeAllocation(buf);
    wired_set_->commit();
  }
}

void ResidencySet::resize(size_t size) {
  if (!wired_set_) {
    return;
  }

  if (capacity_ == size) {
    return;
  }
  capacity_ = size;

  size_t current_size = wired_set_->allocatedSize();

  if (current_size < size) {
    auto pool = new_scoped_memory_pool();
    // Add unwired allocations to the set
    for (auto it = unwired_set_.begin(); it != unwired_set_.end();) {
      auto buf_size = (*it)->allocatedSize();
      if (current_size + buf_size > size) {
        it++;
      } else {
        current_size += buf_size;
        wired_set_->addAllocation(*it);
        unwired_set_.erase(it++);
      }
    }
    wired_set_->commit();
    wired_set_->requestResidency();
  } else if (current_size > size) {
    auto pool = new_scoped_memory_pool();
    // Remove wired allocations until under capacity
    auto allocations = wired_set_->allAllocations();
    auto num_allocations = wired_set_->allocationCount();
    for (int i = 0; i < num_allocations && current_size > size; ++i) {
      auto buf = static_cast<const MTL::Allocation*>(allocations->object(i));
      wired_set_->removeAllocation(buf);
      current_size -= buf->allocatedSize();
      unwired_set_.insert(buf);
    }
    wired_set_->commit();
  }
}

ResidencySet::~ResidencySet() {
  if (wired_set_) {
    auto pool = new_scoped_memory_pool();
    wired_set_->release();
  }
}

} // namespace mlx::core::metal
