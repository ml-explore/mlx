// Copyright © 2024 Apple Inc.

#include <sys/sysctl.h>
#include <cstddef>

#include "mlx/backend/metal/resident.h"
#include "mlx/backend/metal/metal_impl.h"

namespace mlx::core::metal {

// TODO maybe worth including tvos / visionos
#define supported __builtin_available(macOS 15, iOS 18, *)

// Trying to create a residency set in a VM leads to errors.
static bool in_vm() {
  auto check_vm = []() {
    int hv_vmm_present = 0;

    std::size_t len = sizeof(hv_vmm_present);
    sysctlbyname("kern.hv_vmm_present", &hv_vmm_present, &len, NULL, 0);

    return hv_vmm_present;
  };

  static int in_vm = check_vm();
  return in_vm;
}

ResidencySet::ResidencySet(MTL::Device* d) {
  if (supported && !in_vm()) {
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
  if (supported && !in_vm()) {
    if (wired_set_->allocatedSize() + buf->allocatedSize() <= capacity_) {
      wired_set_->addAllocation(buf);
      wired_set_->commit();
      wired_set_->requestResidency();
    } else {
      unwired_set_.insert(buf);
    }
  }
}

void ResidencySet::erase(MTL::Allocation* buf) {
  if (supported && !in_vm()) {
    if (auto it = unwired_set_.find(buf); it != unwired_set_.end()) {
      unwired_set_.erase(it);
    } else {
      wired_set_->removeAllocation(buf);
      wired_set_->commit();
    }
  }
}

void ResidencySet::resize(size_t size) {
  if (supported && !in_vm()) {
    if (capacity_ == size) {
      return;
    }
    capacity_ = size;

    size_t current_size = wired_set_->allocatedSize();

    if (current_size < size) {
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
}

ResidencySet::~ResidencySet() {
  if (supported && !in_vm()) {
    wired_set_->release();
  }
}

} // namespace mlx::core::metal
