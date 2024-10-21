// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core::metal {

MTL::ResidencySet* setup_residency_set(MTL::Device* d) {
  auto desc = MTL::ResidencySetDescriptor::alloc()->init();
  NS::Error* error;
  auto residency_set = d->newResidencySet(desc, &error);
  desc->release();
  if (!residency_set) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to construct residency set.\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }
  return residency_set;
}

void Device::wire(std::vector<array> arrays) {
  for (auto& a : arrays) {
    residency_set_->addAllocation(
        static_cast<const MTL::Buffer*>(a.buffer().ptr()));
  }
  residency_set_->commit();
  //  residency_set_->requestResidency();
}

void Device::unwire(std::vector<array> arrays) {
  for (auto& a : arrays) {
    residency_set_->removeAllocation(
        static_cast<const MTL::Buffer*>(a.buffer().ptr()));
  }
  residency_set_->commit();
  //  residency_set_->endResidency();
}

} // namespace mlx::core::metal
