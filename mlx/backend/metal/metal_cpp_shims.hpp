#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#if defined(__APPLE__)
#include <objc/runtime.h>
#include <objc/message.h>
#include <objc/objc.h>
#endif

namespace mlx::core::metal {

inline bool set_profiling_enabled(
    MTL::CommandBufferDescriptor* desc,
    bool enabled) {
#if defined(__APPLE__)
  if (desc == nullptr) {
    return false;
  }
  static SEL selector = sel_registerName("setProfilingEnabled:");
  if (selector == nullptr) {
    return false;
  }
  Class desc_cls = objc_getClass("MTLCommandBufferDescriptor");
  if (desc_cls == nullptr || !class_respondsToSelector(desc_cls, selector)) {
    return false;
  }
  using Setter = void (*)(void*, SEL, BOOL);
  reinterpret_cast<Setter>(objc_msgSend)(
      reinterpret_cast<void*>(desc), selector, enabled ? YES : NO);
  return true;
#else
  (void)desc;
  (void)enabled;
  return false;
#endif
}

} // namespace mlx::core::metal
