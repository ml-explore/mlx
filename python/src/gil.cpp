// Copyright © 2026 Apple Inc.

#include "python/src/gil.h"

#include <mutex>
#include <vector>

namespace {

struct DeferredReleases {
  std::mutex mtx;
  // Raw PyObject* because dropping a reference needs the GIL, which the
  // queueing thread does not hold. Whatever is still queued at exit is leaked.
  std::vector<PyObject*> refs;
};

DeferredReleases& deferred() {
  // Leak - see Scheduler singleton comment in scheduler.cpp. Threads exiting
  // late can queue here after static destruction would have run.
  static DeferredReleases* deferred_ = new DeferredReleases;
  return *deferred_;
}

} // namespace

bool py_is_finalizing() {
#if PY_VERSION_HEX >= 0x030D0000
  return Py_IsFinalizing();
#else
  return _Py_IsFinalizing();
#endif
}

void defer_release(nb::object&& obj) {
  PyObject* ptr = obj.release().ptr();
  if (ptr == nullptr) {
    return;
  }
  auto& d = deferred();
  std::lock_guard<std::mutex> lk(d.mtx);
  d.refs.push_back(ptr);
}

void drain_deferred_releases() {
  auto& d = deferred();
  std::vector<PyObject*> pending;
  {
    std::lock_guard<std::mutex> lk(d.mtx);
    pending.swap(d.refs);
  }
  for (PyObject* ptr : pending) {
    Py_DECREF(ptr);
  }
}
