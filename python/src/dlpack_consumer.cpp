// Copyright © 2026 Apple Inc.

#include "python/src/dlpack_consumer.h"

#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/dtype.h"
#include "python/src/convert.h"
#include "python/src/dlpack_format.h"

mx::Dtype dlpack_to_mlx_dtype(const nb::dlpack::dtype& dt) {
  if (dt.lanes != 1) {
    std::ostringstream msg;
    msg << "[array] DLPack tensors with lanes != 1 are not supported "
        << "(got lanes=" << dt.lanes << ").";
    throw std::invalid_argument(msg.str());
  }
  using Code = nb::dlpack::dtype_code;
  switch (static_cast<Code>(dt.code)) {
    case Code::Bool:
      if (dt.bits == 8)
        return mx::bool_;
      break;
    case Code::Int:
      switch (dt.bits) {
        case 8:
          return mx::int8;
        case 16:
          return mx::int16;
        case 32:
          return mx::int32;
        case 64:
          return mx::int64;
      }
      break;
    case Code::UInt:
      switch (dt.bits) {
        case 8:
          return mx::uint8;
        case 16:
          return mx::uint16;
        case 32:
          return mx::uint32;
        case 64:
          return mx::uint64;
      }
      break;
    case Code::Float:
      switch (dt.bits) {
        case 16:
          return mx::float16;
        case 32:
          return mx::float32;
        case 64:
          return mx::float64;
      }
      break;
    case Code::Bfloat:
      if (dt.bits == 16)
        return mx::bfloat16;
      break;
    case Code::Complex:
      if (dt.bits == 64)
        return mx::complex64;
      break;
    default:
      break;
  }
  std::ostringstream msg;
  msg << "[array] Unsupported DLPack dtype: code=" << int(dt.code)
      << ", bits=" << int(dt.bits) << ".";
  throw std::invalid_argument(msg.str());
}

mx::Shape validate_and_extract_shape(const nb::dlpack::dltensor& t) {
  if (t.ndim < 0) {
    throw std::invalid_argument("[array] ndim must be non-negative.");
  }
  if (t.ndim > 0 && t.shape == nullptr) {
    throw std::invalid_argument(
        "[array] shape must not be null when ndim > 0.");
  }
  mx::Shape shape;
  shape.reserve(t.ndim);
  for (int i = 0; i < t.ndim; ++i) {
    if (t.shape[i] < 0) {
      throw std::invalid_argument("[array] shape dims must be non-negative.");
    }
    if (t.shape[i] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("[array] shape dim exceeds int32 range.");
    }
    shape.push_back(static_cast<int32_t>(t.shape[i]));
  }
  return shape;
}

bool is_row_contiguous(const mx::Shape& shape, const int64_t* strides) {
  if (strides == nullptr) {
    return true;
  }
  int64_t expected = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (strides[i] != expected) {
      return false;
    }
    if (shape[i] != 0 &&
        expected > std::numeric_limits<int64_t>::max() / shape[i]) {
      return false;
    }
    expected *= shape[i];
  }
  return true;
}

size_t checked_num_bytes(const mx::Shape& shape, mx::Dtype dtype) {
  size_t nelems = 1;
  for (auto dim : shape) {
    if (dim != 0 &&
        nelems >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::invalid_argument(
          "[array] shape element count overflows size_t.");
    }
    nelems *= static_cast<size_t>(dim);
  }
  if (dtype.size() != 0 &&
      nelems > std::numeric_limits<size_t>::max() / dtype.size()) {
    throw std::invalid_argument("[array] tensor byte size overflows.");
  }
  return nelems * dtype.size();
}

namespace {

struct ParsedCapsule {
  PyObject* capsule = nullptr;
  bool versioned = false;
  nb::dlpack::dltensor* tensor = nullptr;
  void* managed = nullptr; // typed by `versioned`
};

ParsedCapsule parse_capsule(PyObject* obj) {
  ParsedCapsule out;
  if (PyCapsule_IsValid(obj, "dltensor_versioned")) {
    out.versioned = true;
    auto* m = static_cast<dlpack_format::DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(obj, "dltensor_versioned"));
    if (m == nullptr) {
      throw std::invalid_argument(
          "[array] dltensor_versioned capsule is null.");
    }
    out.managed = static_cast<void*>(m);
    out.tensor = &m->dl_tensor;
  } else if (PyCapsule_IsValid(obj, "dltensor")) {
    out.versioned = false;
    auto* m = static_cast<dlpack_format::DLManagedTensor*>(
        PyCapsule_GetPointer(obj, "dltensor"));
    if (m == nullptr) {
      throw std::invalid_argument("[array] dltensor capsule is null.");
    }
    out.managed = static_cast<void*>(m);
    out.tensor = &m->dl_tensor;
  } else {
    throw std::invalid_argument(
        "[array] expected a PyCapsule named 'dltensor' or "
        "'dltensor_versioned'.");
  }
  out.capsule = obj;
  return out;
}

void mark_capsule_consumed(PyObject* capsule, bool versioned) {
  const char* used = versioned ? "used_dltensor_versioned" : "used_dltensor";
  if (PyCapsule_SetName(capsule, used) != 0 ||
      PyCapsule_SetDestructor(capsule, nullptr) != 0) {
    PyErr_Clear();
    throw std::runtime_error(
        "[array] failed to mark DLPack capsule as consumed.");
  }
}

mx::array build_cpu_array(nb::dlpack::dltensor& t, const mx::Shape& shape) {
  if (!is_row_contiguous(shape, t.strides)) {
    throw std::invalid_argument(
        "[array] non-row-contiguous DLPack strides are not supported "
        "for kDLCPU tensors yet.");
  }
  if (t.byte_offset != 0) {
    throw std::invalid_argument(
        "[array] kDLCPU capsule with non-zero byte_offset is not "
        "supported yet.");
  }
  auto dtype = dlpack_to_mlx_dtype(t.dtype);
  size_t nbytes = checked_num_bytes(shape, dtype);
  if (nbytes > 0 && t.data == nullptr) {
    throw std::invalid_argument(
        "[array] kDLCPU capsule has null data pointer.");
  }

  // Allocate a fresh mlx buffer and copy the producer's bytes in. This
  // mirrors the semantics of nd_array_to_mlx_contiguous for the kDLCPU
  // path. We use the (allocator::Buffer, Shape, Dtype, Deleter) overload to
  // get an array whose status() == Status::available immediately.
  auto buffer = mx::allocator::malloc(nbytes);
  if (nbytes > 0) {
    std::memcpy(static_cast<uint8_t*>(buffer.raw_ptr()), t.data, nbytes);
  }
  mx::array out(buffer, shape, dtype, mx::allocator::free);

  return out;
}

} // namespace

void DLPackOwner::invoke() {
  if (!active_ || mt_ == nullptr)
    return;
  if (versioned_) {
    auto* m = static_cast<dlpack_format::DLManagedTensorVersioned*>(mt_);
    if (m->deleter)
      m->deleter(m);
  } else {
    auto* m = static_cast<dlpack_format::DLManagedTensor*>(mt_);
    if (m->deleter)
      m->deleter(m);
  }
  mt_ = nullptr;
  active_ = false;
}

mx::array dlpack_to_mlx(nb::object obj) {
  // Accept either:
  //   * a PyCapsule (raw DLPack output),
  //   * an object that returns a PyCapsule from __dlpack__(),
  //   * an object whose __dlpack__() returns *another* object that is itself
  //     PEP-3118 / DLPack-compliant (e.g. nanobind's nb_ndarray wrapper that
  //     mlx returns from mx.array.__dlpack__). We unwrap up to N times.
  constexpr int kMaxUnwrap = 4;
  PyObject* raw = obj.ptr();
  nb::object current = obj; // own a reference for the chain

  for (int i = 0; i < kMaxUnwrap; ++i) {
    if (PyCapsule_CheckExact(raw)) {
      break;
    }
    if (!nb::hasattr(current, "__dlpack__")) {
      throw std::invalid_argument(
          "[array] expected a PyCapsule or an object exposing "
          "__dlpack__().");
    }
    current = current.attr("__dlpack__")();
    raw = current.ptr();
  }
  if (!PyCapsule_CheckExact(raw)) {
    throw std::invalid_argument(
        "[array] could not resolve input to a DLPack PyCapsule "
        "after repeated __dlpack__() calls.");
  }

  ParsedCapsule p = parse_capsule(raw);
  auto& t = *p.tensor;
  auto shape = validate_and_extract_shape(t);

  switch (t.device.device_type) {
    case dlpack_format::kDLCPU: {
      auto owner = std::make_shared<DLPackOwner>(p.versioned, p.managed);
      auto out = build_cpu_array(t, shape);
      mark_capsule_consumed(p.capsule, p.versioned);
      owner->activate();
      owner->invoke();
      return out;
    }
    case dlpack_format::kDLMetal: {
      auto owner = std::make_shared<DLPackOwner>(p.versioned, p.managed);
      auto out = build_dlpack_metal_array(t, owner);
      mark_capsule_consumed(p.capsule, p.versioned);
      owner->activate();
      return out;
    }
    case dlpack_format::kDLCUDA:
      throw std::invalid_argument(
          "[array] kDLCUDA tensors are not supported by MLX. Move the "
          "tensor to host memory or to a Metal-backed framework first.");
    default: {
      std::ostringstream msg;
      msg << "[array] unsupported DLPack device_type " << t.device.device_type
          << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}
