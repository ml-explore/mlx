// Copyright © 2024 Apple Inc.
#pragma once
#include <optional>

#include <nanobind/nanobind.h>

#include "mlx/array.h"
#include "mlx/utils.h"

// Only defined in >= Python 3.9
// https://github.com/python/cpython/blob/f6cdc6b4a191b75027de342aa8b5d344fb31313e/Include/typeslots.h#L2-L3
#ifndef Py_bf_getbuffer
#define Py_bf_getbuffer 1
#define Py_bf_releasebuffer 2
#endif

namespace nb = nanobind;
using namespace mlx::core;

std::string buffer_format(const array& a) {
  // https://docs.python.org/3.10/library/struct.html#format-characters
  switch (a.dtype()) {
    case bool_:
      return "?";
    case uint8:
      return "B";
    case uint16:
      return "H";
    case uint32:
      return "I";
    case uint64:
      return "Q";
    case int8:
      return "b";
    case int16:
      return "h";
    case int32:
      return "i";
    case int64:
      return "q";
    case float16:
      return "e";
    case float32:
      return "f";
    case bfloat16:
      return "B";
    case complex64:
      return "Zf\0";
    default: {
      std::ostringstream os;
      os << "bad dtype: " << a.dtype();
      throw std::runtime_error(os.str());
    }
  }
}

struct buffer_info {
  std::string format;
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;

  buffer_info(
      const std::string& format,
      std::vector<ssize_t> shape_in,
      std::vector<ssize_t> strides_in)
      : format(format),
        shape(std::move(shape_in)),
        strides(std::move(strides_in)) {}

  buffer_info(const buffer_info&) = delete;
  buffer_info& operator=(const buffer_info&) = delete;

  buffer_info(buffer_info&& other) noexcept {
    (*this) = std::move(other);
  }

  buffer_info& operator=(buffer_info&& rhs) noexcept {
    format = std::move(rhs.format);
    shape = std::move(rhs.shape);
    strides = std::move(rhs.strides);
    return *this;
  }
};

extern "C" inline int getbuffer(PyObject* obj, Py_buffer* view, int flags) {
  std::memset(view, 0, sizeof(Py_buffer));
  auto a = nb::cast<array>(nb::handle(obj));

  if (!a.is_evaled()) {
    nb::gil_scoped_release nogil;
    a.eval();
  }

  std::vector<ssize_t> shape(a.shape().begin(), a.shape().end());
  std::vector<ssize_t> strides(a.strides().begin(), a.strides().end());
  for (auto& s : strides) {
    s *= a.itemsize();
  }
  buffer_info* info =
      new buffer_info(buffer_format(a), std::move(shape), std::move(strides));

  view->obj = obj;
  view->ndim = a.ndim();
  view->internal = info;
  view->buf = a.data<void>();
  view->itemsize = a.itemsize();
  view->len = a.size();
  view->readonly = false;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    view->format = const_cast<char*>(info->format.c_str());
  }
  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
    view->strides = info->strides.data();
    view->shape = info->shape.data();
  }
  Py_INCREF(view->obj);
  return 0;
}

extern "C" inline void releasebuffer(PyObject*, Py_buffer* view) {
  delete (buffer_info*)view->internal;
}
