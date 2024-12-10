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

namespace mx = mlx::core;
namespace nb = nanobind;

std::string buffer_format(const mx::array& a) {
  // https://docs.python.org/3.10/library/struct.html#format-characters
  switch (a.dtype()) {
    case mx::bool_:
      return "?";
    case mx::uint8:
      return "B";
    case mx::uint16:
      return "H";
    case mx::uint32:
      return "I";
    case mx::uint64:
      return "Q";
    case mx::int8:
      return "b";
    case mx::int16:
      return "h";
    case mx::int32:
      return "i";
    case mx::int64:
      return "q";
    case mx::float16:
      return "e";
    case mx::float32:
      return "f";
    case mx::bfloat16:
      return "B";
    case mx::complex64:
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
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;

  buffer_info(
      std::string format,
      std::vector<Py_ssize_t> shape_in,
      std::vector<Py_ssize_t> strides_in)
      : format(std::move(format)),
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
  auto a = nb::cast<mx::array>(nb::handle(obj));

  {
    nb::gil_scoped_release nogil;
    a.eval();
  }

  std::vector<Py_ssize_t> shape(a.shape().begin(), a.shape().end());
  std::vector<Py_ssize_t> strides(a.strides().begin(), a.strides().end());
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
  view->len = a.nbytes();
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
