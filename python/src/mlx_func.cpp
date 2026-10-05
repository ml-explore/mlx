// Copyright © 2025 Apple Inc.

#include "python/src/mlx_func.h"

// The wrapper owns callbacks because free-threaded nanobind functions are
// immortal

struct gc_func {
  std::unique_ptr<PyFunction> func;

  // The original wrapped func
  PyObject* orig_func;
  // A non-owning reference to dependencies owned by 'func'
  std::vector<PyObject*> deps;
};

int gc_func_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (nb::inst_ready(self)) {
    for (auto d : nb::inst_ptr<gc_func>(self)->deps) {
      Py_VISIT(d);
    }
  }
  return 0;
}

int gc_func_tp_clear(PyObject* self) {
  auto* w = nb::inst_ptr<gc_func>(self);
  w->orig_func = nullptr;
  w->deps.clear();
  w->func.reset();
  return 0;
}

static PyObject* gc_func_getattro(PyObject* self, PyObject* name_) {
  return PyObject_GenericGetAttr(nb::inst_ptr<gc_func>(self)->orig_func, name_);
}

// Table of custom type slots we want to install
PyType_Slot gc_func_slots[] = {
    {Py_tp_traverse, (void*)gc_func_tp_traverse},
    {Py_tp_clear, (void*)gc_func_tp_clear},
    {Py_tp_getattro, (void*)gc_func_getattro},
    {0, 0}};

nb::callable mlx_func(
    std::unique_ptr<PyFunction> func,
    const nb::callable& orig_func,
    std::vector<PyObject*> deps) {
  deps.push_back(orig_func.ptr());
  return nb::borrow<nb::callable>(
      nb::cast(gc_func{std::move(func), orig_func.ptr(), std::move(deps)}));
}

void init_mlx_func(nb::module_& m) {
  nb::class_<gc_func>(m, "_gc_func", nb::type_slots(gc_func_slots))
      .def(
          "__call__",
          [](gc_func& self, nb::args args, nb::kwargs kwargs) {
            return (*self.func)(args, kwargs);
          })
      .freeze();
}
