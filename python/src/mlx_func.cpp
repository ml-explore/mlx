// Copyright © 2025 Apple Inc.

#include "python/src/mlx_func.h"

// A garbage collected function which wraps nb::cpp_function
// See https://github.com/wjakob/nanobind/discussions/919

struct gc_func {
  PyObject_HEAD
      // Vector call implementation that forwards calls to nanobind
      PyObject* (*vectorcall)(PyObject*, PyObject* const*, size_t, PyObject*);
  // The function itself
  PyObject* func;
  // A non-owning reference to dependencies owned by 'func'
  std::vector<PyObject*> deps;
};

int gc_func_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  gc_func* w = (gc_func*)self;
  Py_VISIT(w->func);
  for (auto d : w->deps) {
    Py_VISIT(d);
  }
  Py_VISIT(Py_TYPE(self));
  return 0;
};

int gc_func_tp_clear(PyObject* self) {
  gc_func* w = (gc_func*)self;
  Py_CLEAR(w->func);
  return 0;
}

PyObject* gc_func_get_doc(PyObject* self, void*) {
  return PyObject_GetAttrString(((gc_func*)self)->func, "__doc__");
}

PyObject* gc_func_get_sig(PyObject* self, void*) {
  return PyObject_GetAttrString(((gc_func*)self)->func, "__nb_signature__");
}

PyObject* gc_func_vectorcall(
    PyObject* self,
    PyObject* const* args,
    size_t nargs,
    PyObject* kwnames) {
  return PyObject_Vectorcall(((gc_func*)self)->func, args, nargs, kwnames);
}

void gc_func_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  Py_XDECREF(((gc_func*)self)->func);
  PyObject_GC_Del(self);
}

static PyMemberDef gc_func_members[] = {
    {"__vectorcalloffset__",
     T_PYSSIZET,
     (Py_ssize_t)offsetof(gc_func, vectorcall),
     READONLY,
     nullptr},
    {nullptr, 0, 0, 0, nullptr}};

static PyGetSetDef gc_func_getset[] = {
    {"__doc__", gc_func_get_doc, nullptr, nullptr, nullptr},
    {"__nb_signature__", gc_func_get_sig, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

static PyObject* gc_func_getattro(PyObject* self, PyObject* name_) {
  gc_func* w = (gc_func*)self;
  auto f = PyCFunction(PyType_GetSlot(Py_TYPE(w->func), Py_tp_getattro));
  return f(w->func, name_);
}

// Table of custom type slots we want to install
PyType_Slot gc_func_slots[] = {
    {Py_tp_traverse, (void*)gc_func_tp_traverse},
    {Py_tp_clear, (void*)gc_func_tp_clear},
    {Py_tp_getset, (void*)gc_func_getset},
    {Py_tp_getattro, (void*)gc_func_getattro},
    {Py_tp_members, (void*)gc_func_members},
    {Py_tp_call, (void*)PyVectorcall_Call},
    {Py_tp_dealloc, (void*)gc_func_dealloc},
    {0, 0}};

static PyType_Spec gc_func_spec = {
    /* .name = */ "mlx.gc_func",
    /* .basicsize = */ (int)sizeof(gc_func),
    /* .itemsize = */ 0,
    /* .flags = */ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | NB_HAVE_VECTORCALL,
    /* .slots = */ gc_func_slots};

static PyTypeObject* gc_func_tp = nullptr;

nb::callable mlx_func(nb::object func, std::vector<PyObject*> deps) {
  gc_func* r = (gc_func*)PyType_GenericAlloc(gc_func_tp, 0);
  r->func = func.inc_ref().ptr();
  r->deps = std::move(deps);
  r->vectorcall = gc_func_vectorcall;
  return nb::steal<nb::callable>((PyObject*)r);
}

void init_mlx_func(nb::module_& m) {
  gc_func_tp = (PyTypeObject*)PyType_FromSpec(&gc_func_spec);
  if (!gc_func_tp) {
    nb::raise("Could not register MLX function type.");
  }
}
