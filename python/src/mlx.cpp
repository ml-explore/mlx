// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>

#include "mlx/version.h"

#ifdef _MSC_VER
#include <csignal>
#include <cstdlib>
#endif

namespace mx = mlx::core;
namespace nb = nanobind;

void init_mlx_func(nb::module_&);
void init_array(nb::module_&);
void init_device(nb::module_&);
void init_stream(nb::module_&);
void init_metal(nb::module_&);
void init_cuda(nb::module_&);
void init_memory(nb::module_&);
void init_ops(nb::module_&);
void init_transforms(nb::module_&);
void init_random(nb::module_&);
void init_fft(nb::module_&);
void init_linalg(nb::module_&);
void init_constants(nb::module_&);
void init_fast(nb::module_&);
void init_distributed(nb::module_&);
void init_export(nb::module_&);

NB_MODULE(core, m) {
#ifdef _MSC_VER
  // Suppress MSVC CRT abort dialog boxes for testing/debugging.
  // This allows crashes to be handled programmatically rather than
  // requiring user interaction with a dialog.
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);

  // Install a SIGABRT handler that exits without a dialog.
  // This catches abort() calls from any CRT instance (including nanobind's).
  std::signal(SIGABRT, [](int) {
    std::_Exit(3); // Exit immediately without cleanup or dialogs
  });
#endif

  m.doc() = "mlx: A framework for machine learning on Apple silicon.";

  auto reprlib_fix = nb::module_::import_("mlx._reprlib_fix");
  nb::set_leak_warnings(false);

  init_mlx_func(m);
  init_device(m);
  init_stream(m);
  init_array(m);
  init_metal(m);
  init_cuda(m);
  init_memory(m);
  init_ops(m);
  init_transforms(m);
  init_random(m);
  init_fft(m);
  init_linalg(m);
  init_constants(m);
  init_fast(m);
  init_distributed(m);
  init_export(m);

  m.attr("__version__") = mx::version();
}
