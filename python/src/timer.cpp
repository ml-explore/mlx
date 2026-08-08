// Copyright © 2026 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx/timer.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"
#include "python/src/trees.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

mx::Stream timer_stream(mx::StreamOrDevice stream) {
  if (std::holds_alternative<std::monostate>(stream)) {
    return mx::default_stream(mx::Device::gpu);
  }
  return mx::to_stream(stream);
}

class PyTimer {
 public:
  explicit PyTimer(mx::StreamOrDevice stream)
      : timer_(timer_stream(std::move(stream))) {}

  nb::object start(const nb::args& args) {
    if (started_) {
      throw std::runtime_error("[Timer.start] The timer has already started.");
    }
    auto outputs = marker(args, true);
    started_ = true;
    return outputs;
  }

  nb::object stop(const nb::args& args) {
    if (!started_) {
      throw std::runtime_error("[Timer.stop] The timer has not started.");
    }
    if (stopped_) {
      throw std::runtime_error("[Timer.stop] The timer has already stopped.");
    }
    auto outputs = marker(args, false);
    timed_outputs_ = tree_flatten(outputs);
    stopped_ = true;
    return outputs;
  }

  double elapsed_time() {
    if (!stopped_) {
      throw std::runtime_error(
          "[Timer.elapsed_time] The timer has not stopped.");
    }
    mx::eval(timed_outputs_);
    auto elapsed = timer_.elapsed_time();
    timed_outputs_.clear();
    return elapsed;
  }

  const mx::Stream& stream() const {
    return timer_.stream();
  }

 private:
  nb::object marker(const nb::args& args, bool start) {
    if (args.empty()) {
      throw std::invalid_argument(
          start ? "[Timer.start] Expected at least one array."
                : "[Timer.stop] Expected at least one array.");
    }
    auto inputs = tree_flatten(args);
    auto outputs = start ? mx::timer_start(inputs, timer_)
                         : mx::timer_stop(inputs, timer_);
    if (args.size() == 1) {
      return tree_unflatten(nb::borrow<nb::object>(args[0]), outputs);
    }
    return tree_unflatten(args, outputs);
  }

  mx::Timer timer_;
  std::vector<mx::array> timed_outputs_;
  bool started_{false};
  bool stopped_{false};
};

} // namespace

void init_timer(nb::module_& m) {
  nb::class_<PyTimer>(m, "Timer", R"pbdoc(
        Measure the GPU execution time of a lazy subgraph.

        :meth:`start` and :meth:`stop` insert pass-through timing markers in
        the graph. Operations which produce the inputs to :meth:`start` are
        excluded from the measured interval.

        Args:
            stream (Stream or Device, optional): GPU stream on which to place
                the timing markers. If ``None``, use the default GPU stream.
                Default: ``None``.
      )pbdoc")
      .def(nb::init<mx::StreamOrDevice>(), "stream"_a = nb::none())
      .def_prop_ro("stream", &PyTimer::stream)
      .def(
          "start",
          &PyTimer::start,
          nb::sig("def start(self, *args)"),
          R"pbdoc(
          Insert a start timing marker before one or more arrays.

          The returned arrays share storage with the inputs. Operations that
          consume them are ordered after the start marker.

          Args:
              *args: Arrays or array pytrees marking the start dependencies.

          Returns:
              A single array or pytree for one argument, otherwise a tuple.
      )pbdoc")
      .def(
          "stop",
          &PyTimer::stop,
          nb::sig("def stop(self, *args)"),
          R"pbdoc(
          Insert an end timing marker after one or more arrays.

          Evaluating the returned arrays schedules the measured operations and
          the end marker.

          Args:
              *args: Arrays or array pytrees marking the end dependencies.

          Returns:
              A single array or pytree for one argument, otherwise a tuple.
      )pbdoc")
      .def(
          "elapsed_time",
          &PyTimer::elapsed_time,
          nb::call_guard<nb::gil_scoped_release>(),
          R"pbdoc(
          Evaluate the timed subgraph and return its GPU time.

          This method schedules the end marker if necessary, then blocks the
          CPU until it completes.

          Returns:
              float: GPU elapsed time in milliseconds.
      )pbdoc");
}
