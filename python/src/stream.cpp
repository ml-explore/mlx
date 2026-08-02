// Copyright © 2023-2024 Apple Inc.

#include <memory>
#include <sstream>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "mlx/event.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

thread_local std::vector<
    std::pair<mx::Stream, std::unique_ptr<mx::StreamContext>>>
    stream_contexts;

} // namespace

// Create the StreamContext on enter and delete on exit.
class PyStreamContext {
 public:
  PyStreamContext(mx::StreamOrDevice s) : _inner(nullptr) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error(
          "[StreamContext] Invalid argument, please specify a stream or device.");
    }
    _s = s;
  }

  void enter() {
    _inner = new mx::StreamContext(_s);
  }

  void exit() {
    if (_inner != nullptr) {
      delete _inner;
      _inner = nullptr;
    }
  }

 private:
  mx::StreamOrDevice _s;
  mx::StreamContext* _inner;
};

void init_stream(nb::module_& m) {
  nb::class_<mx::Stream>(
      m,
      "Stream",
      R"pbdoc(
      An in-order queue of work on a device.

      Evaluated operations on the same stream run in order. Operations on
      different streams may overlap. MLX operations are lazy, so assigning an
      operation to a stream does not submit it for execution.

      Args:
          device (Device, optional): The device for the stream. If ``None``,
              use the default device. Default: ``None``.
      )pbdoc")
      .def(
          nb::new_([](std::optional<mx::Device> d) {
            return mx::new_stream(d.value_or(mx::default_device()));
          }),
          "device"_a = nb::none())
      .def_ro("device", &mx::Stream::device)
      .def(
          "__repr__",
          [](const mx::Stream& s) {
            std::ostringstream os;
            os << s;
            return os.str();
          })
      .def(
          "__eq__",
          [](const mx::Stream& s, const nb::object& other) {
            return nb::isinstance<mx::Stream>(other) &&
                s == nb::cast<mx::Stream>(other);
          })
      .def(
          "query",
          &mx::query,
          R"pbdoc(
          Check whether work already scheduled on the stream has completed.

          This method does not block the CPU, evaluate lazy arrays, or submit
          pending stream work.

          Returns:
              bool: ``True`` if the stream has no incomplete work.
      )pbdoc")
      .def(
          "record_event",
          [](const mx::Stream& stream, nb::object event) {
            if (event.is_none()) {
              mx::Event recorded(stream, false);
              recorded.record(stream);
              return nb::cast(std::move(recorded), nb::rv_policy::move);
            }
            auto& recorded = nb::cast<mx::Event&>(event);
            recorded.record(stream);
            return event;
          },
          "event"_a = nb::none(),
          R"pbdoc(
          Record an event after work already scheduled on the stream.

          Recording on a Metal stream submits its current command buffer, but
          does not wait for completion. This method does not evaluate lazy
          arrays and is only supported on GPU streams.

          Args:
              event (Event, optional): The event to record. If ``None``,
                  allocate a synchronization-only event. Default: ``None``.

          Returns:
              Event: The recorded event.
      )pbdoc")
      .def(
          "synchronize",
          [](const mx::Stream& stream) { mx::synchronize(stream); },
          nb::call_guard<nb::gil_scoped_release>(),
          R"pbdoc(
          Submit pending stream work and block the CPU until it completes.

          This method does not evaluate lazy arrays.
      )pbdoc")
      .def(
          "wait_event",
          [](const mx::Stream& stream, mx::Event& event) {
            event.wait(stream);
          },
          "event"_a,
          R"pbdoc(
          Make future work submitted to the stream wait for an event.

          Only work scheduled on the stream after this call waits for the
          event. This method does not block the calling CPU thread.

          Args:
              event (Event): The event to wait for.
      )pbdoc")
      .def(
          "wait_stream",
          [](const mx::Stream& stream, const mx::Stream& other) {
            if (other.device == mx::Device::gpu) {
              mx::Event event(other, false);
              event.record(other);
              event.wait(stream);
            } else {
              mx::Event event(other);
              event.set_value(1);
              event.signal(other);
              event.wait(stream);
            }
          },
          "stream"_a,
          R"pbdoc(
          Make future work submitted to this stream wait for work already
          submitted to another stream.

          Work submitted to the other stream after this call is not included.
          This method does not block the calling CPU thread or evaluate lazy
          arrays.

          Args:
              stream (Stream): The other stream to wait for.
      )pbdoc")
      .def(
          "__enter__",
          [](mx::Stream& stream) -> mx::Stream& {
            stream_contexts.emplace_back(
                stream, std::make_unique<mx::StreamContext>(stream));
            return stream;
          },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](const mx::Stream& stream,
             const std::optional<nb::type_object>&,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&) {
            if (stream_contexts.empty() ||
                stream_contexts.back().first != stream) {
              throw std::runtime_error(
                  "[Stream::__exit__] Stream contexts must exit in order.");
            }
            stream_contexts.pop_back();
          },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());

  nb::class_<mx::ThreadLocalStream>(
      m,
      "ThreadLocalStream",
      R"pbdoc(
      A stream that will be unique per thread and can be used to run operations on a given device.
      )pbdoc")
      .def_ro("device", &mx::ThreadLocalStream::device)
      .def(
          "__repr__",
          [](const mx::ThreadLocalStream& s) {
            std::ostringstream os;
            os << "ThreadLocalStream(" << s.device << ", " << s.index << ")";
            return os.str();
          })
      .def(
          "__eq__",
          [](const mx::ThreadLocalStream& s, const nb::object& other) {
            return nb::isinstance<mx::ThreadLocalStream>(other) &&
                s == nb::cast<mx::ThreadLocalStream>(other);
          });

  nb::class_<mx::Event>(m, "Event", R"pbdoc(
        Query and record Stream status to identify or control dependencies across Stream and measure timing.

        Args:
            device (Device, optional): The GPU device for the event. If
                ``None``, use the default GPU device. Default: ``None``.
            enable_timing (bool, optional): Whether the event can measure
                elapsed time. Default: ``False``.
  )pbdoc")
      .def(
          nb::new_([](std::optional<mx::Device> d, bool enable_timing) {
            auto device = d.value_or(mx::Device(mx::Device::gpu));
            return mx::Event(mx::default_stream(device), enable_timing);
          }),
          "device"_a = nb::none(),
          nb::kw_only(),
          "enable_timing"_a = false)
      .def(
          "record",
          [](mx::Event& event, mx::StreamOrDevice s) {
            auto stream = std::holds_alternative<std::monostate>(s)
                ? mx::default_stream(event.stream().device)
                : mx::to_stream(s);
            event.record(stream);
          },
          "stream"_a = nb::none(),
          R"pbdoc(
          Record the event at the current position in the stream.

          Args:
              stream (Stream or Device, optional): The GPU stream or device to
                  record on. If ``None``, use the default stream for the
                  event's GPU device. Default: ``None``.
      )pbdoc")
      .def(
          "query",
          &mx::Event::query,
          R"pbdoc(
          Check whether all work captured by the event has completed.

          An event that has not been recorded is considered complete.

          Returns:
              bool: ``True`` if the event has completed.
      )pbdoc")
      .def(
          "wait",
          [](mx::Event& event, mx::StreamOrDevice s) {
            auto stream = std::holds_alternative<std::monostate>(s)
                ? mx::default_stream(event.stream().device)
                : mx::to_stream(s);
            event.wait(stream);
          },
          "stream"_a = nb::none(),
          R"pbdoc(
          Make future work on a stream wait for this event.

          This method does not block the CPU.

          Args:
              stream (Stream or Device, optional): The stream or device to
                  wait on. If ``None``, use the default stream for the event's
                  device. Default: ``None``.
      )pbdoc")
      .def(
          "synchronize",
          &mx::Event::synchronize,
          nb::call_guard<nb::gil_scoped_release>(),
          R"pbdoc(
          Wait for the event to complete.

          This method blocks the CPU until all work captured by the event has
          completed.
      )pbdoc")
      .def(
          "elapsed_time",
          &mx::Event::elapsed_time,
          "end_event"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          R"pbdoc(
          Wait for both events and return the elapsed time in milliseconds.

          Both events must be created with ``enable_timing=True``.

          Args:
              end_event (Event): The ending event.

          Returns:
              float: The elapsed time in milliseconds.
      )pbdoc");

  nb::implicitly_convertible<mx::Device::DeviceType, mx::Device>();

  m.def(
      "default_stream",
      &mx::default_stream,
      "device"_a,
      R"pbdoc(Get the device's default stream.)pbdoc");
  m.def(
      "set_default_stream",
      &mx::set_default_stream,
      "stream"_a,
      R"pbdoc(
        Set the default stream.

        This will make the given stream the default for the
        streams device. It will not change the default device.

        Args:
          stream (stream): Stream to make the default.
      )pbdoc");
  m.def(
      "new_stream",
      &mx::new_stream,
      "device"_a,
      R"pbdoc(
        Make a new stream on the given device.

        The stream can only be used on the thread where it was created on, using
        it in any other thread would result in errors.
      )pbdoc");
  m.def(
      "new_thread_unsafe_stream",
      &mx::new_thread_unsafe_stream,
      "device"_a,
      R"pbdoc(
        Make a new stream that can be used in any thread.

        Unlike :func:`new_stream` which can only work on the thread of creation,
        streams created by this API can be passed to and evaluated anywhere, but
        note that currently all nodes in a graph must be evaluated in sequence
        and it is user's responsibilty to ensure there is no race condition.
      )pbdoc");
  m.def(
      "new_thread_local_stream",
      &mx::new_thread_local_stream,
      "device"_a,
      R"pbdoc(Make a new stream that will be unique per thread.)pbdoc");
  m.def(
      "clear_streams",
      &mx::clear_streams,
      R"pbdoc(Destroy all streams created in current thread.)pbdoc");

  nb::class_<PyStreamContext>(m, "StreamContext", R"pbdoc(
        A context manager for setting the current device and stream.

        See :func:`stream` for usage.

        Args:
            s: The stream or device to set as the default.
  )pbdoc")
      .def(nb::init<mx::StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContext& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContext& scm,
             const std::optional<nb::type_object>& exc_type,
             const std::optional<nb::object>& exc_value,
             const std::optional<nb::object>& traceback) { scm.exit(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());
  m.def(
      "stream",
      [](mx::StreamOrDevice s) { return PyStreamContext(s); },
      "s"_a,
      R"pbdoc(
        Create a context manager to set the default device and stream.

        Args:
            s: The :obj:`Stream` or :obj:`Device` to set as the default.

        Returns:
            A context manager that sets the default device and stream.

        Example:

        .. code-block::python

          import mlx.core as mx

          # Create a context manager for the default device and stream.
          with mx.stream(mx.cpu):
              # Operations here will use mx.cpu by default.
              pass
      )pbdoc");
  m.def(
      "synchronize",
      [](mx::StreamOrDevice s) {
        if (std::holds_alternative<std::monostate>(s)) {
          mx::synchronize();
        } else {
          mx::synchronize(mx::to_stream(s));
        }
      },
      "stream"_a = nb::none(),
      R"pbdoc(
      Synchronize with the given stream.

      Args:
        stream (Stream, optional): Stream to synchronize. If device is
           provided the default stream for that device is used. If ``None``
           then the default stream of the default device is used.
           Default: ``None``.
      )pbdoc");
}
