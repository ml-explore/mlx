.. _timing_gpu_work:

Timing GPU Work
===============

.. currentmodule:: mlx.core

Use :class:`Timer` to measure the GPU execution time between two points in a
lazy graph:

.. code-block:: python

   import mlx.core as mx

   x = mx.random.uniform(shape=(1024, 1024))
   y = mx.random.uniform(shape=(1024, 1024))
   mx.eval(x, y)

   timer = mx.Timer()
   x, y = timer.start(x, y)
   z = x @ y
   z = timer.stop(z)

   mx.async_eval(z)
   print(timer.elapsed_time())  # Synchronizes and returns milliseconds.

The timer reports the elapsed GPU time between the start and stop markers on
its stream. It does not report the lifetime of an individual argument or the
sum of individual kernel durations. GPU work ordered between the markers is
included, as are idle gaps if the work is scheduled in separate evaluations.

Each marker is a single pass-through graph operation, even when it receives
multiple arguments. The start marker runs only after all of its arguments are
ready, and the stop marker runs only after all of its arguments are produced.
Thus, independently produced branches which consume the start outputs and feed
the stop inputs are all inside the timed interval, regardless of their argument
order or relative production times.

Operations which produce the arguments to :meth:`Timer.start` are outside the
timed interval. Only graph operations ordered after the start marker and needed
to produce the arguments to :meth:`Timer.stop` are guaranteed to be inside it.
A timer measures one interval and only supports GPU streams.

The timing markers and measured operations must use the same stream. This is
automatic for the default GPU stream. When passing an explicit stream to
:class:`Timer`, create the measured operations inside the corresponding
:func:`stream` context.

Calling :meth:`Timer.elapsed_time` schedules the stop marker if necessary and
blocks the CPU until it completes. Using :func:`async_eval` makes that the only
synchronization point. If :func:`eval` is used instead, it already waits for the
timed graph to finish before :meth:`Timer.elapsed_time` is called.

To measure a compiled function, place the markers outside the compiled
function:

.. code-block:: python

   compiled_matmul = mx.compile(lambda a, b: a @ b)
   timer = mx.Timer()
   x, y = timer.start(x, y)
   z = compiled_matmul(x, y)
   z = timer.stop(z)
   elapsed = timer.elapsed_time()
