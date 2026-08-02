.. _using_streams:

Using Streams and Events
========================

.. currentmodule:: mlx.core

Specifying a :obj:`Stream`
--------------------------

All operations (including random number generation) take an optional
keyword argument ``stream``. The argument specifies the :obj:`Stream` on which
the operation runs. If it is unspecified, the operation uses the default stream
of the default device, ``mx.default_stream(mx.default_device())``. A
:obj:`Device` can also be used as the ``stream`` argument, in which case the
operation uses that device's default stream.

Streams are in-order queues. Evaluated operations on the same stream run in
order, while operations on different streams may overlap. Create a stream
directly with :class:`Stream`, or use a stream as a context manager to make it
the default for operations created in the context:

.. code-block:: python

   compute_stream = mx.Stream(mx.gpu)

   with compute_stream:
       y = mx.exp(x)
       mx.async_eval(y)

Using ``with mx.stream(compute_stream):`` is equivalent to using
``with compute_stream:``.

.. note::

   An event records a point in a stream's GPU execution timeline; it does not
   evaluate lazy arrays. :meth:`Stream.record_event` submits this point even if
   no computation precedes it. Call :func:`async_eval` on the arrays whose work
   should precede the event. Similarly, :meth:`Stream.synchronize` only waits
   for work that has already been scheduled on the stream.

Synchronizing streams
---------------------

Use an :class:`Event` to order work across streams. Recording an event places
it after work already scheduled on the recording stream. Waiting for the event
places it before work subsequently scheduled on the waiting stream:

.. code-block:: python

   producer = mx.Stream(mx.gpu)
   consumer = mx.Stream(mx.gpu)

   produced = mx.exp(x, stream=producer)
   mx.async_eval(produced)

   ready = producer.record_event()
   consumer.wait_event(ready)

   consumed = mx.exp(x, stream=consumer)
   mx.async_eval(consumed)

On a Metal stream, :meth:`Stream.record_event` commits the current command
buffer so the event can progress, but it does not wait for completion.
:meth:`Stream.wait_event` adds a GPU-side dependency and does not block the
CPU. The dependency applies only to work scheduled on the waiting stream after
the call.

:meth:`Stream.wait_stream` is shorthand for recording an event on another
stream and waiting for it:

.. code-block:: python

   consumer.wait_stream(producer)

Only work already scheduled on ``producer`` is included in this dependency.
Work scheduled on ``producer`` after :meth:`Stream.wait_stream` is not included.

Use :meth:`Stream.query` to check completion without blocking. It does not
evaluate arrays or submit pending work. Use :meth:`Stream.synchronize` to
submit pending stream work and block the CPU until it completes:

.. code-block:: python

   if not consumer.query():
       consumer.synchronize()

Timing GPU work
---------------

Use timing-enabled events to measure work on a Metal GPU stream:

.. code-block:: python

   timing_stream = mx.Stream(mx.gpu)
   start = mx.Event(mx.gpu, enable_timing=True)
   end = mx.Event(mx.gpu, enable_timing=True)

   start.record(timing_stream)

   y = mx.exp(x, stream=timing_stream)
   mx.async_eval(y)

   end.record(timing_stream)
   milliseconds = start.elapsed_time(end)

Recording an event submits the recording stream but does not wait for it.
:meth:`Event.elapsed_time` waits for both events before returning. Metal timing
events use GPU command-buffer completion timestamps, so the result covers the
GPU interval between the two record points rather than the execution time of a
single kernel.
