.. _using_streams:

Using Streams
=============

.. currentmodule:: mlx.core

Specifying the :obj:`Stream`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All operations (including random number generation) take an optional
keyword argument ``stream``. The ``stream`` kwarg specifies which
:obj:`Stream` the operation should run on. If the stream is unspecified then
the operation is run on the default stream of the default device:
``mx.default_stream(mx.default_device())``.  The ``stream`` kwarg can also
be a :obj:`Device` (e.g. ``stream=my_device``) in which case the operation is
run on the default stream of the provided device
``mx.default_stream(my_device)``.

Function transformations such as :func:`grad`, :func:`vjp`, :func:`jvp` and
:func:`vmap` take no ``stream`` argument. An input that is already evaluated is
traced as a view of its data, so it adds no operation. An operation that a
transformation adds on its own, for example moving the mapped axis of a
:func:`vmap` output, runs on the stream of the operation or array it derives
from, and otherwise on the default stream of the default device.
