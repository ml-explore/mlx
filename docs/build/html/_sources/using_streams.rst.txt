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
