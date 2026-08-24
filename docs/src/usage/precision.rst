.. _precision:

Numerical Precision
===================

By default, MLX may run ``float32`` matrix-multiplication family
operations (matmul, quantized matmul, grouped matmul, convolution and
attention) at reduced precision on hardware with dedicated
matrix-multiplication units. Inputs and outputs stay ``float32``, but
results can differ from a full-precision reference by several orders of
magnitude more than ``float32`` rounding alone would explain.

To keep these operations in full ``float32``, set
:envvar:`MLX_ENABLE_TF32` to ``0`` when launching the process:

.. code-block:: shell

  MLX_ENABLE_TF32=0 python my_script.py

Which operations take the reduced-precision path, and how large the
difference is, depends on the backend and the hardware.
