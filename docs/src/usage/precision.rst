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

On Metal, ``conv2d`` can use the Winograd algorithm for 3x3 convolutions
with stride 1. Winograd is faster for large inputs, but its rounding error
in ``float32`` can be approximately 10 times larger. MLX selects Winograd
from the total input size, which includes the batch size. Thus, the same
image can give different results in a batch of 3 and in a batch of 1.

To disable Winograd, set :envvar:`MLX_CONV_WINOGRAD` to ``0``:

.. code-block:: shell

  MLX_CONV_WINOGRAD=0 python my_script.py

8-bit Quantized Matmul
----------------------

Quantized matmuls dequantize the weights to the type of the input and
multiply in floating point. On hardware with dedicated
matrix-multiplication units MLX can instead round the weights and the input
to 8 bits and multiply those, which is faster and less precise. This is off
by default.

For affine quantized weights set :envvar:`MLX_QMM_INT8`. The weights and the
input are rounded to 8-bit integers with one scale per row. With ``1`` the
scale of the input is refined for every block of 64 elements, with ``4`` for
every four blocks, and ``2`` uses the row scale alone. Other values keep the
default path:

.. code-block:: shell

  MLX_QMM_INT8=1 python my_script.py

Expect a relative difference of one to a few percent from the floating point
result. Only products of 64 rows or more with transposed weights take this
path. Like the other variables it is read once, when the first quantized
matmul with transposed affine weights is built; to change it later use :func:`mlx.core.config.update` or
:func:`mlx.core.config.scoped_update`, which apply to the quantized matmuls
built afterwards.
