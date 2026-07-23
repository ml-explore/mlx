.. _precision:

Numerical Precision
===================

By default MLX runs ``float32`` matrix-multiplication family operations
(matmul, quantized matmul, grouped matmul, convolution, and attention)
at reduced, TF32-class precision on hardware with dedicated
matrix-multiplication units:

* On the CUDA backend, cuBLAS and cuDNN are asked for TF32 tensor-core
  math.
* On Apple silicon with a neural accelerator (the M5 generation), the
  Metal backend dispatches these operations to accelerator kernels of
  comparable precision. Earlier Apple silicon (M1–M4) runs them in full
  ``float32``.

TF32-class math keeps the ``float32`` exponent range but rounds the
significand to about 10 bits, so results carry roughly three decimal
digits instead of seven. For a 512x512 ``float32`` matrix product the
relative error against a ``float64`` reference is on the order of
``1e-4`` to ``1e-3``, compared with about ``2e-7`` in full ``float32``
(measured on both backends). Individual operations still look correct,
but workloads that compare nearly equal values downstream — for
example an ``argmax`` over correlation scores, or a kernel-vs-reference
test with tight tolerances — can change results.

To run these operations in full ``float32`` precision, set the
environment variable ``MLX_ENABLE_TF32=0``. The variable is read once,
when the first operation that could use reduced precision runs, and the
value is latched for the lifetime of the process. Setting it later has
no effect. Setting it in-process works as long as it happens before the
first such operation:

.. code-block:: python

  import os

  os.environ["MLX_ENABLE_TF32"] = "0"  # before the first matmul

  import mlx.core as mx

  # float32 matmuls now run at full float32 precision

A few details worth knowing:

* Matrix-vector products (output shapes with ``M == 1`` or ``N == 1``)
  retain full ``float32`` accuracy on both backends even under the
  default, so within one program the effect can appear shape-dependent.
* On the CUDA backend, ``complex64`` matrix products use the same
  reduced-precision path and are likewise restored to full precision by
  ``MLX_ENABLE_TF32=0``.
* ``float16`` and ``bfloat16`` inputs are not affected by this setting.
