.. _kv_cache:

Writing a Fast KV Cache
=======================

Autoregressive generation appends one position to the key and value arrays for
each generated token. The append strategy can dominate KV-cache performance.

Avoid appending naively with :func:`concatenate`:

.. code-block:: python

  # Avoid this
  cache = mx.zeros((1, 0, d))
  for x in steps:
      cache = mx.concatenate([cache, x], axis=1)
      mx.eval(cache)

Instead, preallocate fixed-size chunks and update the cache in place:

.. code-block:: python

  chunk = 256
  cache = mx.zeros((1, chunk, d))
  offset = 0
  for x in steps:
      if offset == cache.shape[1]:
          cache = mx.concatenate([cache, mx.zeros((1, chunk, d))], axis=1)
      cache = mx.slice_update(cache, x, mx.array(offset), (1,))
      offset += 1
      mx.eval(cache)

  keys = cache[:, :offset]

You can also use indexed assignment, which may read more naturally:
``cache[:, offset : offset + 1, :] = x``.

The following measurements append one position per step to 20 ``bfloat16``
caches of shape ``[1, 4, N, 512]`` on an M4 Max:

.. list-table::
   :widths: 20 30 30
   :header-rows: 1

   * - Context
     - Concatenate
     - Preallocate + update
   * - 512
     - 0.90 ms / step
     - 0.24 ms / step
   * - 1024
     - 1.11 ms / step
     - 0.21 ms / step
   * - 4096
     - 3.73 ms / step
     - 0.22 ms / step

With preallocation, step time remains nearly constant as context length grows.
Concatenation becomes progressively slower.

Why Concatenating Is Slow
-------------------------

Concatenation has two costs: copying data and preventing buffer reuse.

:func:`concatenate` creates a new array and copies the existing cache at every
step. Appending ``n`` positions therefore copies on the order of ``n^2``
elements.

Growing the cache also prevents buffer reuse. MLX pools freed device buffers,
but reuses a buffer only for a similarly sized request. It does not split a
larger buffer for a smaller request or combine smaller buffers for a larger
one. Because a growing cache has a new size at every step, each allocation
typically comes from the driver while freed buffers remain unused.

This allocation work occurs on the CPU. In profiles, it appears as GPU idle
time between kernels rather than as a slow kernel, which can make the model
appear to be the bottleneck.

Preallocation avoids both costs. The cache shape remains fixed, eliminating
both repeated copies and per-step allocations.

Choosing a Chunk Size
---------------------

Use a chunk size that is a multiple of 256.

This both amortizes growth and enables the fused cuDNN attention kernel on
CUDA. For single-token attention, the key and value arrays must be slices of a
contiguous cache with a capacity that is a multiple of 256, and at least 256
positions must be in use. Other chunk sizes silently use a slower path.
