.. _kv_cache:

Writing a Fast KV Cache
=======================

Autoregressive generation appends one position to the key and value arrays for
every token it produces. How that append is written dominates the cost of the
cache.

The direct way is :func:`concatenate`:

.. code-block:: python

  # Avoid this
  cache = mx.zeros((1, 0, d))
  for x in steps:
      cache = mx.concatenate([cache, x], axis=1)
      mx.eval(cache)

Prefer preallocating storage in fixed-size chunks and updating it in place:

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

Indexed assignment ``cache[:, offset : offset + 1, :] = x`` is equivalent and
reads more naturally.

Appending one position per step to 20 caches of shape ``[1, 4, N, 512]`` in
``bfloat16``, measured on an M4 Max:

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

The preallocated version is flat in context length. The concatenating version
is not.

Why Concatenating Is Slow
-------------------------

There are two costs, one obvious and one not.

The obvious one is copying. :func:`concatenate` makes a new array and copies
the entire contents on every step, so appending ``n`` positions copies on the
order of ``n^2`` elements.

The less visible one is that growing defeats buffer reuse. MLX pools freed
device buffers, but a pooled buffer is only taken back by a request that
closely matches its size: the pool does not split a buffer to serve a smaller
request, nor combine buffers to serve a larger one. A loop that grows every
step never produces a match, so every step allocates from the driver while the
freed buffers collect at sizes nothing will ask for again. That cost is paid on
the CPU, and in a profile it appears as GPU idle time between kernels rather
than as a slow kernel, which makes it easy to misread as a problem with the
model.

Allocation sizes are rounded up to the page size, so while each step adds less
than a page some steps do reuse a pooled buffer. Once the per-step growth
reaches the page size, every step misses.

Preallocating avoids both costs: the shape stops changing, so there is nothing
to copy and nothing to allocate.

Choosing a Chunk Size
---------------------

Use a multiple of 256.

Besides amortizing the growth step, this matters for attention routing on CUDA.
Sending single-token attention to the fused cuDNN kernel requires the key and
value arrays to be slices of a contiguous cache whose capacity is a multiple of
256, with at least 256 positions in use. A cache grown by any other increment
silently takes a slower path.
