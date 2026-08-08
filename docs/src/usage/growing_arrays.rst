.. _growing_arrays:

Growing Arrays
==============

Some workloads append to an array one step at a time. The canonical example is
a key-value cache in autoregressive generation, which gains one position per
generated token.

The direct way to write this is with :func:`concatenate`:

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

Indexed assignment works as well and reads more naturally:

.. code-block:: python

  cache[:, offset : offset + 1, :] = x

The same pattern in C++:

.. code-block:: C++

  int chunk = 256;
  auto cache = mx::zeros({1, chunk, d}, mx::bfloat16);
  int offset = 0;

  for (auto& x : steps) {
    if (offset == cache.shape(1)) {
      auto pad = mx::zeros({1, chunk, d}, cache.dtype());
      cache = mx::concatenate({cache, pad}, 1);
    }
    cache = mx::slice_update(cache, x, {0, offset, 0}, {1, offset + 1, d});
    offset++;
    mx::eval(cache);
  }

  auto keys = mx::slice(cache, {0, 0, 0}, {1, offset, d});

Why It Matters
--------------

Growing by concatenation has two costs, one obvious and one not.

The obvious one is copying. :func:`concatenate` makes a new array and copies
the entire contents on every step, so appending ``n`` positions copies on the
order of ``n^2`` elements.

The less visible one is that the pattern defeats buffer reuse.

MLX gives every array its own device buffer. When an array is freed its buffer
goes into a pool, and a later allocation can take it back, but only whole. The
pool does not split a buffer to serve a smaller request, and it does not
combine buffers to serve a larger one. A pooled buffer is reused only by a
request that closely matches its size.

A loop that grows each step never produces such a match: every request is a
little larger than what was just freed. So every step allocates from the
driver, and the freed buffers collect in the pool at sizes nothing will ask
for again.

That second cost is paid on the CPU, in the driver. In a profile it appears as
GPU idle time between kernels rather than as a slow kernel, which makes it easy
to misread as a problem with the model.

Allocation sizes are rounded up to the page size, so if each step adds less
than a page the requested size only changes every few steps and some steps do
reuse a pooled buffer. Once the per-step growth reaches the page size, every
step misses.

Preallocating avoids both costs: the shape stops changing, so there is nothing
to copy and nothing to allocate. The chunked version below is also flat in
context length, while the concatenating version is not.

Appending one position per step to 20 buffers of shape ``[1, 4, N, 512]`` in
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

Choosing a Chunk Size
---------------------

Use a multiple of 256.

Besides amortizing the growth step, this matters for attention routing on CUDA.
Sending single-token attention to the fused cuDNN kernel requires the key and
value arrays to be slices of a contiguous cache whose capacity is a multiple of
256, with at least 256 positions in use. A cache grown by any other increment
silently takes a slower path.
