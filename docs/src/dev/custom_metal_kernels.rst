Custom Metal Kernels
====================

MLX supports writing custom Metal kernels through the Python and C++ APIs.

Simple Example
--------------

Let's write a custom kernel that computes ``exp`` elementwise:

.. code-block:: python

  def exp_elementwise(a: mx.array):
      source = """
          uint elem = thread_position_in_grid.x;
          T tmp = inp[elem];
          out[elem] = metal::exp(tmp);
      """

      kernel = mx.fast.metal_kernel(
          name="myexp",
          source=source,
      )
      outputs = kernel(
          inputs={"inp": a},
          template={"T": mx.float32},
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes={"out": a.shape},
          output_dtypes={"out": a.dtype},
      )
      return outputs["out"]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))

.. note::
    We are only required to pass the body of the Metal kernel in ``source``.

The full function signature will be generated using:

* The keys and shapes/dtypes of ``inputs``
    In the above, ``a`` is an ``mx.array`` of type ``mx.float16`` and we pass it with the key ``inp``
    so we will add ``const device float16_t* inp`` to the signature.
    ``inp_shape``, ``inp_strides`` and ``inp_ndim`` are also added for convenience.
* The keys and values of ``output_shapes`` and ``output_dtypes``
    In the above, ``out`` is an ``mx.array`` of type ``mx.float16``
    so we add ``device float16_t* out``.
* Template parameters passed using ``template``
    In the above, ``template={"T": mx.float32}`` adds a template of ``template <typename T>`` to the function
    and instantiates the template with ``custom_kernel_myexp_float<float>``.
    Template parameters can be ``mx.core.Dtype``, ``int`` or ``bool``.
* Metal attributes used in ``source`` such as ``[[thread_position_in_grid]]``
    These will be added as function arguments.
    All the attributes defined in Table 5.8 of the `Metal Shading Language Specification <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_ are supported.

Putting this all together, the generated function signature for ``myexp`` is as follows:

.. code-block:: cpp

  template <typename T>
  [[kernel]] void custom_kernel_myexp_float(
    const device float16_t* inp [[buffer(0)]],
    device float16_t* out [[buffer(1)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {

          uint elem = thread_position_in_grid.x;
          T tmp = inp[elem];
          out[elem] = metal::exp(tmp);

  }

  template [[host_name("custom_kernel_myexp_float")]] [[kernel]] decltype(custom_kernel_myexp_float<float>) custom_kernel_myexp_float<float>;

You can print the generated code for a ``mx.fast.metal_kernel`` by passing ``verbose=True`` when you call it.

Using Shape/Strides
-------------------

``mx.fast.metal_kernel`` supports an argument ``ensure_row_contiguous`` which is ``True`` by default.
This will copy the ``mx.array`` inputs if needed before the kernel is launched to ensure that the memory layout is row contiguous.
Generally this makes writing the kernel easier, since we don't have to worry about gaps or the ordering of the dims
when indexing.

If we want to avoid this copy, ``metal_kernel`` automatically passes ``a_shape``, ``a_strides`` and ``a_ndim`` for each
input array ``a`` if any are present in ``source``.
We can then use MLX's built in indexing utils to fetch the right elements for each thread.

Let's convert ``myexp`` above to support arbitrarily strided arrays without relying on a copy from ``ensure_row_contiguous``:

.. code-block:: python

  def exp_elementwise(a: mx.array):
      source = """
          uint elem = thread_position_in_grid.x;
          // Utils from `mlx/backend/metal/kernels/utils.h` are automatically included
          uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
          T tmp = inp[loc];
          // Output arrays are always row contiguous
          out[elem] = metal::exp(tmp);
      """

      kernel = mx.fast.metal_kernel(
          name="myexp_strided",
          source=source
      )
      outputs = kernel(
          inputs={"inp": a},
          template={"T": mx.float32},
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes={"out": a.shape},
          output_dtypes={"out": a.dtype},
          ensure_row_contiguous=False,
      )
      return outputs["out"]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  # make non-contiguous
  a = a[::2]
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))
