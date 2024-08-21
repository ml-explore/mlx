.. _custom_metal_kernels:

Custom Metal Kernels
====================

MLX supports writing custom Metal kernels through the Python and C++ APIs.

Simple Example
--------------

Let's write a custom kernel that computes ``exp`` elementwise:

.. code_block:: python

  def exp_elementwise(a: mx.array):
      source = """
          uint elem = thread_position_in_grid.x;
          T tmp = inp[elem];
          out[elem] = metal::exp(tmp);
      """

      kernel = mx.fast.MetalKernel(
          name="myexp",
          source=source,
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes={"out": a.shape},
          output_dtypes={"out": a.dtype},
      )
      kernel.template(T=mx.float32)
      outputs = kernel(inp=a)
      return outputs["out"]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))

We are only required to pass the body of the Metal kernel in ``source``.
``mx.fast.MetalKernel`` will generate the full function signature using:

* The keys and shapes/dtypes of the arrays passed to ``MetalKernel.__call__()``.
    In the above, ``a`` is an ``mx.float16`` ``mx.array`` and we pass it with the key ``inp``
    so we will add ``const device float16_t* inp [[buffer(0)]]`` to the signature.
    It will also add `inp_shape`, `inp_strides` and `inp_ndim` for convenience.
* The keys and values of ``output_shapes`` and ``output_dtypes``.
    In the above, ``out`` is an ``mx.float16`` ``mx.array``
    so we add ``device float16_t* out [[buffer(3)]]``.
* Template parameters passed using ``MetalKernel.template()``.
    In the above, ``kernel.template(T=mx.float32)`` adds a template of ``template <typename T>`` to the function
    and instantiates the template with ``custom_kernel_myexp<float>``.
    Template parameters can be ``mx.core.Dtype``, ``int`` or ``bool``.
* Metal attributes used in ``source`` such as ``[[thread_position_in_grid]]`` or ``[[simdgroup_index_in_threadgroup]]``.
    These will be added as function arguments.
    All the attributes defined in Table 5.8 of the Metal Shading Language Specification are supported.

Putting this all together, the generated function signature for ``myexp`` is as follows:

.. code_block:: cpp

  template <typename T>
  [[kernel]] void myexp_7406405795239204910(
    const device float16_t* inp [[buffer(0)]],
    const constant int* inp_shape [[buffer(1)]],
    const constant size_t* inp_strides [[buffer(2)]],
    const constant int& inp_ndim [[buffer(3)]],
    device float16_t* out [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {

          uint elem = thread_position_in_grid.x;
          T tmp = inp[elem];
          out[elem] = metal::exp(tmp);

  }

  template [[host_name("myexp_7406405795239204910")]] [[kernel]] decltype(myexp_7406405795239204910<float>) myexp_7406405795239204910<float>;

You can print the generated code for a ``mx.fast.MetalKernel`` by passing ``verbose=True`` when you create it.

A hash of the source code and template arguments is added to function name above to prevent cache invalidation.

Using Shape/Strides
-------------------

``mx.fast.MetalKernel`` supports an initialization argument ``ensure_row_contiguous`` which is ``True`` by default.
This will copy the ``mx.array`` inputs if needed before the kernel is launched to ensure that the memory layout is row contiguous.
Generally this makes writing the kernel easier, since we don't have to worry about gaps or the ordering of the dims
when indexing.

If we want to avoid this copy, ``MetalKernel`` automatically passes ``a_shape``, ``a_strides`` and ``a_ndim`` for each
input array ``a`` (as shown in the generated function signature above).
With these we can use MLX's built in indexing utils to fetch the right elements for each thread.

Let's convert ``myexp`` above to support arbitrarily strided arrays without relying on a copy from ``ensure_row_contiguous``:

.. code_block:: python

  def exp_elementwise(a: mx.array):
      source = """
          uint elem = thread_position_in_grid.x;
          // Utils from `mlx/backend/metal/kernels/utils.h` are automatically included
          uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
          T tmp = inp[loc];
          // Output arrays are always row contiguous
          out[elem] = metal::exp(tmp);
      """

      kernel = mx.fast.MetalKernel(
          name="myexp_strided",
          source=source,
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes={"out": a.shape},
          output_dtypes={"out": a.dtype},
          ensure_row_contiguous=False,
      )
      kernel.template(T=mx.float32)
      outputs = kernel(inp=a)
      return outputs["out"]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  # Make a non-contiguous
  a = a[::2]
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))
