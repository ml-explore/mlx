.. _custom_metal_kernels:

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
          input_names=["inp"],
          output_names=["out"],
          source=source,
      )
      outputs = kernel(
          inputs=[a],
          template=[("T", mx.float32)],
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes=[a.shape],
          output_dtypes=[a.dtype],
      )
      return outputs[0]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))

.. note::
    We are only required to pass the body of the Metal kernel in ``source``.

The full function signature will be generated using:

* The shapes/dtypes of ``inputs``
    In the above, ``a`` is an ``mx.array`` of type ``mx.float16`` and we pass it with the key ``inp``
    so we will add ``const device float16_t* inp`` to the signature.
    ``inp_shape``, ``inp_strides`` and ``inp_ndim`` are also added for convenience if they are present
    in ``source``.
* The list of ``output_dtypes``
    In the above, ``out`` is an ``mx.array`` of type ``mx.float16``
    so we add ``device float16_t* out``.
* Template parameters passed using ``template``
    In the above, ``template=[("T", mx.float32)]`` adds a template of ``template <typename T>`` to the function
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

Note: ``grid`` and ``threadgroup`` are parameters to the Metal `dispatchThreads <https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/2866532-dispatchthreads>`_ function.
This means we will launch ``mx.prod(grid)`` threads, subdivided into ``threadgroup`` size threadgroups.
For optimal performance, each thread group dimension should be less than or equal to the corresponding grid dimension.

Passing ``verbose=True`` to ``mx.fast.metal_kernel.__call__`` will print the generated code for debugging purposes.

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
          input_names=["inp"],
          output_names=["out"],
          source=source
      )
      outputs = kernel(
          inputs=[a],
          template=[("T", mx.float32)],
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes=[a.shape],
          output_dtypes=[a.dtype],
          ensure_row_contiguous=False,
      )
      return outputs[0]

  a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
  # make non-contiguous
  a = a[::2]
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))

Complex Example
-----------------------------

Let's implement a more complex example: ``grid_sample`` in ``"bilinear"`` mode.

We'll start with the following MLX implementation using standard ops:

.. code-block:: python

    def grid_sample_ref(x, grid):
        N, H_in, W_in, _ = x.shape
        ix = ((grid[..., 0] + 1) * W_in - 1) / 2
        iy = ((grid[..., 1] + 1) * H_in - 1) / 2

        ix_nw = mx.floor(ix).astype(mx.int32)
        iy_nw = mx.floor(iy).astype(mx.int32)

        ix_ne = ix_nw + 1
        iy_ne = iy_nw

        ix_sw = ix_nw
        iy_sw = iy_nw + 1

        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

        nw = (ix_se - ix)    * (iy_se - iy)
        ne = (ix    - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix)    * (iy    - iy_ne)
        se = (ix    - ix_nw) * (iy    - iy_nw)

        I_nw = x[mx.arange(N)[:, None, None], iy_nw, ix_nw, :]
        I_ne = x[mx.arange(N)[:, None, None], iy_ne, ix_ne, :]
        I_sw = x[mx.arange(N)[:, None, None], iy_sw, ix_sw, :]
        I_se = x[mx.arange(N)[:, None, None], iy_se, ix_se, :]

        mask_nw = (iy_nw >= 0) & (iy_nw <= H_in - 1) & (ix_nw >= 0) & (ix_nw <= W_in - 1)
        mask_ne = (iy_ne >= 0) & (iy_ne <= H_in - 1) & (ix_ne >= 0) & (ix_ne <= W_in - 1)
        mask_sw = (iy_sw >= 0) & (iy_sw <= H_in - 1) & (ix_sw >= 0) & (ix_sw <= W_in - 1)
        mask_se = (iy_se >= 0) & (iy_se <= H_in - 1) & (ix_se >= 0) & (ix_se <= W_in - 1)

        I_nw *= mask_nw[..., None]
        I_ne *= mask_ne[..., None]
        I_sw *= mask_sw[..., None]
        I_se *= mask_se[..., None]

        output = nw[..., None] * I_nw + ne[..., None] * I_ne + sw[..., None] * I_sw + se[..., None] * I_se

        return output

Now let's use ``mx.custom_function`` together with ``mx.fast.metal_kernel``
to write a fast GPU kernel for both the forward and backward passes.

First we'll implement the forward pass as a fused kernel:

.. code-block:: python

    @mx.custom_function
    def grid_sample(x, grid):

        assert x.ndim == 4, "`x` must be 4D."
        assert grid.ndim == 4, "`grid` must be 4D."

        B, _, _, C = x.shape
        _, gN, gM, D = grid.shape
        out_shape = (B, gN, gM, C)

        assert D == 2, "Last dim of `grid` must be size 2."

        source = """
            uint elem = thread_position_in_grid.x;
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            uint grid_idx = elem / C * 2;
            float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
            float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;

            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            T nw = (ix_se - ix)    * (iy_se - iy);
            T ne = (ix    - ix_sw) * (iy_sw - iy);
            T sw = (ix_ne - ix)    * (iy    - iy_ne);
            T se = (ix    - ix_nw) * (iy    - iy_nw);

            int batch_idx = elem / C / gH / gW * b_stride;
            int channel_idx = elem % C;
            int base_idx = batch_idx + channel_idx;

            T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
            T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
            T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
            T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

            I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
            I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
            I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
            I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;

            out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
        """
        kernel = mx.fast.metal_kernel(
            name="grid_sample",
            input_names=["x", "grid"],
            output_names=["out"],
            source=source,
        )
        outputs = kernel(
            inputs=[x, grid],
            template=[("T", x.dtype)],
            output_shapes=[out_shape],
            output_dtypes=[x.dtype],
            grid=(np.prod(out_shape), 1, 1),
            threadgroup=(256, 1, 1),
        )
        return outputs[0]

For a reasonably sized input such as:

.. code-block:: python

    x.shape = (8, 1024, 1024, 64)
    grid.shape = (8, 256, 256, 2)

On an M1 Max, we see a big performance improvement:

``55.7ms -> 6.7ms => 8x speed up``

Grid Sample VJP
---------------

Since we decorated ``grid_sample`` with ``mx.custom_function``, we can now define
its custom vjp transform so MLX can differentiate it.

The backwards pass requires atomically updating ``x_grad``/``grid_grad`` and so
requires a few extra ``mx.fast.metal_kernel`` features:

* ``init_value=0``
    Initialize all of the kernel's outputs to this value before it runs. This allows us to update only part of the output arrays with the kernel.

* ``atomic_outputs=True``
    Designate all of the kernel outputs as ``atomic`` in the function signature. 
    This means we can use Metal's ``atomic`` features to simultaneously update the ``x_grad`` and ``grid_grad`` arrays from multiple threadgroups. 
    See section 6.15 of the `Metal Shading Language Specification <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_ for more details.

We can then implement the backwards pass as follows:

.. code-block:: python

    @grid_sample.vjp
    def grid_sample_vjp(primals, cotangent, _):
        x, grid = primals
        B, _, _, C = x.shape
        _, gN, gM, D = grid.shape

        assert D == 2, "Last dim of `grid` must be size 2."

        source = """
            uint elem = thread_position_in_grid.x;
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            // Pad C to the nearest larger simdgroup size multiple
            int C_padded = ceildiv(C, threads_per_simdgroup) * threads_per_simdgroup;

            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            uint grid_idx = elem / C_padded * 2;
            float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
            float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;

            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            T nw = (ix_se - ix)    * (iy_se - iy);
            T ne = (ix    - ix_sw) * (iy_sw - iy);
            T sw = (ix_ne - ix)    * (iy    - iy_ne);
            T se = (ix    - ix_nw) * (iy    - iy_nw);

            int batch_idx = elem / C_padded / gH / gW * b_stride;
            int channel_idx = elem % C_padded;
            int base_idx = batch_idx + channel_idx;

            T gix = T(0);
            T giy = T(0);
            if (channel_idx < C) {
                int cot_index = elem / C_padded * C + channel_idx;
                T cot = cotangent[cot_index];
                if (iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1) {
                    int offset = base_idx + iy_nw * h_stride + ix_nw * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], nw * cot, memory_order_relaxed);

                    T I_nw = x[offset];
                    gix -= I_nw * (iy_se - iy) * cot;
                    giy -= I_nw * (ix_se - ix) * cot;
                }
                if (iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1) {
                    int offset = base_idx + iy_ne * h_stride + ix_ne * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], ne * cot, memory_order_relaxed);

                    T I_ne = x[offset];
                    gix += I_ne * (iy_sw - iy) * cot;
                    giy -= I_ne * (ix - ix_sw) * cot;
                }
                if (iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1) {
                    int offset = base_idx + iy_sw * h_stride + ix_sw * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], sw * cot, memory_order_relaxed);

                    T I_sw = x[offset];
                    gix -= I_sw * (iy - iy_ne) * cot;
                    giy += I_sw * (ix_ne - ix) * cot;
                }
                if (iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1) {
                    int offset = base_idx + iy_se * h_stride + ix_se * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], se * cot, memory_order_relaxed);

                    T I_se = x[offset];
                    gix += I_se * (iy - iy_nw) * cot;
                    giy += I_se * (ix - ix_nw) * cot;
                }
            }

            T gix_mult = W / 2;
            T giy_mult = H / 2;

            // Reduce across each simdgroup first.
            // This is much faster than relying purely on atomics.
            gix = simd_sum(gix);
            giy = simd_sum(giy);

            if (thread_index_in_simdgroup == 0) {
                atomic_fetch_add_explicit(&grid_grad[grid_idx], gix * gix_mult, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy * giy_mult, memory_order_relaxed);
            }
        """
        kernel = mx.fast.metal_kernel(
            name="grid_sample_grad",
            input_names=["x", "grid", "cotangent"],
            output_names=["x_grad", "grid_grad"],
            source=source,
            atomic_outputs=True,
        )
        # pad the output channels to simd group size
        # so that our `simd_sum`s don't overlap.
        simdgroup_size = 32
        C_padded = (C + simdgroup_size - 1) // simdgroup_size * simdgroup_size
        grid_size = B * gN * gM * C_padded
        outputs = kernel(
            inputs=[x, grid, cotangent],
            template=[("T", x.dtype)],
            output_shapes=[x.shape, grid.shape],
            output_dtypes=[x.dtype, x.dtype],
            grid=(grid_size, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )
        return outputs[0], outputs[1]

There's an even larger speed up for the vjp:

``676.4ms -> 16.7ms => 40x speed up``
