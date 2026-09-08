.. _environment_variables:

Environment Variables
=====================

MLX uses environment variables to configure compilation, numerical precision,
backend behavior, and distributed execution. Set them before starting the
process. Many variables are read when the corresponding subsystem is first
initialized and changing them later may have no effect.

Boolean variables use ``0`` to disable and a nonzero integer to enable unless
otherwise noted.

General
-------

.. envvar:: MLX_DISABLE_COMPILE

   Disable compilation globally. This variable is enabled by its presence, so
   setting it to ``0`` also disables compilation. Calling
   :func:`mlx.core.enable_compile` overrides it.

.. envvar:: MLX_ENABLE_TF32

   Allow reduced-precision ``float32`` matrix-multiplication family operations
   on supported hardware. The default is ``1``. Set it to ``0`` to keep these
   operations in full ``float32`` precision. See :doc:`precision`.

Distributed
-----------

The variables required to initialize a distributed process depend on the
backend. :doc:`distributed` describes their formats and how ``mlx.launch``
sets them.

.. envvar:: MLX_RANK

   The zero-based rank of the current process. This is used by the Ring,
   JACCL, and NCCL backends. ``JACCL_RANK`` is accepted as a higher-priority
   alias by JACCL.

.. envvar:: MLX_HOSTFILE

   The path to the JSON host file used by the Ring backend.

.. envvar:: MLX_RING_VERBOSE

   Enable verbose logging for the Ring backend. This variable is enabled by
   its presence.

.. envvar:: MLX_IBV_DEVICES

   The path to the JSON device-connectivity file used by JACCL.
   ``JACCL_IBV_DEVICES`` is accepted as a higher-priority alias.

.. envvar:: MLX_JACCL_COORDINATOR

   The coordinator address in ``IP:port`` form used to establish JACCL
   connections. ``JACCL_COORDINATOR`` is accepted as a higher-priority alias.

.. envvar:: MLX_JACCL_RING

   Prefer a ring topology for JACCL. This variable is enabled by its presence.
   ``JACCL_RING`` is accepted as a higher-priority alias.

.. envvar:: MLX_WORLD_SIZE

   The total number of processes in an NCCL group.

.. envvar:: MLX_NCCL_TIMEOUT

   The timeout in milliseconds for establishing NCCL bootstrap connections.
   The default is ``300000``.

.. envvar:: MLX_MPI_LIBNAME

   Override the MPI dynamic-library name. The default is ``libmpi.dylib`` on
   macOS and ``libmpi.so`` on other platforms.

The NCCL backend also requires ``NCCL_HOST_IP`` and ``NCCL_PORT``. Setting
``NCCL_DEBUG=INFO`` enables additional logging while MLX establishes the
bootstrap connection. NCCL itself recognizes additional `NCCL environment
variables <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>`_.
``CUDA_VISIBLE_DEVICES`` selects the local CUDA device for each process and is
handled by the CUDA runtime.

Metal
-----

.. envvar:: MLX_METAL_FAST_SYNCH

   Enable the faster Metal CPU/GPU synchronization path. The default is ``0``.
   This requires Metal 3.2 or later (macOS 15 or later, or iOS 18 or later).

Advanced tuning
---------------

These variables tune MLX implementation details. They are primarily intended
for development, diagnostics, and performance experiments. The defaults are
selected automatically for the current hardware and are appropriate for most
users. Their behavior may change as the implementation evolves.

.. envvar:: MLX_BFS_MAX_WIDTH

   Set the breadth-first-search width limit used when constructing an
   evaluation tape. The default is ``20``.

.. envvar:: MLX_MAX_OPS_PER_BUFFER

   Override the maximum number of operations encoded in one Metal command
   buffer or CUDA graph. The default depends on the device.

.. envvar:: MLX_MAX_MB_PER_BUFFER

   Override the approximate memory limit, in megabytes, for one Metal command
   buffer or CUDA graph. The default depends on the device.

.. envvar:: MLX_METAL_GPU_ARCH

   Override the Metal GPU architecture string reported to MLX. This affects
   architecture-specific kernel and scheduling choices, but does not change
   the capabilities of the physical GPU. Forcing an architecture that does not
   match the GPU can select incompatible kernels and produce incorrect results.

.. envvar:: MLX_SDPA_BLOCKS

   Override the number of reduction blocks used by the Metal scaled
   dot-product attention kernel. Positive values are rounded up to a multiple
   of ``32``.

CUDA
----

The MLX-prefixed variables in this section are advanced CUDA backend controls.

.. envvar:: MLX_USE_CUDA_GRAPHS

   Enable CUDA graph capture and replay. The default is ``1``.

.. envvar:: MLX_SAVE_CUDA_GRAPHS_DOT_FILE

   Use the specified value as the filename prefix when writing captured CUDA
   graphs to numbered DOT files. An unset or empty value disables the output.

.. envvar:: MLX_PTX_CACHE_DIR

   Override the directory used to cache runtime-compiled PTX. By default MLX
   uses an ``mlx/<version>/ptx`` directory under the system temporary
   directory.

.. envvar:: MLX_CUDA_USE_CUDNN_SDPA

   Allow the CUDA backend to use cuDNN scaled dot-product attention when the
   inputs and device are supported. The default is ``1``.

.. envvar:: MLX_CUDA_CONV_CACHE_SIZE

   Set the CUDA convolution cache capacity. The default is ``128``.

.. envvar:: MLX_CUDA_FFT_CACHE_SIZE

   Set the CUDA FFT plan cache capacity. The default is ``128``.

.. envvar:: MLX_CUDA_GRAPH_CACHE_SIZE

   Set the CUDA graph cache capacity. The default is ``400``.

.. envvar:: MLX_CUDA_SDPA_CACHE_SIZE

   Set the CUDA forward scaled dot-product attention cache capacity. The
   default is ``256``.

.. envvar:: MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE

   Set the CUDA backward scaled dot-product attention cache capacity. The
   default is ``64``.

.. envvar:: MLX_ENABLE_CACHE_THRASHING_CHECK

   Detect repeated CUDA cache misses and raise an error suggesting a larger
   cache capacity. The default is ``1``.

MLX also uses ``CUDA_HOME`` or ``CUDA_PATH`` to locate CUDA headers for runtime
kernel compilation when they cannot be found in the Python environment.
