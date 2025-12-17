Metal Logging
=============

In debug builds, MLX compiles Metal kernels with ``os_log`` enabled so shader
warnings and debug messages are visible during development.

.. note::
    Metal logging is only available with Metal 3.2 or higher (macOS 15 and up,
    iOS 18 and up).

To enable logging from kernels, first make sure to build in debug mode:

.. code-block:: bash

    DEBUG=1 python -m pip install -e .

Then, in the kernel source code include MLX's logging shim and use
``mlx::os_log``:

.. code-block::

    #include "mlx/backend/metal/kernels/logging.h"

    constant mlx::os_log logger("mlx", "my_kernel");

    kernel void my_kernel(/* ... */) {
    // ...
      logger.log_debug("unexpected state: idx=%u", idx);
    }

When you run the program, set the Metal log level to your desired level and
forward logs to ``stderr``:

.. code-block:: bash

    MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_TO_STDERR=1 python script.py

See the `Metal logging guide`_ for more details.

.. _`Metal logging guide`: https://developer.apple.com/documentation/metal/logging-shader-debug-messages
