Metal Logging
=============

In **Debug** builds, MLX compiles Metal kernels with `os_log` enabled so shader
warnings and debug messages are visible during development.

Build in Debug
--------------

.. code-block:: bash

    DEBUG=1 python -m pip install -e .

Logging from inside a kernel
----------------------------

Inside a Metal kernel, include `metal_logging` and use `os_log`:

.. code-block:: 

    #include <metal_logging>
    using namespace metal;

    constant os_log logger("mlx", "my_kernel");

    kernel void my_kernel(/* ... */) {
    // ...
      logger.log_debug("unexpected state: idx=%u", idx);
    }

Run
---

For console apps, enable Metal shader logging at launch and forward it to stderr:

.. code-block:: bash

    MTL_LOG_LEVEL=MTLLogLevelInfo MTL_LOG_TO_STDERR=1 your_app

Where to see logs
-----------------

Logs show up in:

* **Xcode Run console** (when you launch your app under Xcode)
* **Command line** via `log stream` (for console apps or any run)

See Apple's `Metal logging guide`_ for details.

.. _Metal logging guide: [https://developer.apple.com/documentation/metal/logging-shader-debug-messages](https://developer.apple.com/documentation/metal/logging-shader-debug-messages)
