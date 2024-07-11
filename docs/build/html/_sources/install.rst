Build and Install
=================

Python Installation
-------------------

MLX is available on PyPI. All you have to do to use MLX with your own Apple
silicon computer is

.. code-block:: shell

    pip install mlx

To install from PyPI you must meet the following requirements:

- Using an M series chip (Apple silicon)
- Using a native Python >= 3.8
- macOS >= 13.5

.. note::
    MLX is only available on devices running macOS >= 13.5
    It is highly recommended to use macOS 14 (Sonoma)


MLX is also available on conda-forge. To install MLX with conda do:

.. code-block:: shell

   conda install conda-forge::mlx


Troubleshooting
^^^^^^^^^^^^^^^

*My OS and Python versions are in the required range but pip still does not find
a matching distribution.*

Probably you are using a non-native Python. The output of

.. code-block:: shell

  python -c "import platform; print(platform.processor())"

should be ``arm``. If it is ``i386`` (and you have M series machine) then you
are using a non-native Python. Switch your Python to a native Python. A good
way to do this is with `Conda <https://stackoverflow.com/q/65415996>`_.


Build from source
-----------------

Build Requirements
^^^^^^^^^^^^^^^^^^

- A C++ compiler with C++17 support (e.g. Clang >= 5.0)
- `cmake <https://cmake.org/>`_ -- version 3.24 or later, and ``make``
- Xcode >= 15.0 and macOS SDK >= 14.0

.. note::
   Ensure your shell environment is native ``arm``, not ``x86`` via Rosetta. If
   the output of ``uname -p`` is ``x86``, see the :ref:`troubleshooting section <build shell>` below.

Python API
^^^^^^^^^^

To build and install the MLX python library from source, first, clone MLX from
`its GitHub repo <https://github.com/ml-explore/mlx>`_:

.. code-block:: shell

   git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

Install `nanobind <https://nanobind.readthedocs.io/en/latest/>`_ with:

.. code-block:: shell

    pip install git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4

Then simply build and install MLX using pip:

.. code-block:: shell

   env CMAKE_BUILD_PARALLEL_LEVEL="" pip install .

For developing use an editable install:

.. code-block:: shell

  env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e .

To make sure the install is working run the tests with:

.. code-block:: shell

  pip install ".[testing]"
  python -m unittest discover python/tests

Optional: Install stubs to enable auto completions and type checking from your IDE:

.. code-block:: shell

  pip install ".[dev]"
  python setup.py generate_stubs

C++ API
^^^^^^^

Currently, MLX must be built and installed from source.

Similarly to the python library, to build and install the MLX C++ library start
by cloning MLX from `its GitHub repo
<https://github.com/ml-explore/mlx>`_:

.. code-block:: shell

   git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

Create a build directory and run CMake and make:

.. code-block:: shell

   mkdir -p build && cd build
   cmake .. && make -j

Run tests with:

.. code-block:: shell

   make test

Install with:

.. code-block:: shell

   make install

Note that the built ``mlx.metallib`` file should be either at the same
directory as the executable statically linked to ``libmlx.a`` or the
preprocessor constant ``METAL_PATH`` should be defined at build time and it
should point to the path to the built metal library.

.. list-table:: Build Options
   :widths: 25 8
   :header-rows: 1

   * - Option
     - Default
   * - MLX_BUILD_TESTS
     - ON
   * - MLX_BUILD_EXAMPLES
     - OFF
   * - MLX_BUILD_BENCHMARKS
     - OFF
   * - MLX_BUILD_METAL
     - ON
   * - MLX_BUILD_CPU
     - ON
   * - MLX_BUILD_PYTHON_BINDINGS
     - OFF
   * - MLX_METAL_DEBUG
     - OFF
   * - MLX_BUILD_SAFETENSORS
     - ON
   * - MLX_BUILD_GGUF
     - ON
   * - MLX_METAL_JIT
     - OFF

.. note::

    If you have multiple Xcode installations and wish to use
    a specific one while building, you can do so by adding the
    following environment variable before building

    .. code-block:: shell

      export DEVELOPER_DIR="/path/to/Xcode.app/Contents/Developer/"

    Further, you can use the following command to find out which
    macOS SDK will be used

    .. code-block:: shell

      xcrun -sdk macosx --show-sdk-version

Binary Size Minimization
~~~~~~~~~~~~~~~~~~~~~~~~

To produce a smaller binary use the CMake flags ``CMAKE_BUILD_TYPE=MinSizeRel``
and ``BUILD_SHARED_LIBS=ON``.

The MLX CMake build has several additional options to make smaller binaries.
For example, if you don't need the CPU backend or support for safetensors and
GGUF, you can do:

.. code-block:: shell

  cmake .. \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DBUILD_SHARED_LIBS=ON \
    -DMLX_BUILD_CPU=OFF \
    -DMLX_BUILD_SAFETENSORS=OFF \
    -DMLX_BUILD_GGUF=OFF \
    -DMLX_METAL_JIT=ON

THE ``MLX_METAL_JIT`` flag minimizes the size of the MLX Metal library which
contains pre-built GPU kernels. This substantially reduces the size of the
Metal library by run-time compiling kernels the first time they are used in MLX
on a given machine. Note run-time compilation incurs a cold-start cost which can
be anwywhere from a few hundred millisecond to a few seconds depending on the
application. Once a kernel is compiled, it will be cached by the system. The
Metal kernel cache persists accross reboots.

Troubleshooting
^^^^^^^^^^^^^^^

Metal not found
~~~~~~~~~~~~~~~

You see the following error when you try to build:

.. code-block:: shell

  error: unable to find utility "metal", not a developer tool or in PATH

To fix this, first make sure you have Xcode installed:

.. code-block:: shell

  xcode-select --install

Then set the active developer directory:

.. code-block:: shell

  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

x86 Shell
~~~~~~~~~

.. _build shell:

If the ouptut of ``uname -p``  is ``x86`` then your shell is running as x86 via
Rosetta instead of natively.

To fix this, find the application in Finder (``/Applications`` for iTerm,
``/Applications/Utilities`` for Terminal), right-click, and click “Get Info”.
Uncheck “Open using Rosetta”, close the “Get Info” window, and restart your
terminal.

Verify the terminal is now running natively the following command:

.. code-block:: shell

  $ uname -p
  arm

Also check that cmake is using the correct architecture:

.. code-block:: shell

  $ cmake --system-information | grep CMAKE_HOST_SYSTEM_PROCESSOR
  CMAKE_HOST_SYSTEM_PROCESSOR "arm64"

If you see ``"x86_64"``, try re-installing ``cmake``. If you see ``"arm64"``
but the build errors out with "Building for x86_64 on macOS is not supported."
wipe your build cahce with ``rm -rf build/`` and try again.
