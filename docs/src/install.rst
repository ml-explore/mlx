.. _build_and_install:

Build and Install
=================

Python Installation
-------------------

MLX is available on PyPI. All you have to do to use MLX with your own Apple
silicon computer is

.. code-block:: shell

    pip install mlx

To install from PyPI your system must meet the following requirements:

- Using an M series chip (Apple silicon)
- Using a native Python >= 3.10
- macOS >= 14.0

.. note::
    MLX is only available on devices running macOS >= 14.0 and higher.

CUDA
^^^^

MLX has a CUDA backend which you can install with:

.. code-block:: shell

    pip install mlx[cuda12]


To install the CUDA package from PyPi your system must meet the following
requirements:

- Nvidia architecture >= SM 7.5
- Nvidia driver >= 550.54.14
- CUDA toolkit >= 12.0
- Linux distribution with glibc >= 2.35
- Python >= 3.10

For CUDA 13 use ``pip install mlx[cuda13]``. The CUDA 13 package requires
an Nvidia driver >= 580 or an appropriate CUDA compatibility package.

CPU-only (Linux)
^^^^^^^^^^^^^^^^

For a CPU-only version of MLX that runs on Linux use:

.. code-block:: shell

    pip install mlx[cpu]

To install the CPU-only package from PyPi your system must meet the following
requirements:

- Linux distribution with glibc >= 2.35
- Python >= 3.10


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
- `cmake <https://cmake.org/>`_ -- version 3.25 or later, and ``make``
- Xcode >= 15.0 and macOS SDK >= 14.0

.. note::
   Ensure your shell environment is native ``arm``, not ``x86`` via Rosetta. If
   the output of ``uname -p`` is ``x86``, see the :ref:`troubleshooting section <build shell>` below.

Python API
^^^^^^^^^^

.. _python install:

To build and install the MLX python library from source, first, clone MLX from
`its GitHub repo <https://github.com/ml-explore/mlx>`_:

.. code-block:: shell

   git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

Then simply build and install MLX using pip:

.. code-block:: shell

  pip install .

For developing, install the package with development dependencies, and use an
editable install:

.. code-block:: shell

  pip install -e ".[dev]"

Once the development dependencies are installed, you can build faster with:

.. code-block:: shell

 python setup.py build_ext --inplace

Run the tests with:

.. code-block:: shell

  python -m unittest discover python/tests

C++ API
^^^^^^^

.. _cpp install:

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
Metal kernel cache persists across reboots.

Linux
^^^^^

To build from source on Linux (CPU only), install the BLAS and LAPACK headers.
For example on Ubuntu, run the following:

.. code-block:: shell

   apt-get update -y
   apt-get install libblas-dev liblapack-dev liblapacke-dev -y

From here follow the instructions to install either the :ref:`Python <python
install>` or :ref:`C++ <cpp install>` APIs.

CUDA
^^^^

To build from source on Linux with CUDA, install the BLAS and LAPACK headers
and the CUDA toolkit. For example on Ubuntu, run the following:

.. code-block:: shell

   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   dpkg -i cuda-keyring_1.1-1_all.deb
   apt-get update -y
   apt-get -y install cuda-toolkit-12-9
   apt-get install libblas-dev liblapack-dev liblapacke-dev libcudnn9-dev-cuda-12 -y


When building either the Python or C++ APIs make sure to pass the cmake flag
``MLX_BUILD_CUDA=ON``. For example, to build the Python API run:

.. code-block:: shell

  CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install -e ".[dev]"

To build the C++ package run:

.. code-block:: shell

   mkdir -p build && cd build
   cmake .. -DMLX_BUILD_CUDA=ON && make -j


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

If the output of ``uname -p``  is ``x86`` then your shell is running as x86 via
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
wipe your build cache with ``rm -rf build/`` and try again.
