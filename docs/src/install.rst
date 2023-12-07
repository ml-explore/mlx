Build and Install
=================

Install from PyPI
-----------------

MLX is available on PyPI. All you have to do to use MLX with your own Apple
silicon computer is

.. code-block:: shell

    pip install mlx

To install from PyPI you must meet the following requirements:

- Using an M series chip (Apple silicon)
- Using a native Python >= 3.8
- MacOS >= 13.3

.. note::
    MLX is only available on devices running MacOS >= 13.3 
    It is highly recommended to use MacOS 14 (Sonoma)

Troubleshooting
^^^^^^^^^^^^^^^

*My OS and Python versions are in the required range but pip still does not find
a matching distribution.*

Probably you are using a non-native Python. The output of

.. code-block:: shell

  python -c "import platform; print(platform.processor())"

should be ``arm``. If it is ``i386`` (and you have M series machine) then you
are using a non-native Python. Switch your Python to a native Python. A good
way to do this is with
`Conda <https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple>`_.


Build from source
-----------------

Build Requirements
^^^^^^^^^^^^^^^^^^

- A C++ compiler with C++17 support (e.g. Clang >= 5.0)
- `cmake <https://cmake.org/>`_ -- version 3.24 or later, and ``make``
- Xcode >= 14.3 (Xcode >= 15.0 for MacOS 14 and above)


Python API
^^^^^^^^^^

To build and install the MLX python library from source, first, clone MLX from
`its GitHub repo <https://github.com/ml-explore/mlx>`_:

.. code-block:: shell

   git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

Make sure that you have `pybind11 <https://pybind11.readthedocs.io/en/stable/index.html>`_
installed. You can install ``pybind11`` with ``pip``, ``brew`` or ``conda`` as follows:

.. code-block:: shell

    pip install "pybind11[global]"
    conda install pybind11
    brew install pybind11

Then simply build and install it using pip:

.. code-block:: shell

   env CMAKE_BUILD_PARALLEL_LEVEL="" pip install .

For developing use an editable install:

.. code-block:: shell

  env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e .

To make sure the install is working run the tests with:

.. code-block:: shell

  pip install ".[testing]"
  python -m unittest discover python/tests

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
   * - MLX_BUILD_PYTHON_BINDINGS
     - OFF


.. note::

    If you have multiple Xcode installations and wish to use 
    a specific one while building, you can do so by adding the 
    following environment variable before building 

    .. code-block:: shell

      export DEVELOPER_DIR="/path/to/Xcode.app/Contents/Developer/"

    Further, you can use the following command to find out which 
    MacOS SDK will be used

    .. code-block:: shell

      xcrun -sdk macosx --show-sdk-version
