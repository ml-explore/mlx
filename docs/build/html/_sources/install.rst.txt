Build and Install
=================

Install from PyPI
-----------------

MLX is available at Apple's internal PyPI repository. All you have to do to use
MLX with your own Apple silicon computer is

.. code-block:: shell

    pip install apple-mlx -i https://pypi.apple.com/simple

Build from source
-----------------

Build Requirements
^^^^^^^^^^^^^^^^^^

- A C++ compiler with C++17 support (e.g. Clang >= 5.0)
- `cmake <https://cmake.org/>`_ -- version 3.24 or later, and ``make``


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
