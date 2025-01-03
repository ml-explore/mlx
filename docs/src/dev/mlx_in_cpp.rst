.. _mlx_in_cpp:

Using MLX in C++
================

You can use MLX in a C++ project with CMake.

.. note::

  This guide is based one the following `example using MLX in C++ 
  <https://github.com/ml-explore/mlx/tree/main/examples/cmake_project>`_

First install MLX:

.. code-block:: bash

  pip install -U mlx

You can also install the MLX Python package from source or just the C++
library. For more information see the :ref:`documentation on installing MLX
<build_and_install>`.

Next make an example program in ``example.cpp``: 

.. code-block:: C++

  #include <iostream>

  #include "mlx/mlx.h"

  namespace mx = mlx::core;

  int main() {
    auto x = mx::array({1, 2, 3});
    auto y = mx::array({1, 2, 3});
    std::cout << x + y << std::endl;
    return 0;
  }

The next step is to setup a CMake file in ``CMakeLists.txt``:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.27)

  project(example LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)


Depending on how you installed MLX, you may need to tell CMake where to
find it. 

If you installed MLX with Python, then add the following to the CMake file:

.. code-block:: cmake

  find_package(
    Python 3.9
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT)

If you installed the MLX C++ package to a system path, then CMake should be
able to find it. If you installed it to a non-standard location or CMake can't
find MLX then set ``MLX_ROOT`` to the location where MLX is installed:

.. code-block:: cmake

  set(MLX_ROOT "/path/to/mlx/")

Next, instruct CMake to find MLX:

.. code-block:: cmake

  find_package(MLX CONFIG REQUIRED)

Finally, add the ``example.cpp`` program as an executable and link MLX.

.. code-block:: cmake

  add_executable(example example.cpp)
  target_link_libraries(example PRIVATE mlx)

You can build the example with:

.. code-block:: bash

  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build

And run it with:

.. code-block:: bash

  ./build/example

Note ``find_package(MLX CONFIG REQUIRED)`` sets the following variables:

.. list-table:: Package Variables
   :widths: 20 20 
   :header-rows: 1

   * - Variable 
     - Description 
   * - MLX_FOUND
     - ``True`` if MLX is found
   * - MLX_INCLUDE_DIRS
     - Include directory
   * - MLX_LIBRARIES
     - Libraries to link against
   * - MLX_CXX_FLAGS
     - Additional compiler flags
   * - MLX_BUILD_ACCELERATE
     - ``True`` if MLX was built with Accelerate 
   * - MLX_BUILD_METAL
     - ``True`` if MLX was built with Metal
