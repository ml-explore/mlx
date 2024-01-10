.. _saving_and_loading:

Saving and Loading Arrays
=========================

.. currentmodule:: mlx.core

MLX supports multiple array serialization formats.

.. list-table:: Serialization Formats
   :widths: 20 8 25 25 
   :header-rows: 1

   * - Format 
     - Extension 
     - Function
     - Notes 
   * - NumPy 
     - ``.npy`` 
     - :func:`save`
     - Single arrays only
   * - NumPy archive 
     - ``.npz`` 
     - :func:`savez` and :func:`savez_compressed`
     - Multiple arrays 
   * - Safetensors
     - ``.safetensors`` 
     - :func:`save_safetensors`
     - Multiple arrays 
   * - GGUF 
     - ``.gguf`` 
     - :func:`save_gguf`
     - Multiple arrays

The :func:`load` function will load any of the supported serialization
formats. It determines the format from the extensions. The output of
:func:`load` depends on the format. 

Here's an example of saving a single array to a file:

.. code-block:: shell

   >>> a = mx.array([1.0])
   >>> mx.save("array", a)

The array ``a`` will be saved in the file ``array.npy`` (notice the extension
is automatically added). Including the extension is optional; if it is missing
it will be added. You can load the array with:

.. code-block:: shell

   >>> mx.load("array.npy", a)
   array([1], dtype=float32)

Here's an example of saving several arrays to a single file:

.. code-block:: shell

   >>> a = mx.array([1.0])
   >>> b = mx.array([2.0])
   >>> mx.savez("arrays", a, b=b)

For compatibility with :func:`numpy.savez` the MLX :func:`savez` takes arrays
as arguments. If the keywords are missing, then default names will be
provided. This can be loaded with:

.. code-block:: shell

   >>> mx.load("arrays.npz")
   {'b': array([2], dtype=float32), 'arr_0': array([1], dtype=float32)}

In this case :func:`load` returns a dictionary of names to arrays.

The functions :func:`save_safetensors` and :func:`save_gguf` are similar to
:func:`savez`, but they take as input a :obj:`dict` of string names to arrays:

.. code-block:: shell

   >>> a = mx.array([1.0])
   >>> b = mx.array([2.0])
   >>> mx.save_safetensors("arrays", {"a": a, "b": b})
