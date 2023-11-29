.. _data_types:

:orphan:

Data Types
==========

.. currentmodule:: mlx.core

The default floating point type is ``float32`` and the default integer type is
``int32``. The table below shows supported values for :obj:`Dtype`. 

.. list-table:: Supported Data Types 
   :widths: 5 3 20
   :header-rows: 1

   * - Type 
     - Bytes
     - Description
   * - ``bool_``
     - 1 
     - Boolean (``True``, ``False``) data type
   * - ``uint8``
     - 1 
     - 8-bit unsigned integer 
   * - ``uint16``
     - 2 
     - 16-bit unsigned integer 
   * - ``uint32``
     - 4 
     - 32-bit unsigned integer 
   * - ``uint32``
     - 8 
     - 32-bit unsigned integer 
   * - ``int8``
     - 1 
     - 8-bit signed integer 
   * - ``int16``
     - 2 
     - 16-bit signed integer 
   * - ``int32``
     - 4 
     - 32-bit signed integer 
   * - ``int64``
     - 8 
     - 64-bit signed integer 
   * - ``float16``
     - 2 
     - 16-bit float, only available with `ARM C language extensions <https://developer.arm.com/documentation/101028/0012/3--C-language-extensions?lang=en>`_
   * - ``float32``
     - 4 
     - 32-bit float
