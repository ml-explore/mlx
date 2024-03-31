.. _data_types:

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
   * - ``uint64``
     - 8 
     - 64-bit unsigned integer 
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
   * - ``bfloat16``
     - 2 
     - 16-bit brain float (e8, m7)
   * - ``float16``
     - 2 
     - 16-bit IEEE float (e5, m10)
   * - ``float32``
     - 4 
     - 32-bit float
   * - ``complex64``
     - 8 
     - 64-bit complex float


Data type are aranged in a hierarchy. See the :obj:`DtypeCategory` object
documentation for more information. Use :func:`issubdtype` to determine if one
``dtype`` (or category) is a subtype of another category.

.. autosummary::
   :toctree: _autosummary

   Dtype
   DtypeCategory
   issubdtype
