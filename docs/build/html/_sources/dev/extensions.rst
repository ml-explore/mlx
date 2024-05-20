Custom Extensions in MLX
========================

You can extend MLX with custom operations on the CPU or GPU. This guide
explains how to do that with a simple example.

Introducing the Example
-----------------------

Let's say you would like an operation that takes in two arrays, ``x`` and
``y``, scales them both by coefficients ``alpha`` and ``beta`` respectively,
and then adds them together to get the result ``z = alpha * x + beta * y``.
You can do that in MLX directly:

.. code-block:: python

    import mlx.core as mx

    def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
        return alpha * x + beta * y

This function performs that operation while leaving the implementation and
function transformations to MLX.

However you may need to customize the underlying implementation, perhaps to
make it faster or for custom differentiation. In this tutorial we will go
through adding custom extensions. It will cover:

* The structure of the MLX library.
* Implementing a CPU operation that redirects to Accelerate_ when appropriate.
* Implementing a GPU operation using metal.
* Adding the ``vjp`` and ``jvp`` function transformation.
* Building a custom extension and binding it to python.

Operations and Primitives
-------------------------

Operations in MLX build the computation graph. Primitives provide the rules for
evaluating and transforming the graph. Let's start by discussing operations in
more detail.

Operations
^^^^^^^^^^^

Operations are the front-end functions that operate on arrays. They are defined
in the C++ API (:ref:`cpp_ops`), and the Python API (:ref:`ops`) binds them.

We would like an operation, :meth:`axpby` that takes in two arrays ``x`` and
``y``, and two scalars, ``alpha`` and ``beta``. This is how to define it in
C++:

.. code-block:: C++

    /**
    *  Scale and sum two vectors element-wise
    *  z = alpha * x + beta * y
    *
    *  Follow numpy style broadcasting between x and y
    *  Inputs are upcasted to floats if needed
    **/
    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s = {} // Stream on which to schedule the operation
    );

The simplest way to this operation is in terms of existing operations:

.. code-block:: C++

    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
    ) {
        // Scale x and y on the provided stream
        auto ax = multiply(array(alpha), x, s);
        auto by = multiply(array(beta), y, s);

        // Add and return
        return add(ax, by, s);
    }

The operations themselves do not contain the implementations that act on the
data, nor do they contain the rules of transformations. Rather, they are an
easy to use interface that use :class:`Primitive` building blocks.

Primitives
^^^^^^^^^^^

A :class:`Primitive` is part of the computation graph of an :class:`array`. It
defines how to create outputs arrays given a input arrays. Further, a
:class:`Primitive` has methods to run on the CPU or GPU and for function
transformations such as ``vjp`` and ``jvp``.  Lets go back to our example to be
more concrete:

.. code-block:: C++

    class Axpby : public Primitive {
      public:
        explicit Axpby(Stream stream, float alpha, float beta)
            : Primitive(stream), alpha_(alpha), beta_(beta){};

        /**
        * A primitive must know how to evaluate itself on the CPU/GPU
        * for the given inputs and populate the output array.
        *
        * To avoid unnecessary allocations, the evaluation function
        * is responsible for allocating space for the array.
        */
        void eval_cpu(
            const std::vector<array>& inputs,
            std::vector<array>& outputs) override;
        void eval_gpu(
            const std::vector<array>& inputs,
            std::vector<array>& outputs) override;

        /** The Jacobian-vector product. */
        std::vector<array> jvp(
            const std::vector<array>& primals,
            const std::vector<array>& tangents,
            const std::vector<int>& argnums) override;

        /** The vector-Jacobian product. */
        std::vector<array> vjp(
            const std::vector<array>& primals,
            const array& cotan,
            const std::vector<int>& argnums,
            const std::vector<array>& outputs) override;

        /**
        * The primitive must know how to vectorize itself across
        * the given axes. The output is a pair containing the array
        * representing the vectorized computation and the axis which
        * corresponds to the output vectorized dimension.
        */
        virtual std::pair<std::vector<array>, std::vector<int>> vmap(
            const std::vector<array>& inputs,
            const std::vector<int>& axes) override;

        /** Print the primitive. */
        void print(std::ostream& os) override {
            os << "Axpby";
        }

        /** Equivalence check **/
        bool is_equivalent(const Primitive& other) const override;

      private:
        float alpha_;
        float beta_;

        /** Fall back implementation for evaluation on CPU */
        void eval(const std::vector<array>& inputs, array& out);
    };

The :class:`Axpby` class derives from the base :class:`Primitive` class. The
:class:`Axpby` treats ``alpha`` and ``beta`` as parameters. It then provides
implementations of how the output array is produced given the inputs through
:meth:`Axpby::eval_cpu` and :meth:`Axpby::eval_gpu`. It also provides rules
of transformations in :meth:`Axpby::jvp`, :meth:`Axpby::vjp`, and
:meth:`Axpby::vmap`.

Using the Primitive
^^^^^^^^^^^^^^^^^^^

Operations can use this :class:`Primitive` to add a new :class:`array` to the
computation graph. An :class:`array` can be constructed by providing its data
type, shape, the :class:`Primitive` that computes it, and the :class:`array`
inputs that are passed to the primitive.

Let's reimplement our operation now in terms of our :class:`Axpby` primitive.

.. code-block:: C++

    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
    ) {
        // Promote dtypes between x and y as needed
        auto promoted_dtype = promote_types(x.dtype(), y.dtype());

        // Upcast to float32 for non-floating point inputs x and y
        auto out_dtype = is_floating_point(promoted_dtype)
            ? promoted_dtype
            : promote_types(promoted_dtype, float32);

        // Cast x and y up to the determined dtype (on the same stream s)
        auto x_casted = astype(x, out_dtype, s);
        auto y_casted = astype(y, out_dtype, s);

        // Broadcast the shapes of x and y (on the same stream s)
        auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
        auto out_shape = broadcasted_inputs[0].shape();

        // Construct the array as the output of the Axpby primitive
        // with the broadcasted and upcasted arrays as inputs
        return array(
            /* const std::vector<int>& shape = */ out_shape,
            /* Dtype dtype = */ out_dtype,
            /* std::unique_ptr<Primitive> primitive = */
            std::make_shared<Axpby>(to_stream(s), alpha, beta),
            /* const std::vector<array>& inputs = */ broadcasted_inputs);
    }


This operation now handles the following:

#. Upcast inputs and resolve the output data type.
#. Broadcast the inputs and resolve the output shape.
#. Construct the primitive :class:`Axpby` using the given stream, ``alpha``, and ``beta``.
#. Construct the output :class:`array` using the primitive and the inputs.

Implementing the Primitive
--------------------------

No computation happens when we call the operation alone. The operation only
builds the computation graph. When we evaluate the output array, MLX schedules
the execution of the computation graph, and calls :meth:`Axpby::eval_cpu` or
:meth:`Axpby::eval_gpu` depending on the stream/device specified by the user.

.. warning::
    When :meth:`Primitive::eval_cpu` or :meth:`Primitive::eval_gpu` are called,
    no memory has been allocated for the output array. It falls on the implementation
    of these functions to allocate memory as needed.

Implementing the CPU Back-end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by implementing a naive and generic version of
:meth:`Axpby::eval_cpu`. We declared this as a private member function of
:class:`Axpby` earlier called :meth:`Axpby::eval`.

Our naive method will go over each element of the output array, find the
corresponding input elements of ``x`` and ``y`` and perform the operation
point-wise. This is captured in the templated function :meth:`axpby_impl`.

.. code-block:: C++

    template <typename T>
    void axpby_impl(
            const array& x,
            const array& y,
            array& out,
            float alpha_,
            float beta_) {
        // We only allocate memory when we are ready to fill the output
        // malloc_or_wait synchronously allocates available memory
        // There may be a wait executed here if the allocation is requested
        // under memory-pressured conditions
        out.set_data(allocator::malloc_or_wait(out.nbytes()));

        // Collect input and output data pointers
        const T* x_ptr = x.data<T>();
        const T* y_ptr = y.data<T>();
        T* out_ptr = out.data<T>();

        // Cast alpha and beta to the relevant types
        T alpha = static_cast<T>(alpha_);
        T beta = static_cast<T>(beta_);

        // Do the element-wise operation for each output
        for (size_t out_idx = 0; out_idx < out.size(); out_idx++) {
            // Map linear indices to offsets in x and y
            auto x_offset = elem_to_loc(out_idx, x.shape(), x.strides());
            auto y_offset = elem_to_loc(out_idx, y.shape(), y.strides());

            // We allocate the output to be contiguous and regularly strided
            // (defaults to row major) and hence it doesn't need additional mapping
            out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
        }
    }

Our implementation should work for all incoming floating point arrays.
Accordingly, we add dispatches for ``float32``, ``float16``, ``bfloat16`` and
``complex64``. We throw an error if we encounter an unexpected type.

.. code-block:: C++

    /** Fall back implementation for evaluation on CPU */
    void Axpby::eval(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs) {
        auto& x = inputs[0];
        auto& y = inputs[1];
        auto& out = outputs[0];

        // Dispatch to the correct dtype
        if (out.dtype() == float32) {
            return axpby_impl<float>(x, y, out, alpha_, beta_);
        } else if (out.dtype() == float16) {
            return axpby_impl<float16_t>(x, y, out, alpha_, beta_);
        } else if (out.dtype() == bfloat16) {
            return axpby_impl<bfloat16_t>(x, y, out, alpha_, beta_);
        } else if (out.dtype() == complex64) {
            return axpby_impl<complex64_t>(x, y, out, alpha_, beta_);
        } else {
            throw std::runtime_error(
                "[Axpby] Only supports floating point types.");
        }
    }

This is good as a fallback implementation. We can use the ``axpby`` routine
provided by the Accelerate_ framework for a faster implementation in certain
cases:

#.  Accelerate does not provide implementations of ``axpby`` for half precision
    floats. We can only use it for ``float32`` types.
#.  Accelerate assumes the inputs ``x`` and ``y`` are contiguous and all
    elements have fixed strides between them. We only direct to Accelerate
    if both ``x`` and ``y`` are row contiguous or column contiguous.
#.  Accelerate performs the routine ``Y = (alpha * X) + (beta * Y)`` in-place.
    MLX expects to write the output to a new array. We must copy the elements
    of ``y`` into the output and use that as an input to ``axpby``.

Let's write an implementation that uses Accelerate in the right conditions.
It allocates data for the output, copies ``y`` into it, and then calls the
:func:`catlas_saxpby` from accelerate.

.. code-block:: C++

    template <typename T>
    void axpby_impl_accelerate(
            const array& x,
            const array& y,
            array& out,
            float alpha_,
            float beta_) {
        // Accelerate library provides catlas_saxpby which does
        // Y = (alpha * X) + (beta * Y) in place
        // To use it, we first copy the data in y over to the output array
        out.set_data(allocator::malloc_or_wait(out.nbytes()));

        // We then copy over the elements using the contiguous vector specialization
        copy_inplace(y, out, CopyType::Vector);

        // Get x and y pointers for catlas_saxpby
        const T* x_ptr = x.data<T>();
        T* y_ptr = out.data<T>();

        T alpha = static_cast<T>(alpha_);
        T beta = static_cast<T>(beta_);

        // Call the inplace accelerate operator
        catlas_saxpby(
            /* N = */ out.size(),
            /* ALPHA = */ alpha,
            /* X = */ x_ptr,
            /* INCX = */ 1,
            /* BETA = */ beta,
            /* Y = */ y_ptr,
            /* INCY = */ 1);
    }

For inputs that do not fit the criteria for accelerate, we fall back to
:meth:`Axpby::eval`. With this in mind, let's finish our
:meth:`Axpby::eval_cpu`.

.. code-block:: C++

    /** Evaluate primitive on CPU using accelerate specializations */
    void Axpby::eval_cpu(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs) {
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];
        auto& out = outputs[0];

        // Accelerate specialization for contiguous single precision float arrays
        if (out.dtype() == float32 &&
            ((x.flags().row_contiguous && y.flags().row_contiguous) ||
            (x.flags().col_contiguous && y.flags().col_contiguous))) {
            axpby_impl_accelerate<float>(x, y, out, alpha_, beta_);
            return;
        }

        // Fall back to common back-end if specializations are not available
        eval(inputs, outputs);
    }

Just this much is enough to run the operation :meth:`axpby` on a CPU stream! If
you do not plan on running the operation on the GPU or using transforms on
computation graphs that contain :class:`Axpby`, you can stop implementing the
primitive here and enjoy the speed-ups you get from the Accelerate library.

Implementing the GPU Back-end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apple silicon devices address their GPUs using the Metal_ shading language, and
GPU kernels in MLX are written using Metal.

.. note::

    Here are some helpful resources if you are new to Metal:

    * A walkthrough of the metal compute pipeline: `Metal Example`_
    * Documentation for metal shading language: `Metal Specification`_
    * Using metal from C++: `Metal-cpp`_

Let's keep the GPU kernel simple. We will launch exactly as many threads as
there are elements in the output. Each thread will pick the element it needs
from ``x`` and ``y``, do the point-wise operation, and update its assigned
element in the output.

.. code-block:: C++

    template <typename T>
    [[kernel]] void axpby_general(
            device const T* x [[buffer(0)]],
            device const T* y [[buffer(1)]],
            device T* out [[buffer(2)]],
            constant const float& alpha [[buffer(3)]],
            constant const float& beta [[buffer(4)]],
            constant const int* shape [[buffer(5)]],
            constant const size_t* x_strides [[buffer(6)]],
            constant const size_t* y_strides [[buffer(7)]],
            constant const int& ndim [[buffer(8)]],
            uint index [[thread_position_in_grid]]) {
        // Convert linear indices to offsets in array
        auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
        auto y_offset = elem_to_loc(index, shape, y_strides, ndim);

        // Do the operation and update the output
        out[index] =
            static_cast<T>(alpha) * x[x_offset] + static_cast<T>(beta) * y[y_offset];
    }

We then need to instantiate this template for all floating point types and give
each instantiation a unique host name so we can identify it.

.. code-block:: C++

    #define instantiate_axpby(type_name, type)              \
        template [[host_name("axpby_general_" #type_name)]] \
        [[kernel]] void axpby_general<type>(                \
            device const type* x [[buffer(0)]],             \
            device const type* y [[buffer(1)]],             \
            device type* out [[buffer(2)]],                 \
            constant const float& alpha [[buffer(3)]],      \
            constant const float& beta [[buffer(4)]],       \
            constant const int* shape [[buffer(5)]],        \
            constant const size_t* x_strides [[buffer(6)]], \
            constant const size_t* y_strides [[buffer(7)]], \
            constant const int& ndim [[buffer(8)]],         \
            uint index [[thread_position_in_grid]]);

    instantiate_axpby(float32, float);
    instantiate_axpby(float16, half);
    instantiate_axpby(bfloat16, bfloat16_t);
    instantiate_axpby(complex64, complex64_t);

The logic to determine the kernel, set the inputs, resolve the grid dimensions,
and dispatch to the GPU are contained in :meth:`Axpby::eval_gpu` as shown
below.

.. code-block:: C++

    /** Evaluate primitive on GPU */
    void Axpby::eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) {
        // Prepare inputs
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];
        auto& out = outputs[0];

        // Each primitive carries the stream it should execute on
        // and each stream carries its device identifiers
        auto& s = stream();
        // We get the needed metal device using the stream
        auto& d = metal::device(s.device);

        // Allocate output memory
        out.set_data(allocator::malloc_or_wait(out.nbytes()));

        // Resolve name of kernel
        std::ostringstream kname;
        kname << "axpby_" << "general_" << type_to_name(out);

        // Make sure the metal library is available and look for it
        // in the same folder as this executable if needed
        d.register_library("mlx_ext", metal::get_colocated_mtllib_path);

        // Make a kernel from this metal library
        auto kernel = d.get_kernel(kname.str(), "mlx_ext");

        // Prepare to encode kernel
        auto& compute_encoder = d.get_command_encoder(s.index);
        compute_encoder->setComputePipelineState(kernel);

        // Kernel parameters are registered with buffer indices corresponding to
        // those in the kernel declaration at axpby.metal
        int ndim = out.ndim();
        size_t nelem = out.size();

        // Encode input arrays to kernel
        compute_encoder.set_input_array(x, 0);
        compute_encoder.set_input_array(y, 1);

        // Encode output arrays to kernel
        compute_encoder.set_output_array(out, 2);

        // Encode alpha and beta
        compute_encoder->setBytes(&alpha_, sizeof(float), 3);
        compute_encoder->setBytes(&beta_, sizeof(float), 4);

        // Encode shape, strides and ndim
        compute_encoder->setBytes(x.shape().data(), ndim * sizeof(int), 5);
        compute_encoder->setBytes(x.strides().data(), ndim * sizeof(size_t), 6);
        compute_encoder->setBytes(y.strides().data(), ndim * sizeof(size_t), 7);
        compute_encoder->setBytes(&ndim, sizeof(int), 8);

        // We launch 1 thread for each input and make sure that the number of
        // threads in any given threadgroup is not higher than the max allowed
        size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());

        // Fix the 3D size of each threadgroup (in terms of threads)
        MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);

        // Fix the 3D size of the launch grid (in terms of threads)
        MTL::Size grid_dims = MTL::Size(nelem, 1, 1);

        // Launch the grid with the given number of threads divided among
        // the given threadgroups
        compute_encoder.dispatchThreads(grid_dims, group_dims);
    }

We can now call the :meth:`axpby` operation on both the CPU and the GPU!

A few things to note about MLX and Metal before moving on. MLX keeps track of
the active ``command_buffer`` and the ``MTLCommandBuffer`` to which it is
associated. We rely on :meth:`d.get_command_encoder` to give us the active
metal compute command encoder instead of building a new one and calling
:meth:`compute_encoder->end_encoding` at the end. MLX adds kernels (compute
pipelines) to the active command buffer until some specified limit is hit or
the command buffer needs to be flushed for synchronization.

Primitive Transforms
^^^^^^^^^^^^^^^^^^^^^

Next, let's add implementations for transformations in a :class:`Primitive`.
These transformations can be built on top of other operations, including the
one we just defined:

.. code-block:: C++

    /** The Jacobian-vector product. */
    std::vector<array> Axpby::jvp(
            const std::vector<array>& primals,
            const std::vector<array>& tangents,
            const std::vector<int>& argnums) {
        // Forward mode diff that pushes along the tangents
        // The jvp transform on the primitive can built with ops
        // that are scheduled on the same stream as the primitive

        // If argnums = {0}, we only push along x in which case the
        // jvp is just the tangent scaled by alpha
        // Similarly, if argnums = {1}, the jvp is just the tangent
        // scaled by beta
        if (argnums.size() > 1) {
            auto scale = argnums[0] == 0 ? alpha_ : beta_;
            auto scale_arr = array(scale, tangents[0].dtype());
            return {multiply(scale_arr, tangents[0], stream())};
        }
        // If, argnums = {0, 1}, we take contributions from both
        // which gives us jvp = tangent_x * alpha + tangent_y * beta
        else {
            return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
        }
    }

.. code-block:: C++

    /** The vector-Jacobian product. */
    std::vector<array> Axpby::vjp(
            const std::vector<array>& primals,
            const std::vector<array>& cotangents,
            const std::vector<int>& argnums,
            const std::vector<int>& /* unused */) {
        // Reverse mode diff
        std::vector<array> vjps;
        for (auto arg : argnums) {
            auto scale = arg == 0 ? alpha_ : beta_;
            auto scale_arr = array(scale, cotangents[0].dtype());
            vjps.push_back(multiply(scale_arr, cotangents[0], stream()));
        }
        return vjps;
    }

Note, a transformation does not need to be fully defined to start using
the :class:`Primitive`.

.. code-block:: C++

    /** Vectorize primitive along given axis */
    std::pair<std::vector<array>, std::vector<int>> Axpby::vmap(
            const std::vector<array>& inputs,
            const std::vector<int>& axes) {
        throw std::runtime_error("[Axpby] vmap not implemented.");
    }

Building and Binding
--------------------

Let's look at the overall directory structure first.

| extensions
| ├── axpby
| │   ├── axpby.cpp
| │   ├── axpby.h
| │   └── axpby.metal
| ├── mlx_sample_extensions
| │   └── __init__.py
| ├── bindings.cpp
| ├── CMakeLists.txt
| └── setup.py

* ``extensions/axpby/`` defines the C++ extension library
* ``extensions/mlx_sample_extensions`` sets out the structure for the
  associated Python package
* ``extensions/bindings.cpp`` provides Python bindings for our operation
* ``extensions/CMakeLists.txt`` holds CMake rules to build the library and
  Python bindings
* ``extensions/setup.py`` holds the ``setuptools`` rules to build and install
  the Python package

Binding to Python
^^^^^^^^^^^^^^^^^^

We use nanobind_ to build a Python API for the C++ library. Since bindings for
components such as :class:`mlx.core.array`, :class:`mlx.core.stream`, etc. are
already provided, adding our :meth:`axpby` is simple.

.. code-block:: C++

   NB_MODULE(_ext, m) {
        m.doc() = "Sample extension for MLX";

        m.def(
            "axpby",
            &axpby,
            "x"_a,
            "y"_a,
            "alpha"_a,
            "beta"_a,
            nb::kw_only(),
            "stream"_a = nb::none(),
            R"(
                Scale and sum two vectors element-wise
                ``z = alpha * x + beta * y``

                Follows numpy style broadcasting between ``x`` and ``y``
                Inputs are upcasted to floats if needed

                Args:
                    x (array): Input array.
                    y (array): Input array.
                    alpha (float): Scaling factor for ``x``.
                    beta (float): Scaling factor for ``y``.

                Returns:
                    array: ``alpha * x + beta * y``
            )");
    }

Most of the complexity in the above example comes from additional bells and
whistles such as the literal names and doc-strings.

.. warning::

    :mod:`mlx.core` must be imported before importing
    :mod:`mlx_sample_extensions` as defined by the nanobind module above to
    ensure that the casters for :mod:`mlx.core` components like
    :class:`mlx.core.array` are available.

.. _Building with CMake:

Building with CMake
^^^^^^^^^^^^^^^^^^^^

Building the C++ extension library only requires that you ``find_package(MLX
CONFIG)`` and then link it to your library.

.. code-block:: cmake

    # Add library
    add_library(mlx_ext)

    # Add sources
    target_sources(
        mlx_ext
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp
    )

    # Add include headers
    target_include_directories(
        mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )

    # Link to mlx
    target_link_libraries(mlx_ext PUBLIC mlx)

We also need to build the attached Metal library. For convenience, we provide a
:meth:`mlx_build_metallib` function that builds a ``.metallib`` target given
sources, headers, destinations, etc. (defined in ``cmake/extension.cmake`` and
automatically imported with MLX package).

Here is what that looks like in practice:

.. code-block:: cmake

    # Build metallib
    if(MLX_BUILD_METAL)

    mlx_build_metallib(
        TARGET mlx_ext_metallib
        TITLE mlx_ext
        SOURCES ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
        INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
        OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )

    add_dependencies(
        mlx_ext
        mlx_ext_metallib
    )

    endif()

Finally, we build the nanobind_ bindings

.. code-block:: cmake

    nanobind_add_module(
      _ext
      NB_STATIC STABLE_ABI LTO NOMINSIZE
      NB_DOMAIN mlx
      ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
    )
    target_link_libraries(_ext PRIVATE mlx_ext)

    if(BUILD_SHARED_LIBS)
      target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
    endif()

Building with ``setuptools``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have set out the CMake build rules as described above, we can use the
build utilities defined in :mod:`mlx.extension`:

.. code-block:: python

    from mlx import extension
    from setuptools import setup

    if __name__ == "__main__":
        setup(
            name="mlx_sample_extensions",
            version="0.0.0",
            description="Sample C++ and Metal extensions for MLX primitives.",
            ext_modules=[extension.CMakeExtension("mlx_sample_extensions._ext")],
            cmdclass={"build_ext": extension.CMakeBuild},
            packages=["mlx_sample_extensions"],
            package_data={"mlx_sample_extensions": ["*.so", "*.dylib", "*.metallib"]},
            extras_require={"dev":[]},
            zip_safe=False,
            python_requires=">=3.8",
        )

.. note::
    We treat ``extensions/mlx_sample_extensions`` as the package directory
    even though it only contains a ``__init__.py`` to ensure the following:

    * :mod:`mlx.core` must be imported before importing :mod:`_ext`
    * The C++ extension library and the metal library are co-located with the python
      bindings and copied together if the package is installed

To build the package, first install the build dependencies with ``pip install
-r requirements.txt``.  You can then build inplace for development using
``python setup.py build_ext -j8 --inplace`` (in ``extensions/``)

This results in the directory structure:

| extensions
| ├── mlx_sample_extensions
| │   ├── __init__.py
| │   ├── libmlx_ext.dylib # C++ extension library
| │   ├── mlx_ext.metallib # Metal library
| │   └── _ext.cpython-3x-darwin.so # Python Binding
| ...

When you try to install using the command ``python -m pip install .`` (in
``extensions/``), the package will be installed with the same structure as
``extensions/mlx_sample_extensions`` and the C++ and Metal library will be
copied along with the Python binding since they are specified as
``package_data``.

Usage
-----

After installing the extension as described above, you should be able to simply
import the Python package and play with it as you would any other MLX operation.

Let's look at a simple script and its results:

.. code-block:: python

    import mlx.core as mx
    from mlx_sample_extensions import axpby

    a = mx.ones((3, 4))
    b = mx.ones((3, 4))
    c = axpby(a, b, 4.0, 2.0, stream=mx.cpu)

    print(f"c shape: {c.shape}")
    print(f"c dtype: {c.dtype}")
    print(f"c correct: {mx.all(c == 6.0).item()}")

Output:

.. code-block::

    c shape: [3, 4]
    c dtype: float32
    c correctness: True

Results
^^^^^^^

Let's run a quick benchmark and see how our new ``axpby`` operation compares
with the naive :meth:`simple_axpby` we first defined on the CPU.

.. code-block:: python

    import mlx.core as mx
    from mlx_sample_extensions import axpby
    import time

    mx.set_default_device(mx.cpu)

    def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
        return alpha * x + beta * y

    M = 256
    N = 512

    x = mx.random.normal((M, N))
    y = mx.random.normal((M, N))
    alpha = 4.0
    beta = 2.0

    mx.eval(x, y)

    def bench(f):
        # Warm up
        for i in range(100):
            z = f(x, y, alpha, beta)
            mx.eval(z)

        # Timed run
        s = time.time()
        for i in range(5000):
            z = f(x, y, alpha, beta)
            mx.eval(z)
        e = time.time()
        return e - s

    simple_time = bench(simple_axpby)
    custom_time = bench(axpby)

    print(f"Simple axpby: {simple_time:.3f} s | Custom axpby: {custom_time:.3f} s")

The results are ``Simple axpby: 0.114 s | Custom axpby: 0.109 s``. We see
modest improvements right away!

This operation is now good to be used to build other operations, in
:class:`mlx.nn.Module` calls, and also as a part of graph transformations like
:meth:`grad`.

Scripts
-------

.. admonition:: Download the code

   The full example code is available in `mlx <https://github.com/ml-explore/mlx/tree/main/examples/extensions/>`_.

.. _Accelerate: https://developer.apple.com/documentation/accelerate/blas?language=objc
.. _Metal: https://developer.apple.com/documentation/metal?language=objc
.. _Metal-cpp: https://developer.apple.com/metal/cpp/
.. _`Metal Specification`: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
.. _`Metal Example`: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc
.. _nanobind: https://nanobind.readthedocs.io/en/latest/
