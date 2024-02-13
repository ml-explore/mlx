Developer Documentation
=======================

MLX provides a open and flexible backend to which users may add operations 
and specialized implementations without much hassle. While the library supplies
efficient operations that can be used and composed for any number of 
applications, there may arise cases where new functionalities or highly 
optimized implementations are needed. For such cases, you may design and 
implement your own operations that link to and build on top of :mod:`mlx.core`.
We will introduce the inner-workings of MLX and go over a simple example to 
learn the steps involved in adding new operations to MLX with your own CPU 
and GPU implementations. 

Introducing the Example
-----------------------

Let's say that you would like an operation that takes in two arrays, 
``x`` and ``y``, scales them both by some coefficients ``alpha`` and ``beta``
respectively, and then adds them together to get the result 
``z = alpha * x + beta * y``. Well, you can very easily do that by just 
writing out a function as follows:

.. code-block:: python

    import mlx.core as mx

    def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
        return alpha * x + beta * y

This function performs that operation while leaving the implementations and 
differentiation to MLX. 

However, you work with vector math libraries often and realize that the 
``axpby`` routine defines the same operation ``Y = (alpha * X) + (beta * Y)``. 
You would really like the part of your applications that does this operation 
on the CPU to be very fast - so you decide that you want it to rely on the 
``axpby`` routine provided by the Accelerate_ framework. Continuing to impose 
our assumptions on to you, let's also assume that you want to learn how to add 
your own implementation for the gradients of your new operation while going 
over the ins-and-outs of the MLX framework. 

Well, what a coincidence! You are in the right place. Over the course of this 
example, we will learn:

* The structure of the MLX library from the frontend API to the backend implementations.
* How to implement your own CPU backend that redirects to Accelerate_ when appropriate (and a fallback if needed).
* How to implement your own GPU implementation using metal.
* How to add your own ``vjp`` and ``jvp``.
* How to build your implementations, link them to MLX, and bind them to python.

Operations and Primitives
-------------------------

In one sentence, operations in MLX build the computation graph, and primitives 
provide the rules for evaluation and transformations of said graph. Let's start 
by discussing operations in more detail. 

Operations
^^^^^^^^^^^

Operations are the frontend functions that operate on arrays. They are defined 
in the C++ API (:ref:`cpp_ops`) and then we provide bindings to these 
operations in the Python API (:ref:`ops`). 

We would like an operation, :meth:`axpby` that takes in two arrays ``x`` and ``y``,
and two scalars, ``alpha`` and ``beta``. This is how we would define it in the 
C++ API:

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


This operation itself can call other operations within it if needed. So, the 
simplest way to go about implementing this operation would be do so in terms 
of existing operations. 

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

However, as we discussed earlier, this is not our goal. The operations themselves 
do not contain the implementations that act on the data, nor do they contain the
rules of transformations. Rather, they are an easy to use interface that build 
on top of the building blocks we call :class:`Primitive`. 

Primitives
^^^^^^^^^^^

A :class:`Primitive` is part of the computation graph of an :class:`array`. It 
defines how to create an output given a set of input :class:`array` . Further,
a :class:`Primitive` is a class that contains rules on how it is evaluated 
on the CPU or GPU, and how it acts under transformations such as ``vjp`` and 
``jvp``. These words on their own can be a bit abstract, so lets take a step 
back and go to our example to give ourselves a more concrete image. 

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
        void eval_cpu(const std::vector<array>& inputs, array& out) override;
        void eval_gpu(const std::vector<array>& inputs, array& out) override;

        /** The Jacobian-vector product. */
        array jvp(
            const std::vector<array>& primals,
            const std::vector<array>& tangents,
            const std::vector<int>& argnums) override;

        /** The vector-Jacobian product. */
        std::vector<array> vjp(
            const std::vector<array>& primals,
            const array& cotan,
            const std::vector<int>& argnums) override;

        /**
        * The primitive must know how to vectorize itself across
        * the given axes. The output is a pair containing the array
        * representing the vectorized computation and the axis which
        * corresponds to the output vectorized dimension.
        */
        std::pair<array, int> vmap(
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

The :class:`Axpby` class derives from the base :class:`Primitive` class and 
follows the above demonstrated interface. :class:`Axpby` treats ``alpha`` and 
``beta`` as parameters. It then provides implementations of how the array ``out`` 
is produced given ``inputs`` through :meth:`Axpby::eval_cpu` and 
:meth:`Axpby::eval_gpu`. Further, it provides rules of transformations in 
:meth:`Axpby::jvp`, :meth:`Axpby::vjp`, and :meth:`Axpby::vmap`. 

Using the Primitives
^^^^^^^^^^^^^^^^^^^^^

Operations can use this :class:`Primitive` to add a new :class:`array` to 
the computation graph. An :class:`array` can be constructed by providing its 
data type, shape, the :class:`Primitive` that computes it, and the 
:class:`array` inputs that are passed to the primitive.

Let's re-implement our operation now in terms of our :class:`Axpby` primitive.

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
            std::make_unique<Axpby>(to_stream(s), alpha, beta),
            /* const std::vector<array>& inputs = */ broadcasted_inputs);
    }


This operation now handles the following:

#. Upcast inputs and resolve the output data type.
#. Broadcast the inputs and resolve the output shape.
#. Construct the primitive :class:`Axpby` using the given stream, ``alpha``, and ``beta``.
#. Construct the output :class:`array` using the primitive and the inputs.

Implementing the Primitive
--------------------------

No computation happens when we call the operation alone. In effect, the 
operation only builds the computation graph. When we evaluate the output 
array, MLX schedules the execution of the computation graph, and calls
:meth:`Axpby::eval_cpu` or :meth:`Axpby::eval_gpu` depending on the 
stream/device specified by the user. 

.. warning::
    When :meth:`Primitive::eval_cpu` or :meth:`Primitive::eval_gpu` are called,
    no memory has been allocated for the output array. It falls on the implementation
    of these functions to allocate memory as needed

Implementing the CPU Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by trying to implement a naive and generic version of 
:meth:`Axpby::eval_cpu`. We declared this as a private member function of 
:class:`Axpby` earlier called :meth:`Axpby::eval`. 

Our naive method will go over each element of the output array, find the 
corresponding input elements of ``x`` and ``y`` and perform the operation 
pointwise. This is captured in the templated function :meth:`axpby_impl`. 

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

Now, we would like our implementation to be able to do this pointwise operation 
for all incoming floating point arrays. Accordingly, we add dispatches for 
``float32``, ``float16``, ``bfloat16`` and ``complex64``. We throw an error 
if we encounter an unexpected type.

.. code-block:: C++

    /** Fall back implementation for evaluation on CPU */
    void Axpby::eval(const std::vector<array>& inputs, array& out) {
        // Check the inputs (registered in the op while constructing the out array)
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];

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
                "Axpby is only supported for floating point types.");
        }
    }

We have a fallback implementation! Now, to do what we are really here to do. 
Remember we wanted to use the ``axpby`` routine provided by the Accelerate_
framework? Well, there are 3 complications to keep in mind:

#.  Accelerate does not provide implementations of ``axpby`` for half precision
    floats. We can only direct to it for ``float32`` types 
#.  Accelerate assumes the inputs ``x`` and ``y`` are contiguous and all elements
    have fixed strides between them. Possibly due to broadcasts and transposes, 
    we aren't guaranteed that the inputs fit this requirement. We can 
    only direct to Accelerate if both ``x`` and ``y`` are row contiguous or 
    column contiguous. 
#.  Accelerate performs the routine ``Y = (alpha * X) + (beta * Y)`` inplace. 
    MLX expects to write out the answer to a new array. We must copy the elements 
    of ``y`` into the output array and use that as an input to ``axpby``

Let's write out an implementation that uses Accelerate in the right conditions. 
It must simply allocate data for the output, copy elements of ``y`` into it, 
and then call the :meth:`catlas_saxpby` from accelerate. 

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

        // This specialization requires both x and y be contiguous in the same mode
        // i.e: corresponding linear indices in both point to corresponding elements
        // The data in the output array is allocated to match the strides in y
        // such that x, y, and out are contiguous in the same mode and
        // no transposition is needed
        out.set_data(
            allocator::malloc_or_wait(y.data_size() * out.itemsize()),
            y.data_size(),
            y.strides(),
            y.flags());

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

Great! But what about the inputs that do not fit the criteria for accelerate?
Luckily, we can always just direct back to :meth:`Axpby::eval`.

With this in mind, lets finally implement our :meth:`Axpby::eval_cpu`.

.. code-block:: C++

    /** Evaluate primitive on CPU using accelerate specializations */
    void Axpby::eval_cpu(const std::vector<array>& inputs, array& out) {
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];

        // Accelerate specialization for contiguous single precision float arrays
        if (out.dtype() == float32 &&
            ((x.flags().row_contiguous && y.flags().row_contiguous) ||
            (x.flags().col_contiguous && y.flags().col_contiguous))) {
            axpby_impl_accelerate<float>(x, y, out, alpha_, beta_);
            return;
        }

        // Fall back to common backend if specializations are not available
        eval(inputs, out);
    }

We have now hit a milestone! Just this much is enough to run the operation 
:meth:`axpby` on a CPU stream! 

If you do not plan on running the operation on the GPU or using transforms on 
computation graphs that contain :class:`Axpby`, you can stop implementing the 
primitive here and enjoy the speed-ups you get from the Accelerate library. 

Implementing the GPU Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apple silicon devices address their GPUs using the Metal_ shading language, and 
all GPU kernels in MLX are written using metal. 

.. note::

    Here are some helpful resources if you are new to metal!

    * A walkthrough of the metal compute pipeline: `Metal Example`_
    * Documentation for metal shading language: `Metal Specification`_
    * Using metal from C++: `Metal-cpp`_

Let's keep the GPU algorithm simple. We will launch exactly as many threads 
as there are elements in the output. Each thread will pick the element it needs 
from ``x`` and ``y``, do the pointwise operation, and then update its assigned 
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
each instantiation a unique host name so we can identify the right kernel for 
each data type. 

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

This kernel will be compiled into a metal library ``mlx_ext.metallib`` as we 
will see later in :ref:`Building with CMake`. In the following example, we 
assume that the library ``mlx_ext.metallib`` will always be co-located with 
the executable/ shared-library calling the :meth:`register_library` function. 
The :meth:`register_library` function takes the library's name and potential 
path (or in this case, a function that can produce the path of the metal 
library) and tries to load that library if it hasn't already been registered 
by the relevant static :class:`mlx::core::metal::Device` object. This is why, 
it is important to package your C++ library with the metal library. We will 
go over this process in more detail later. 

The logic to determine the kernel, set the inputs, resolve the grid dimensions 
and dispatch it to the GPU are contained in :meth:`Axpby::eval_gpu` as shown 
below.

.. code-block:: C++

    /** Evaluate primitive on GPU */
    void Axpby::eval_gpu(const std::vector<array>& inputs, array& out) {
        // Prepare inputs
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];

        // Each primitive carries the stream it should execute on
        // and each stream carries its device identifiers
        auto& s = stream();
        // We get the needed metal device using the stream
        auto& d = metal::device(s.device);

        // Allocate output memory 
        out.set_data(allocator::malloc_or_wait(out.nbytes()));

        // Resolve name of kernel (corresponds to axpby.metal)
        std::ostringstream kname;
        kname << "axpby_" << "general_" << type_to_name(out);

        // Make sure the metal library is available and look for it
        // in the same folder as this executable if needed
        d.register_library("mlx_ext", metal::get_colocated_mtllib_path);

        // Make a kernel from this metal library
        auto kernel = d.get_kernel(kname.str(), "mlx_ext");

        // Prepare to encode kernel
        auto compute_encoder = d.get_command_encoder(s.index);
        compute_encoder->setComputePipelineState(kernel);

        // Kernel parameters are registered with buffer indices corresponding to
        // those in the kernel declaration at axpby.metal
        int ndim = out.ndim();
        size_t nelem = out.size();

        // Encode input arrays to kernel
        set_array_buffer(compute_encoder, x, 0);
        set_array_buffer(compute_encoder, y, 1);

        // Encode output arrays to kernel
        set_array_buffer(compute_encoder, out, 2);

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
        compute_encoder->dispatchThreads(grid_dims, group_dims);
    }

We can now call the :meth:`axpby` operation on both the CPU and the GPU!

A few things to note about MLX and metal before moving on. MLX keeps track 
of the active ``compute_encoder``. We rely on :meth:`d.get_command_encoder` 
to give us the active metal compute command encoder instead of building a 
new one and calling :meth:`compute_encoder->end_encoding` at the end. 
MLX keeps adding kernels (compute pipelines) to the active command encoder 
until some specified limit is hit or the compute encoder needs to be flushed 
for synchronization. MLX also handles enqueuing and committing the associated 
command buffers as needed. We suggest taking a deeper dive into 
:class:`metal::Device` if you would like to study this routine further.

Primitive Transforms
^^^^^^^^^^^^^^^^^^^^^

Now that we have come this far, let's also learn how to add implementations to 
transformations in a :class:`Primitive`. These transformations can be built on 
top of our operations, including the one we just defined now. Which then gives 
us the following :meth:`Axpby::jvp` and :meth:`Axpby::vjp` implementations.

.. code-block:: C++

    /** The Jacobian-vector product. */
    array Axpby::jvp(
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
            return multiply(scale_arr, tangents[0], stream());
        }
        // If, argnums = {0, 1}, we take contributions from both
        // which gives us jvp = tangent_x * alpha + tangent_y * beta
        else {
            return axpby(tangents[0], tangents[1], alpha_, beta_, stream());
        }
    }

.. code-block:: C++

    /** The vector-Jacobian product. */
    std::vector<array> Axpby::vjp(
            const std::vector<array>& primals,
            const array& cotan,
            const std::vector<int>& argnums) {
        // Reverse mode diff
        std::vector<array> vjps;
        for (auto arg : argnums) {
            auto scale = arg == 0 ? alpha_ : beta_;
            auto scale_arr = array(scale, cotan.dtype());
            vjps.push_back(multiply(scale_arr, cotan, stream()));
        }
        return vjps;
    }

Finally, you need not have a transformation fully defined to start using your 
own :class:`Primitive`.

.. code-block:: C++

    /** Vectorize primitive along given axis */
    std::pair<array, int> Axpby::vmap(
            const std::vector<array>& inputs,
            const std::vector<int>& axes) {
        throw std::runtime_error("Axpby has no vmap implementation.");
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
  associated python package
* ``extensions/bindings.cpp`` provides python bindings for our operation
* ``extensions/CMakeLists.txt`` holds CMake rules to build the library and 
  python bindings
* ``extensions/setup.py`` holds the ``setuptools`` rules to build and install
  the python package

Binding to Python
^^^^^^^^^^^^^^^^^^

We use PyBind11_ to build a Python API for the C++ library. Since bindings for
components such as :class:`mlx.core.array`, :class:`mlx.core.stream`, etc. are
already provided, adding our :meth:`axpby` is simple!

.. code-block:: C++

    PYBIND11_MODULE(mlx_sample_extensions, m) {
        m.doc() = "Sample C++ and metal extensions for MLX";

        m.def(
            "axpby",
            &axpby,
            "x"_a,
            "y"_a,
            py::pos_only(),
            "alpha"_a,
            "beta"_a,
            py::kw_only(),
            "stream"_a = py::none(),
            R"pbdoc(
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
            )pbdoc");
    }

Most of the complexity in the above example comes from additional bells and 
whistles such as the literal names and doc-strings.

.. warning::

    :mod:`mlx.core` needs to be imported before importing 
    :mod:`mlx_sample_extensions` as defined by the pybind11 module above to 
    ensure that the casters for :mod:`mlx.core` components like 
    :class:`mlx.core.array` are available.

.. _Building with CMake:

Building with CMake
^^^^^^^^^^^^^^^^^^^^

Building the C++ extension library itself is simple, it only requires that you 
``find_package(MLX CONFIG)`` and then link it to your library. 

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

We also need to build the attached metal library. For convenience, we provide a 
:meth:`mlx_build_metallib` function that builds a ``.metallib`` target given 
sources, headers, destinations, etc. (defined in ``cmake/extension.cmake`` and 
automatically imported with MLX package). 

Here is what that looks like in practice!

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

Finally, we build the Pybind11_ bindings

.. code-block:: cmake

    pybind11_add_module(
        mlx_sample_extensions
        ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
    )
    target_link_libraries(mlx_sample_extensions PRIVATE mlx_ext)

    if(BUILD_SHARED_LIBS)
        target_link_options(mlx_sample_extensions PRIVATE -Wl,-rpath,@loader_path)
    endif()

Building with ``setuptools``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have set out the CMake build rules as described above, we can use the
build utilities defined in :mod:`mlx.extension` for a simple build process. 

.. code-block:: python 

    from mlx import extension
    from setuptools import setup

    if __name__ == "__main__":
        setup(
            name="mlx_sample_extensions",
            version="0.0.0",
            description="Sample C++ and Metal extensions for MLX primitives.",
            ext_modules=[extension.CMakeExtension("mlx_sample_extensions")],
            cmdclass={"build_ext": extension.CMakeBuild},
            packages = ["mlx_sample_extensions"],
            package_dir = {"": "mlx_sample_extensions"},
            package_data = {"mlx_sample_extensions" : ["*.so", "*.dylib", "*.metallib"]},
            zip_safe=False,
            python_requires=">=3.7",
        )

.. note::
    We treat ``extensions/mlx_sample_extensions`` as the package directory
    even though it only contains a ``__init__.py`` to ensure the following:
    
    * :mod:`mlx.core` is always imported before importing  :mod:`mlx_sample_extensions`
    * The C++ extension library and the metal library are co-located with the python 
      bindings and copied together if the package is installed 

You can build inplace for development using
``python setup.py build_ext -j8 --inplace`` (in ``extensions/``)

This will result in a directory structure as follows:

| extensions
| ├── mlx_sample_extensions
| │   ├── __init__.py
| │   ├── libmlx_ext.dylib # C++ extension library
| │   ├── mlx_ext.metallib # Metal library
| │   └── mlx_sample_extensions.cpython-3x-darwin.so # Python Binding
| ...

When you try to install using the command ``python -m pip install .`` 
(in ``extensions/``), the package will be installed with the same structure as 
``extensions/mlx_sample_extensions`` and the C++ and metal library will be 
copied along with the python binding since they are specified as ``package_data``.

Usage
-----

After installing the extension as described above, you should be able to simply 
import the python package and play with it as you would any other MLX operation!

Let's looks at a simple script and it's results!

.. code-block:: python

    import mlx.core as mx
    from mlx_sample_extensions import axpby

    a = mx.ones((3, 4))
    b = mx.ones((3, 4))
    c = axpby(a, b, 4.0, 2.0, stream=mx.cpu)

    print(f"c shape: {c.shape}")
    print(f"c dtype: {c.dtype}")
    print(f"c correctness: {mx.all(c == 6.0).item()}")

Output:

.. code-block::

    c shape: [3, 4]
    c dtype: float32
    c correctness: True

Results
^^^^^^^^^^^^^^^^

Let's run a quick benchmark and see how our new ``axpby`` operation compares 
with the naive :meth:`simple_axpby` we defined at first on the CPU. 

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

    mx.eval((x, y))

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

Results:

.. code-block::

    Simple axpby: 0.114 s | Custom axpby: 0.109 s

We see some modest improvements right away! 

This operation is now good to be used to build other operations, in
:class:`mlx.nn.Module` calls, and also as a part of graph transformations like
:meth:`grad`!

Scripts
-------

.. admonition:: Download the code

   The full example code is available in `mlx <code>`_.

.. code: `https://github.com/ml-explore/mlx/tree/main/examples/extensions/`_

.. _Accelerate: https://developer.apple.com/documentation/accelerate/blas?language=objc
.. _Metal: https://developer.apple.com/documentation/metal?language=objc
.. _Metal-cpp: https://developer.apple.com/metal/cpp/
.. _`Metal Specification`: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
.. _`Metal Example`: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc
.. _PyBind11: https://pybind11.readthedocs.io/en/stable/
