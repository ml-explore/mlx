// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct PyCustomKernelFunction {
  PyCustomKernelFunction(mx::fast::CustomKernelFunction kernel, const char* tag)
      : kernel_(std::move(kernel)), tag_(tag) {}

  std::vector<mx::array> operator()(
      const std::vector<ScalarOrArray>& inputs_,
      const std::vector<mx::Shape>& output_shapes,
      const std::vector<mx::Dtype>& output_dtypes,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      const std::optional<std::vector<std::pair<std::string, nb::object>>>&
          template_args_ = std::nullopt,
      std::optional<float> init_value = std::nullopt,
      bool verbose = false,
      mx::StreamOrDevice s = {}) const {
    std::vector<mx::array> inputs;
    for (const auto& value : inputs_) {
      inputs.push_back(to_array(value, std::nullopt));
    }
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
    if (template_args_) {
      for (const auto& [name, value] : template_args_.value()) {
        // Handle bool, int and dtype template args
        if (nb::isinstance<bool>(value)) {
          bool bool_val = nb::cast<bool>(value);
          template_args.emplace_back(name, bool_val);
        } else if (nb::isinstance<int>(value)) {
          int int_val = nb::cast<int>(value);
          template_args.emplace_back(name, int_val);
        } else if (nb::isinstance<mx::Dtype>(value)) {
          mx::Dtype dtype = nb::cast<mx::Dtype>(value);
          template_args.emplace_back(name, dtype);
        } else {
          std::ostringstream msg;
          msg << tag_
              << " Invalid template argument. Must be `mlx.core.Dtype`, `int` or `bool`.";
          throw std::invalid_argument(msg.str());
        }
      }
    }
    return kernel_(
        inputs,
        output_shapes,
        output_dtypes,
        grid,
        threadgroup,
        template_args,
        init_value,
        verbose,
        s);
  }

  mx::fast::CustomKernelFunction kernel_;
  const char* tag_;
};

} // namespace

void init_fast(nb::module_& parent_module) {
  auto m =
      parent_module.def_submodule("fast", "mlx.core.fast: fast operations");

  m.def(
      "rms_norm",
      &mx::fast::rms_norm,
      "x"_a,
      "weight"_a.none(),
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rms_norm(x: array, weight: Optional[array], eps: float, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Root Mean Square normalization (RMS norm).

        The normalization is with respect to the last axis of the input ``x``.

        Args:
            x (array): Input array.
            weight (array, optional): A multiplicative weight to scale the result by.
              The ``weight`` should be one-dimensional with the same size
              as the last axis of ``x``. If set to ``None`` then no scaling happens.
            eps (float): A small additive constant for numerical stability.

        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "layer_norm",
      &mx::fast::layer_norm,
      "x"_a,
      "weight"_a.none(),
      "bias"_a.none(),
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def layer_norm(x: array, weight: Optional[array], bias: Optional[array], eps: float, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Layer normalization.

        The normalization is with respect to the last axis of the input ``x``.

        Args:
            x (array): Input array.
            weight (array, optional): A multiplicative weight to scale the result by.
              The ``weight`` should be one-dimensional with the same size
              as the last axis of ``x``. If set to ``None`` then no scaling happens.
            bias (array, optional): An additive offset to be added to the result.
              The ``bias`` should be one-dimensional with the same size
              as the last axis of ``x``. If set to ``None`` then no translation happens.
            eps (float): A small additive constant for numerical stability.

        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "rope",
      [](const mx::array& a,
         int dims,
         bool traditional,
         std::optional<float> base,
         float scale,
         const ScalarOrArray& offset,
         const std::optional<mx::array>& freqs /* = std::nullopt */,
         mx::StreamOrDevice s /* = {} */) {
        return mx::fast::rope(
            a, dims, traditional, base, scale, to_array(offset), freqs, s);
      },
      "a"_a,
      "dims"_a,
      nb::kw_only(),
      "traditional"_a,
      "base"_a.none(),
      "scale"_a,
      "offset"_a,
      "freqs"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rope(a: array, dims: int, *, traditional: bool, base: Optional[float], scale: float, offset: Union[int, array], freqs: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Apply rotary positional encoding to the input.

        Args:
            a (array): Input array.
            dims (int): The feature dimensions to be rotated. If the input feature
              is larger than dims then the rest is left unchanged.
            traditional (bool): If set to ``True`` choose the traditional
              implementation which rotates consecutive dimensions.
            base (float, optional): The base used to compute angular frequency for
              each dimension in the positional encodings. Exactly one of ``base`` and
              ``freqs`` must be ``None``.
            scale (float): The scale used to scale the positions.
            offset (int or array): The position offset to start at.
            freqs (array, optional): Optional frequencies to use with RoPE.
              If set, the ``base`` parameter must be ``None``. Default: ``None``.

        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "scaled_dot_product_attention",
      [](const mx::array& queries,
         const mx::array& keys,
         const mx::array& values,
         const float scale,
         const std::variant<std::monostate, std::string, mx::array>& mask,
         mx::StreamOrDevice s) {
        bool has_mask = !std::holds_alternative<std::monostate>(mask);
        bool has_str_mask =
            has_mask && std::holds_alternative<std::string>(mask);
        bool has_arr_mask = has_mask && std::holds_alternative<mx::array>(mask);

        if (has_mask) {
          if (has_str_mask) {
            auto mask_str = std::get<std::string>(mask);
            if (mask_str != "causal") {
              std::ostringstream msg;
              msg << "[scaled_dot_product_attention] invalid mask option '"
                  << mask_str << "'. Must be 'causal', or an array.";
              throw std::invalid_argument(msg.str());
            }
            return mx::fast::scaled_dot_product_attention(
                queries, keys, values, scale, mask_str, {}, s);
          } else {
            auto mask_arr = std::get<mx::array>(mask);
            return mx::fast::scaled_dot_product_attention(
                queries, keys, values, scale, "", {mask_arr}, s);
          }

        } else {
          return mx::fast::scaled_dot_product_attention(
              queries, keys, values, scale, "", {}, s);
        }
      },
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "scale"_a,
      "mask"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float,  mask: Union[None, str, array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        A fast implementation of multi-head attention: ``O = softmax(Q @ K.T, dim=-1) @ V``.

        Supports:

        * `Multi-Head Attention <https://arxiv.org/abs/1706.03762>`_
        * `Grouped Query Attention <https://arxiv.org/abs/2305.13245>`_
        * `Multi-Query Attention <https://arxiv.org/abs/1911.02150>`_

        .. note::

          * The softmax operation is performed in ``float32`` regardless of
            the input precision.
          * For Grouped Query Attention and Multi-Query Attention, the ``k``
            and ``v`` inputs should not be pre-tiled to match ``q``.

        In the following the dimensions are given by:

        * ``B``: The batch size.
        * ``N_q``: The number of query heads.
        * ``N_kv``: The number of key and value heads.
        * ``T_q``: The number of queries per example.
        * ``T_kv``: The number of keys and values per example.
        * ``D``: The per-head dimension.

        Args:
            q (array): Queries with shape ``[B, N_q, T_q, D]``.
            k (array): Keys with shape ``[B, N_kv, T_kv, D]``.
            v (array): Values with shape ``[B, N_kv, T_kv, D]``.
            scale (float): Scale for queries (typically ``1.0 / sqrt(q.shape(-1)``)
            mask (Union[None, str, array], optional): The mask to apply to the
               query-key scores. The mask can be an array or a string indicating
               the mask type. The only supported string type is ``"causal"``. If
               the mask is an array it can be a boolean or additive mask. The mask
               can have at most 4 dimensions and must be broadcast-compatible with
               the shape ``[B, N, T_q, T_kv]``. If an additive mask is given its
               type must promote to the promoted type of ``q``, ``k``, and ``v``.
        Returns:
            array: The output array.

        Example:

          .. code-block:: python

            B = 2
            N_q = N_kv = 32
            T_q = T_kv = 1000
            D = 128

            q = mx.random.normal(shape=(B, N_q, T_q, D))
            k = mx.random.normal(shape=(B, N_kv, T_kv, D))
            v = mx.random.normal(shape=(B, N_kv, T_kv, D))
            scale = D ** -0.5
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
      )pbdoc");

  m.def(
      "metal_kernel",
      [](const std::string& name,
         const std::vector<std::string>& input_names,
         const std::vector<std::string>& output_names,
         const std::string& source,
         const std::string& header,
         bool ensure_row_contiguous,
         bool atomic_outputs) {
        auto kernel = mx::fast::metal_kernel(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            atomic_outputs);
        return nb::cpp_function(
            PyCustomKernelFunction(std::move(kernel), "[metal_kernel]"),
            nb::kw_only(),
            "inputs"_a,
            "output_shapes"_a,
            "output_dtypes"_a,
            "grid"_a,
            "threadgroup"_a,
            "template"_a = nb::none(),
            "init_value"_a = nb::none(),
            "verbose"_a = false,
            "stream"_a = nb::none(),
            nb::sig(
                "def __call__(self, *, inputs: List[Union[scalar, array]], output_shapes: List[Sequence[int]], output_dtypes: List[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: Optional[List[Tuple[str, Union[bool, int, Dtype]]]] = None, init_value: Optional[float] = None, verbose: bool = false, stream: Union[None, Stream, Device] = None)"),
            R"pbdoc(
            Run the kernel.

            Args:
              inputs (List[array]): The inputs passed to the Metal kernel.
              output_shapes (List[Sequence[int]]): The list of shapes for each output in ``output_names``.
              output_dtypes (List[Dtype]): The list of data types for each output in ``output_names``.
              grid (tuple[int, int, int]): 3-tuple specifying the grid to launch the kernel with.
                This will be passed to ``MTLComputeCommandEncoder::dispatchThreads``.
              threadgroup (tuple[int, int, int]): 3-tuple specifying the threadgroup size to use.
                This will be passed to ``MTLComputeCommandEncoder::dispatchThreads``.
              template (List[Tuple[str, Union[bool, int, Dtype]]], optional): Template arguments.
                  These will be added as template arguments to the kernel definition. Default: ``None``.
              init_value (float, optional): Optional value to use to initialize all of the output arrays.
                  By default, output arrays are uninitialized. Default: ``None``.
              verbose (bool, optional): Whether to print the full generated source code of the kernel
                  when it is run. Default: ``False``.
              stream (mx.stream, optional): Stream to run the kernel on. Default: ``None``.

            Returns:
              List[array]: The list of output arrays.)pbdoc");
      },
      "name"_a,
      "input_names"_a,
      "output_names"_a,
      "source"_a,
      "header"_a = "",
      "ensure_row_contiguous"_a = true,
      "atomic_outputs"_a = false,
      R"pbdoc(
      A jit-compiled custom Metal kernel defined from a source string.

      Full documentation: :ref:`custom_metal_kernels`.

      Args:
        name (str): Name for the kernel.
        input_names (List[str]): The parameter names of the inputs in the
           function signature.
        output_names (List[str]): The parameter names of the outputs in the
           function signature.
        source (str): Source code. This is the body of a function in Metal,
           the function signature will be automatically generated.
        header (str): Header source code to include before the main function.
           Useful for helper functions or includes that should live outside of
           the main function body.
        ensure_row_contiguous (bool): Whether to ensure the inputs are row contiguous
           before the kernel runs. Default: ``True``.
        atomic_outputs (bool): Whether to use atomic outputs in the function signature
           e.g. ``device atomic<float>``. Default: ``False``.

      Returns:
        Callable ``metal_kernel``.

      Example:

        .. code-block:: python

          def exp_elementwise(a: mx.array):
              source = '''
                  uint elem = thread_position_in_grid.x;
                  T tmp = inp[elem];
                  out[elem] = metal::exp(tmp);
              '''

              kernel = mx.fast.metal_kernel(
                  name="myexp",
                  input_names=["inp"],
                  output_names=["out"],
                  source=source
              )
              outputs = kernel(
                  inputs=[a],
                  template=[("T", mx.float32)],
                  grid=(a.size, 1, 1),
                  threadgroup=(256, 1, 1),
                  output_shapes=[a.shape],
                  output_dtypes=[a.dtype],
                  verbose=True,
              )
              return outputs[0]

          a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
          b = exp_elementwise(a)
          assert mx.allclose(b, mx.exp(a))
     )pbdoc");

  m.def(
      "cuda_kernel",
      [](const std::string& name,
         const std::vector<std::string>& input_names,
         const std::vector<std::string>& output_names,
         const std::string& source,
         const std::string& header,
         bool ensure_row_contiguous,
         int shared_mem) {
        auto kernel = mx::fast::cuda_kernel(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            shared_mem);
        return nb::cpp_function(
            PyCustomKernelFunction(std::move(kernel), "[cuda_kernel]"),
            nb::kw_only(),
            "inputs"_a,
            "output_shapes"_a,
            "output_dtypes"_a,
            "grid"_a,
            "threadgroup"_a,
            "template"_a = nb::none(),
            "init_value"_a = nb::none(),
            "verbose"_a = false,
            "stream"_a = nb::none(),
            nb::sig(
                "def __call__(self, *, inputs: List[Union[scalar, array]], output_shapes: List[Sequence[int]], output_dtypes: List[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: Optional[List[Tuple[str, Union[bool, int, Dtype]]]] = None, init_value: Optional[float] = None, verbose: bool = false, stream: Union[None, Stream, Device] = None)"),
            R"pbdoc(
            Run the kernel.

            Args:
              inputs (List[array]): The inputs passed to the CUDA kernel.
              output_shapes (List[Sequence[int]]): The list of shapes for each output in ``output_names``.
              output_dtypes (List[Dtype]): The list of data types for each output in ``output_names``.
              grid (tuple[int, int, int]): 3-tuple specifying the grid to launch the kernel with.
                For compatibility with :func:`metal_kernel` the grid is in threads and not in threadgroups.
              threadgroup (tuple[int, int, int]): 3-tuple specifying the threadgroup size to use.
              template (List[Tuple[str, Union[bool, int, Dtype]]], optional): Template arguments.
                  These will be added as template arguments to the kernel definition. Default: ``None``.
              init_value (float, optional): Optional value to use to initialize all of the output arrays.
                  By default, output arrays are uninitialized. Default: ``None``.
              verbose (bool, optional): Whether to print the full generated source code of the kernel
                  when it is run. Default: ``False``.
              stream (mx.stream, optional): Stream to run the kernel on. Default: ``None``.

            Returns:
              List[array]: The list of output arrays.)pbdoc");
      },
      "name"_a,
      "input_names"_a,
      "output_names"_a,
      "source"_a,
      "header"_a = "",
      "ensure_row_contiguous"_a = true,
      "shared_memory"_a = 0,
      R"pbdoc(
      A jit-compiled custom CUDA kernel defined from a source string.

      This is the CUDA equivalent of :ref:`custom_metal_kernels`.

      Args:
        name (str): Name for the kernel.
        input_names (List[str]): The parameter names of the inputs in the
           function signature.
        output_names (List[str]): The parameter names of the outputs in the
           function signature.
        source (str): Source code. This is the body of a function in CUDA,
           the function signature will be automatically generated.
        header (str): Header source code to include before the main function.
           Useful for helper functions or includes that should live outside of
           the main function body.
        ensure_row_contiguous (bool): Whether to ensure the inputs are row contiguous
           before the kernel runs. Default: ``True``.
        shared_memory (int): The dynamic shared memory to request for the
          kernel. A value of 0 means no dynamic shared memory. Default: ``0``.

      Returns:
        Callable ``cuda_kernel``.

      Example:

        .. code-block:: python

          def exp_elementwise(a: mx.array):
              source = '''
                  auto elem = cooperative_groups::this_grid().thread_rank();
                  T tmp = inp[elem];
                  out[elem] = exp(tmp);
              '''

              kernel = mx.fast.cuda_kernel(
                  name="myexp",
                  input_names=["inp"],
                  output_names=["out"],
                  source=source
              )
              outputs = kernel(
                  inputs=[a],
                  template=[("T", mx.float32)],
                  grid=(a.size, 1, 1),
                  threadgroup=(256, 1, 1),
                  output_shapes=[a.shape],
                  output_dtypes=[a.dtype],
                  verbose=True,
              )
              return outputs[0]

          a = mx.random.normal(shape=(16, 16)).astype(mx.float16)
          b = exp_elementwise(a)
          assert mx.allclose(b, mx.exp(a))
     )pbdoc");

  m.def(
      "precompiled_cuda_kernel",
      [](const std::string& name,
         const nb::bytes compiled_source,
         const std::vector<ScalarOrArray>& inputs_,
         const std::vector<mx::Shape>& output_shapes,
         const std::vector<mx::Dtype>& output_dtypes,
         const std::vector<nb::object>& scalars_,
         std::tuple<int, int, int> grid,
         std::tuple<int, int, int> threadgroup,
         int shared_memory,
         std::optional<float> init_value = std::nullopt,
         bool ensure_row_contiguous = false,
         mx::StreamOrDevice s = {}) {
        // Collect the inputs and cast them to array
        std::vector<mx::array> inputs;
        for (const auto& value : inputs_) {
          inputs.push_back(to_array(value, std::nullopt));
        }

        // Collect the scalar inputs
        std::vector<mx::fast::ScalarArg> scalars;
        scalars.reserve(scalars_.size());
        for (const auto& v : scalars_) {
          if (nb::isinstance<bool>(v)) {
            scalars.push_back(nb::cast<bool>(v));
          } else if (nb::isinstance<int>(v)) {
            scalars.push_back(nb::cast<int>(v));
          } else if (nb::isinstance<float>(v)) {
            scalars.push_back(nb::cast<float>(v));
          } else {
            nb::object vtype = v.attr("__class__");
            std::string vtype_name =
                nb::cast<std::string>(vtype.attr("__name__"));
            std::ostringstream msg;
            msg << "[precompiled_cuda_kernel] Invalid scalar argument type. "
                << "Received " << vtype_name
                << " but must be one of bool, int or float";
            throw std::invalid_argument(msg.str());
          }
        }

        return mx::fast::precompiled_cuda_kernel(
            name,
            std::string(
                static_cast<const char*>(compiled_source.data()),
                compiled_source.size()),
            inputs,
            output_shapes,
            output_dtypes,
            scalars,
            grid,
            threadgroup,
            shared_memory,
            init_value,
            ensure_row_contiguous,
            s);
      },
      nb::kw_only(),
      "name"_a,
      "compiled_source"_a,
      "inputs"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "scalars"_a,
      "grid"_a,
      "threadgroup"_a,
      "shared_memory"_a = 0,
      "init_value"_a = nb::none(),
      "ensure_row_contiguous"_a = false,
      "stream"_a = nb::none(),
      R"pbdoc(
      Run a precompiled CUDA kernel defined from PTX or cubin.

      This op is still experimental and various parts of the API may change.

      Args:
        name (str): Name for the kernel
        compiled_source (bytes): The precompiled kernel in raw bytes.
        inputs (List[array]): The inputs passed to the CUDA kernel.
        output_shapes (List[Sequence[int]]): The list of shapes for each output.
        output_dtypes (List[Dtype]): The list of data types for each output.
        scalars (List[Union[bool, int, float]]): A list of scalar arguments to
          pass to the kernel.
        grid (tuple[int, int, int]): 3-tuple specifying the grid to launch the kernel with.
          For compatibility with :func:`metal_kernel` the grid is in threads and not in threadblocks.
        threadgroup (tuple[int, int, int]): 3-tuple specifying the threadgroup size to use.
        shared_memory (int): The dynamic shared memory to request for the
          kernel. A value of 0 means no dynamic shared memory. Default: ``0``.
        init_value (float, optional): Optional value to use to initialize all of the output arrays.
            By default, output arrays are uninitialized. Default: ``None``.
        ensure_row_contiguous (bool): Whether to ensure the inputs are row contiguous
           before the kernel runs. Default: ``False``.
        stream (mx.stream, optional): Stream to run the kernel on. Default: ``None``.
      )pbdoc");
}
