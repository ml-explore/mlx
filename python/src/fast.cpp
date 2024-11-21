// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "python/src/utils.h"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_fast(nb::module_& parent_module) {
  auto m =
      parent_module.def_submodule("fast", "mlx.core.fast: fast operations");

  m.def(
      "rms_norm",
      &fast::rms_norm,
      "x"_a,
      "weight"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rms_norm(x: array, weight: array, eps: float, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Root Mean Square normalization (RMS norm).

        The normalization is with respect to the last axis of the input ``x``.

        Args:
            x (array): Input array.
            weight (array): A multiplicative weight to scale the result by.
              The ``weight`` should be one-dimensional with the same size
              as the last axis of ``x``.
            eps (float): A small additive constant for numerical stability.

        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "layer_norm",
      &fast::layer_norm,
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
      &fast::rope,
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
          "def rope(a: array, dims: int, *, traditional: bool, base: Optional[float], scale: float, offset: int, freqs: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
            offset (int): The position offset to start at.
            freqs (array, optional): Optional frequencies to use with RoPE.
              If set, the ``base`` parameter must be ``None``. Default: ``None``.

        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "scaled_dot_product_attention",
      &fast::scaled_dot_product_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "scale"_a,
      "mask"_a = nb::none(),
      "memory_efficient_threshold"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float,  mask: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        A fast implementation of multi-head attention: ``O = softmax(Q @ K.T, dim=-1) @ V``.

        Supports:

        * `Multi-Head Attention <https://arxiv.org/abs/1706.03762>`_
        * `Grouped Query Attention <https://arxiv.org/abs/2305.13245>`_
        * `Multi-Query Attention <https://arxiv.org/abs/1911.02150>`_

        Note: The softmax operation is performed in ``float32`` regardless of
        the input precision.

        Note: For Grouped Query Attention and Multi-Query Attention, the ``k``
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
            mask (array, optional): An additive mask to apply to the query-key
               scores. The mask can have at most 4 dimensions and must be
               broadcast-compatible with the shape ``[B, N, T_q, T_kv]``.
        Returns:
            array: The output array.
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
        auto kernel = fast::metal_kernel(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            atomic_outputs);
        return nb::cpp_function(
            [kernel = std::move(kernel)](
                const std::vector<ScalarOrArray>& inputs_,
                const std::vector<std::vector<int>>& output_shapes,
                const std::vector<Dtype>& output_dtypes,
                std::tuple<int, int, int> grid,
                std::tuple<int, int, int> threadgroup,
                const std::optional<
                    std::vector<std::pair<std::string, nb::object>>>&
                    template_args_ = std::nullopt,
                std::optional<float> init_value = std::nullopt,
                bool verbose = false,
                StreamOrDevice s = {}) {
              std::vector<array> inputs;
              for (const auto& value : inputs_) {
                inputs.push_back(to_array(value, std::nullopt));
              }
              std::vector<std::pair<std::string, fast::TemplateArg>>
                  template_args;
              if (template_args_) {
                for (const auto& [name, value] : template_args_.value()) {
                  // Handle bool, int and dtype template args
                  if (nb::isinstance<bool>(value)) {
                    bool bool_val = nb::cast<bool>(value);
                    template_args.emplace_back(name, bool_val);
                  } else if (nb::isinstance<int>(value)) {
                    int int_val = nb::cast<int>(value);
                    template_args.emplace_back(name, int_val);
                  } else if (nb::isinstance<Dtype>(value)) {
                    Dtype dtype = nb::cast<Dtype>(value);
                    template_args.emplace_back(name, dtype);
                  } else {
                    throw std::invalid_argument(
                        "[metal_kernel] Invalid template argument. Must be `mlx.core.Dtype`, `int` or `bool`.");
                  }
                }
              }
              return kernel(
                  inputs,
                  output_shapes,
                  output_dtypes,
                  grid,
                  threadgroup,
                  template_args,
                  init_value,
                  verbose,
                  s);
            },
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
}
