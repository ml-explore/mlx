// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
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
              If set, the ``base`` parameter must be ``None``. ``Default: None``.
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

        Args:
            q (array): Input query array.
            k (array): Input keys array.
            v (array): Input values array.
            scale (float): Scale for queries (typically ``1.0 / sqrt(q.shape(-1)``)
            mask (array, optional): An additive mask to apply to the query-key scores.
        Returns:
            array: The output array.
      )pbdoc");

  m.def(
      "affine_quantize",
      nb::overload_cast<
          const array&,
          const array&,
          const array&,
          int,
          int,
          StreamOrDevice>(&fast::affine_quantize),
      "w"_a,
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def affine_quantize(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Quantize the matrix ``w`` using the provided ``scales`` and
        ``biases`` and the ``group_size`` and ``bits`` configuration.

        Formally, given the notation in :func:`quantize`, we compute
        :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s` and
        :math:`\beta` as follows

        .. math::

          w_i = s (\hat{w_i} + \beta)

        Args:
          w (array): Matrix to be quantize
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: ``4``)

        Returns:
          array: The quantized version of ``w``
      )pbdoc");

  nb::class_<fast::MetalKernel>(
      m,
      "metal_kernel",
      R"pbdoc(
      A jit-compiled custom Metal kernel defined from a source string.
      )pbdoc")
      .def(
          nb::init<const std::string&, const std::string&, bool, bool, bool>(),
          "name"_a,
          "source"_a,
          "ensure_row_contiguous"_a = true,
          "zero_outputs"_a = false,
          "atomic_outputs"_a = false,
          R"pbdoc(
      Initialize a metal_kernel.

      Args:
        name (str): Name for the kernel.
        source (str): Source code. This is the body of a function in Metal,
            the function signature will be generated for you. The names of the inputs/outputs
            are determined by the ``inputs`` and ``output_shapes``/``output_dtypes``
            used when the kernel is called.
        ensure_row_contiguous (bool): Whether to ensure the inputs are row contiguous
            before the kernel runs. Default: ``True``.
        zero_outputs (bool): Whether to initialize the outputs to all zeros before
            the kernel runs. Default: ``False``.
        atomic_outputs (bool): Whether to use atomic outputs in the function signature
            e.g. ``device atomic<float>``. Default: ``False``.
      Returns:
        Callable ``metal_kernel``.

      .. code-block:: python

        def exp_elementwise(a: mx.array):
            source = """
                uint elem = thread_position_in_grid.x;
                T tmp = inp[elem];
                out[elem] = metal::exp(tmp);
            """

            kernel = mx.fast.metal_kernel(
                name="myexp",
                source=source
            )
            outputs = kernel(
                inputs={"inp": a},
                template={"T": mx.float32},
                grid=(a.size, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes={"out": a.shape},
                output_dtypes={"out": a.dtype},
                verbose=True,
            )
            return outputs["out"]

        a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
        b = exp_elementwise(a)
        assert mx.allclose(b, mx.exp(a))

      )pbdoc")
      .def(
          "__call__",
          [](fast::MetalKernel& kernel,
             std::map<std::string, ScalarOrArray>& inputs_,
             std::map<std::string, std::vector<int>>& output_shapes,
             std::map<std::string, Dtype>& output_dtypes,
             std::tuple<int, int, int> grid,
             std::tuple<int, int, int> threadgroup,
             std::optional<std::map<std::string, nb::handle>> template_args_,
             bool verbose,
             StreamOrDevice s) {
            std::map<std::string, array> inputs;
            for (const auto& [name, value] : inputs_) {
              auto arr = to_array(value, std::nullopt);
              inputs.insert({name, arr});
            }
            std::map<std::string, fast::TemplateArg> template_args;
            if (template_args_) {
              for (const auto& [name, value] : template_args_.value()) {
                // Handle bool, int and dtype template args
                if (nb::isinstance<bool>(value)) {
                  bool bool_val = nb::cast<bool>(value);
                  template_args.insert({name, bool_val});
                } else if (nb::isinstance<int>(value)) {
                  int int_val = nb::cast<int>(value);
                  template_args.insert({name, int_val});
                } else if (nb::isinstance<Dtype>(value)) {
                  Dtype dtype = nb::cast<Dtype>(value);
                  template_args.insert({name, dtype});
                } else {
                  throw std::invalid_argument(
                      "[[metal_kernel]] Invalid template argument. Must be `mlx.core.Dtype`, `int` or `bool`.");
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
          "verbose"_a = false,
          "stream"_a = nb::none(),
          nb::sig(
              "def __call__(self, *, inputs: Mapping[str, Union[scalar, array]], output_shapes: Mapping[str, Sequence[int]], output_dtypes: Mapping[str, Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: Optional[Mapping[str, Union[bool, int, Dtype]]] = None, verbose: bool = false, stream: Union[None, Stream, Device] = None)"),
          R"pbdoc(
            Run the kernel.

            Args:
              inputs (Mapping[str, array]): Inputs. These will be added to the function signature and passed to the Metal kernel.
                  The keys will be the names of the arguments to the kernel.
              output_shapes (Mapping[str, Sequence[int]]): Output shapes. A dict mapping
                  output variable names to shapes. These will be added to the function signature.
              output_dtypes (Mapping[str, Dtype]): Output dtypes. A dict mapping output variable
                  names to dtypes. Must have the same keys as ``output_shapes``.
              grid (tuple[int, int, int]): 3-tuple specifying the grid to launch the kernel with.
              threadgroup (tuple[int, int, int]): 3-tuple specifying the threadgroup size to use.
              template (Mapping[str, Union[bool, int, Dtype]], optional): Template arguments.
                  These will be added as template arguments to the kernel definition.
              verbose (bool, optional): Whether to print the full generated source code of the kernel
                  when it is run.
              stream (mx.stream, optional): Stream to run the kernel on. Default: ``None``.

            Returns:
              dict[str, array]: Dictionary of output arrays based on ``output_shapes``/``output_dtypes``.
            )pbdoc");
}
