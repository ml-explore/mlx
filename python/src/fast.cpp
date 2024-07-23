// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

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
      "base"_a,
      "scale"_a,
      "offset"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def rope(a: array, dims: int, *, traditional: bool, base: float, scale: float, offset: int, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Apply rotary positional encoding to the input.

        Args:
            a (array): Input array.
            dims (int): The feature dimensions to be rotated. If the input feature
                is larger than dims then the rest is left unchanged.
            traditional (bool): If set to ``True`` choose the traditional
                implementation which rotates consecutive dimensions.
            base (float): The base used to compute angular frequency for
                each dimension in the positional encodings.
            scale (float): The scale used to scale the positions.
            offset (int): The position offset to start at.

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
      "stream"_a = nb::none(),
      nb::sig(
          "def scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float,  mask: Union[None, array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
      &fast::affine_quantize,
      "w"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def affine_quantize(w: array, /, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]"),
      R"pbdoc(
        Quantize the matrix ``w`` using ``bits`` bits per element.

        Note, every ``group_size`` elements in a row of ``w`` are quantized
        together. Hence, number of columns of ``w`` should be divisible by
        ``group_size``. In particular, the rows of ``w`` are divided into groups of
        size ``group_size`` which are quantized together.

        .. warning::

          ``quantize`` currently only supports 2D inputs with dimensions which are multiples of 32

        Formally, for a group of :math:`g` consecutive elements :math:`w_1` to
        :math:`w_g` in a row of ``w`` we compute the quantized representation
        of each element :math:`\hat{w_i}` as follows

        .. math::

          \begin{aligned}
            \alpha &= \max_i w_i \\
            \beta &= \min_i w_i \\
            s &= \frac{\alpha - \beta}{2^b - 1} \\
            \hat{w_i} &= \textrm{round}\left( \frac{w_i - \beta}{s}\right).
          \end{aligned}

        After the above computation, :math:`\hat{w_i}` fits in :math:`b` bits
        and is packed in an unsigned 32-bit integer from the lower to upper
        bits. For instance, for 4-bit quantization we fit 8 elements in an
        unsigned 32 bit integer where the 1st element occupies the 4 least
        significant bits, the 2nd bits 4-7 etc.

        In order to be able to dequantize the elements of ``w`` we also need to
        save :math:`s` and :math:`\beta` which are the returned ``scales`` and
        ``biases`` respectively.

        Args:
          w (array): Matrix to be quantized
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the returned quantized matrix. (default: ``4``)

        Returns:
          tuple: A tuple containing

          * w_q (array): The quantized version of ``w``
          * scales (array): The scale to multiply each element with, namely :math:`s`
          * biases (array): The biases to add to each element, namely :math:`\beta`
      )pbdoc");

  m.def(
      "affine_quantize_with_params",
      &fast::affine_quantize_with_params,
      "w"_a,
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def affine_quantize_with_params(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
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

  m.def(
      "affine_dequantize",
      &fast::affine_dequantize,
      "w"_a,
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def affine_dequantize(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Dequantize the matrix ``w`` using the provided ``scales`` and
        ``biases`` and the ``group_size`` and ``bits`` configuration.

        Formally, given the notation in :func:`quantize`, we compute
        :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s` and
        :math:`\beta` as follows

        .. math::

          w_i = s \hat{w_i} - \beta

        Args:
          w (array): Matrix to be quantized
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: ``4``)

        Returns:
          array: The dequantized version of ``w``
      )pbdoc");
}
