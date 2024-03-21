// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
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
      [](const array& x,
         const array& weight,
         float eps,
         const StreamOrDevice& s /* = {} */) {
        return fast::rms_norm(x, weight, eps, s);
      },
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
      "rope",
      [](const array& a,
         int dims,
         bool traditional,
         float base,
         float scale,
         int offset,
         const StreamOrDevice& s /* = {} */) {
        return fast::rope(a, dims, traditional, base, scale, offset, s);
      },
      "a"_a,
      "dims"_a,
      nb::kw_only(),
      "traditional"_a,
      "base"_a,
      "scale"_a,
      "offset"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def rope(a: array, dims: int, *, traditinoal: bool, base: float, scale: float, offset: int, stream: Union[None, Stream, Device] = None) -> array"),
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
      [](const array& q,
         const array& k,
         const array& v,
         const float scale,
         const std::optional<array>& mask,
         const StreamOrDevice& s) {
        return fast::scaled_dot_product_attention(q, k, v, scale, mask, s);
      },
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
        * [Multi-Head Attention](https://arxiv.org/abs/1706.03762)
        * [Grouped Query Attention](https://arxiv.org/abs/2305.13245)
        * [Multi-Query Attention](https://arxiv.org/abs/1911.02150).

        Note: The softmax operation is performed in ``float32`` regardless of
        input precision.

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
}
