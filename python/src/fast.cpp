// Copyright © 2023-2024 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_extensions(py::module_& parent_module) {
  py::options options;
  options.disable_function_signatures();

  auto m =
      parent_module.def_submodule("fast", "mlx.core.fast: fast operations");

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
      py::kw_only(),
      "traditional"_a,
      "base"_a,
      "scale"_a,
      "offset"_a,
      "stream"_a = none,
      R"pbdoc(
        rope(a: array, dims: int, *, traditinoal: bool, base: float, scale: float, offset: int, stream: Union[None, Stream, Device] = None) -> array

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
      py::kw_only(),
      "scale"_a,
      "mask"_a = none,
      "stream"_a = none,
      R"pbdoc(
                  scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float, /,  mask: Union[None, array] = None, stream: Union[None, Stream, Device] = None) -> array

            A parallelized implementation of multi-head attention: O = softmax(Q @ K.T, dim=-1) @ V.
            Supports Multi-Head Attention (see https://arxiv.org/abs/1706.03762),
            Grouped Query Attention (https://arxiv.org/abs/2305.13245),
            and Multi-Query Attention (https://arxiv.org/pdf/1911.02150.pdf).

            This function is an inference-focused kernel optimized specifically for KV-cached transformer
            decoder inference (query sequence length = 1) and large KV-cached sequences.
            It handles prompt encoding via MLX primitives.  The optimized metal kernel is for decoding only.

            Args:
                q (array): Input query array.
                k (array): Input keys array.
                v (array): Input values array.
                scale (float): Scale for queries (typically 1.0 / sqrt(q.shape(-1))
                mask (array, optional): Mask for prompt encoding

            Returns:
                array: The output array.

          )pbdoc");
}
