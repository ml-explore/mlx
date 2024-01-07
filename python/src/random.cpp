// Copyright © 2023 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/random.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;
using namespace mlx::core::random;

void init_random(py::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "random",
      "mlx.core.random: functionality related to random number generation");
  m.def(
      "seed",
      &seed,
      "seed"_a,
      R"pbdoc(
        Seed the global PRNG.

        Args:
            seed (int): Seed for the global PRNG.
      )pbdoc");
  m.def(
      "key",
      &key,
      "seed"_a,
      R"pbdoc(
        Get a PRNG key from a seed.

        Args:
            seed (int): Seed for the PRNG.

        Returns:
            array: The PRNG key array.
      )pbdoc");
  m.def(
      "split",
      py::overload_cast<const array&, int, StreamOrDevice>(&random::split),
      "key"_a,
      "num"_a = 2,
      "stream"_a = none,
      R"pbdoc(
        Split a PRNG key into sub keys.

        Args:
            key (array): Input key to split.
            num (int, optional): Number of sub keys. Default is 2.

        Returns:
            array: The array of sub keys with ``num`` as its first dimension.
      )pbdoc");
  m.def(
      "uniform",
      [](const ScalarOrArray& low,
         const ScalarOrArray& high,
         const std::vector<int>& shape,
         std::optional<Dtype> type,
         const std::optional<array>& key,
         StreamOrDevice s) {
        return uniform(
            to_array(low),
            to_array(high),
            shape,
            type.value_or(float32),
            key,
            s);
      },
      "low"_a = 0,
      "high"_a = 1,
      "shape"_a = std::vector<int>{},
      "dtype"_a = std::optional{float32},
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Generate uniformly distributed random numbers.

        The values are sampled uniformly in the half-open interval ``[low, high)``.
        The lower and upper bound can be scalars or arrays and must be
        broadcastable to ``shape``.

        Args:
            low (scalar or array, optional): Lower bound of the distribution. Default is ``0``.
            high (scalar or array, optional): Upper bound of the distribution. Default is ``1``.
            shape (list(int), optional): Shape of the output. Default is ``()``.
            key (array, optional): A PRNG key. Default: None.
            dtype (Dtype, optional): Type of the output. Default is ``float32``.

        Returns:
            array: The output array random values.
      )pbdoc");
  m.def(
      "normal",
      [](const std::vector<int>& shape,
         std::optional<Dtype> type,
         const std::optional<array>& key,
         StreamOrDevice s) {
        return normal(shape, type.value_or(float32), key, s);
      },

      "shape"_a = std::vector<int>{},
      "dtype"_a = std::optional{float32},
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Generate normally distributed random numbers.

        Args:
            shape (list(int), optional): Shape of the output. Default is ``()``.
            dtype (Dtype, optional): Type of the output. Default is ``float32``.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "randint",
      [](const ScalarOrArray& low,
         const ScalarOrArray& high,
         const std::vector<int>& shape,
         std::optional<Dtype> type,
         const std::optional<array>& key,
         StreamOrDevice s) {
        return randint(
            to_array(low), to_array(high), shape, type.value_or(int32), key, s);
      },
      "low"_a,
      "high"_a,
      "shape"_a = std::vector<int>{},
      "dtype"_a = int32,
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Generate random integers from the given interval.

        The values are sampled with equal probability from the integers in
        half-open interval ``[low, high)``. The lower and upper bound can be
        scalars or arrays and must be roadcastable to ``shape``.

        Args:
            low (scalar or array): Lower bound of the interval.
            high (scalar or array): Upper bound of the interval.
            shape (list(int), optional): Shape of the output. Defaults to ``()``.
            dtype (Dtype, optional): Type of the output. Defaults to ``int32``.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The array of random integers.
      )pbdoc");
  m.def(
      "bernoulli",
      [](const ScalarOrArray& p_,
         const std::optional<std::vector<int>> shape,
         const std::optional<array>& key,
         StreamOrDevice s) {
        auto p = to_array(p_);
        if (shape.has_value()) {
          return bernoulli(p, shape.value(), key, s);
        } else {
          return bernoulli(p, key, s);
        }
      },
      "p"_a = 0.5,
      "shape"_a = none,
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Generate Bernoulli random values.

        The values are sampled from the bernoulli distribution with parameter
        ``p``. The parameter ``p`` can be a :obj:`float` or :obj:`array` and
        must be broadcastable to ``shape``.

        Args:
            p (float or array, optional): Parameter of the Bernoulli
              distribution. Default is 0.5.
            shape (list(int), optional): Shape of the output. The default
              shape is ``p.shape``.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The array of random integers.
      )pbdoc");
  m.def(
      "truncated_normal",
      [](const ScalarOrArray& lower_,
         const ScalarOrArray& upper_,
         const std::optional<std::vector<int>> shape_,
         std::optional<Dtype> type,
         const std::optional<array>& key,
         StreamOrDevice s) {
        auto lower = to_array(lower_);
        auto upper = to_array(upper_);
        auto t = type.value_or(float32);
        if (shape_.has_value()) {
          return truncated_normal(lower, upper, shape_.value(), t, key, s);
        } else {
          return truncated_normal(lower, upper, t, key, s);
        }
      },
      "lower"_a,
      "upper"_a,
      "shape"_a = none,
      "dtype"_a = std::optional{float32},
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Generate values from a truncated normal distribution.

        The values are sampled from the truncated normal distribution
        on the domain ``(lower, upper)``. The bounds ``lower`` and ``upper``
        can be scalars or arrays and must be broadcastable to ``shape``.

        Args:
            lower (scalar or array): Lower bound of the domain.
            upper (scalar or array): Upper bound of the domain.
            shape (list(int), optional): The shape of the output.
              Default is ``()``.
            dtype (Dtype, optional): The data type of the output.
              Default is ``float32``.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "gumbel",
      [](const std::vector<int>& shape,
         std::optional<Dtype> type,
         const std::optional<array>& key,
         StreamOrDevice s) {
        return gumbel(shape, type.value_or(float32), key, s);
      },
      "shape"_a = std::vector<int>{},
      "dtype"_a = std::optional{float32},
      "stream"_a = none,
      "key"_a = none,
      R"pbdoc(
        Sample from the standard Gumbel distribution.

        The values are sampled from a standard Gumbel distribution
        which CDF ``exp(-exp(-x))``.

        Args:
            shape (list(int)): The shape of the output.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The :class:`array` with shape ``shape`` and
                   distributed according to the Gumbel distribution
      )pbdoc");
  m.def(
      "categorical",
      [](const array& logits,
         int axis,
         const std::optional<std::vector<int>> shape,
         const std::optional<int> num_samples,
         const std::optional<array>& key,
         StreamOrDevice s) {
        if (shape.has_value() && num_samples.has_value()) {
          throw std::invalid_argument(
              "[categorical] At most one of shape or num_samples can be specified.");
        } else if (shape.has_value()) {
          return categorical(logits, axis, shape.value(), key, s);
        } else if (num_samples.has_value()) {
          return categorical(logits, axis, num_samples.value(), key, s);
        } else {
          return categorical(logits, axis, key, s);
        }
      },
      "logits"_a,
      "axis"_a = -1,
      "shape"_a = none,
      "num_samples"_a = none,
      "key"_a = none,
      "stream"_a = none,
      R"pbdoc(
        Sample from a categorical distribution.

        The values are sampled from the categorical distribution specified by
        the unnormalized values in ``logits``. Note, at most one of ``shape``
        or ``num_samples`` can be specified. If both are ``None``, the output
        has the same shape as ``logits`` with the ``axis`` dimension removed.

        Args:
            logits (array): The *unnormalized* categorical distribution(s).
            axis (int, optional): The axis which specifies the distribution.
               Default is ``-1``.
            shape (list(int), optional): The shape of the output. This must
               be broadcast compatable with ``logits.shape`` with the ``axis``
               dimension removed. Default: ``None``
            num_samples (int, optional): The number of samples to draw from each
              of the categorical distributions in ``logits``. The output will have
              ``num_samples`` in the last dimension. Default: ``None``.
            key (array, optional): A PRNG key. Default: None.

        Returns:
            array: The ``shape``-sized output array with type ``uint32``.
      )pbdoc");
}
