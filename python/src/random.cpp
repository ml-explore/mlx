// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <chrono>

#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/random.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;
using namespace mlx::core::random;

class PyKeySequence {
 public:
  explicit PyKeySequence(uint64_t seed) {
    state_.append(key(seed));
  }

  void seed(uint64_t seed) {
    state_[0] = key(seed);
  }

  array next() {
    auto out = split(nb::cast<array>(state_[0]));
    state_[0] = out.first;
    return out.second;
  }

  nb::list state() {
    return state_;
  }

  void release() {
    nb::gil_scoped_acquire gil;
    state_.release().dec_ref();
  }

 private:
  nb::list state_;
};

PyKeySequence& default_key() {
  auto get_current_time_seed = []() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
  };
  static PyKeySequence ks(get_current_time_seed());
  return ks;
}

void init_random(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "random",
      "mlx.core.random: functionality related to random number generation");

  m.attr("state") = default_key().state();
  m.def(
      "seed",
      [](uint64_t seed) { default_key().seed(seed); },
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
      nb::overload_cast<const array&, int, StreamOrDevice>(&random::split),
      "key"_a,
      "num"_a = 2,
      "stream"_a = nb::none(),
      nb::sig(
          "def split(key: array, num: int = 2, stream: Union[None, Stream, Device] = None) -> array)"),
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
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
      "dtype"_a.none() = float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def uniform(low: Union[scalar, array] = 0, high: Union[scalar, array] = 1, shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate uniformly distributed random numbers.

        The values are sampled uniformly in the half-open interval ``[low, high)``.
        The lower and upper bound can be scalars or arrays and must be
        broadcastable to ``shape``.

        Args:
            low (scalar or array, optional): Lower bound of the distribution. Default is ``0``.
            high (scalar or array, optional): Upper bound of the distribution. Default is ``1``.
            shape (list(int), optional): Shape of the output. Default is ``()``.
            key (array, optional): A PRNG key. Default: ``None``.
            dtype (Dtype, optional): Type of the output. Default is ``float32``.

        Returns:
            array: The output array random values.
      )pbdoc");
  m.def(
      "normal",
      [](const std::vector<int>& shape,
         std::optional<Dtype> type,
         float loc,
         float scale,
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return normal(shape, type.value_or(float32), loc, scale, key, s);
      },
      "shape"_a = std::vector<int>{},
      "dtype"_a.none() = float32,
      "loc"_a = 0.0,
      "scale"_a = 1.0,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def normal(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, loc: float = 0.0, scale: float = 1.0, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate normally distributed random numbers.

        Args:
            shape (list(int), optional): Shape of the output. Default is ``()``.
            dtype (Dtype, optional): Type of the output. Default is ``float32``.
            loc (float, optional): Mean of the distribution. Default is ``0.0``.
            scale (float, optional): Standard deviation of the distribution. Default is ``1.0``.
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return randint(
            to_array(low), to_array(high), shape, type.value_or(int32), key, s);
      },
      "low"_a,
      "high"_a,
      "shape"_a = std::vector<int>{},
      "dtype"_a.none() = int32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def randint(low: Union[scalar, array], high: Union[scalar, array], shape: Sequence[int] = [], dtype: Optional[Dtype] = int32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        auto p = to_array(p_);
        if (shape.has_value()) {
          return bernoulli(p, shape.value(), key, s);
        } else {
          return bernoulli(p, key, s);
        }
      },
      "p"_a = 0.5,
      "shape"_a = nb::none(),
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bernoulli(p: Union[scalar, array] = 0.5, shape: Optional[Sequence[int]] = None, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
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
      "shape"_a = nb::none(),
      "dtype"_a.none() = float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def truncated_normal(lower: Union[scalar, array], upper: Union[scalar, array], shape: Optional[Sequence[int]] = None, dtype: float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return gumbel(shape, type.value_or(float32), key, s);
      },
      "shape"_a = std::vector<int>{},
      "dtype"_a.none() = float32,
      "stream"_a = nb::none(),
      "key"_a = nb::none(),
      nb::sig(
          "def gumbel(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, stream: Optional[array] = None, key: Union[None, Stream, Device] = None) -> array"),
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
         const std::optional<array>& key_,
         StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
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
      "shape"_a = nb::none(),
      "num_samples"_a = nb::none(),
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def categorical(logits: array, axis: int = -1, shape: Optional[Sequence[int]] = None, num_samples: Optional[int] = None, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
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
  // Register static Python object cleanup before the interpreter exits
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() { default_key().release(); }));
}
