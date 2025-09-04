// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <chrono>

#include "mlx/ops.h"
#include "mlx/random.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

class PyKeySequence {
 public:
  explicit PyKeySequence(uint64_t seed) {
    state_.append(mx::random::key(seed));
  }

  void seed(uint64_t seed) {
    state_[0] = mx::random::key(seed);
  }

  mx::array next() {
    auto out = mx::random::split(nb::cast<mx::array>(state_[0]));
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
      &mx::random::key,
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
      nb::overload_cast<const mx::array&, int, mx::StreamOrDevice>(
          &mx::random::split),
      "key"_a,
      "num"_a = 2,
      "stream"_a = nb::none(),
      nb::sig(
          "def split(key: array, num: int = 2, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Split a PRNG key into sub keys.

        Args:
            key (array): Input key to split.
            num (int, optional): Number of sub keys. Default: ``2``.

        Returns:
            array: The array of sub keys with ``num`` as its first dimension.
      )pbdoc");
  m.def(
      "uniform",
      [](const ScalarOrArray& low,
         const ScalarOrArray& high,
         const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return mx::random::uniform(
            to_array(low),
            to_array(high),
            shape,
            type.value_or(mx::float32),
            key,
            s);
      },
      "low"_a = 0,
      "high"_a = 1,
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::float32,
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
            low (scalar or array, optional): Lower bound of the distribution.
              Default: ``0``.
            high (scalar or array, optional): Upper bound of the distribution.
              Default: ``1``.
            shape (list(int), optional): Shape of the output. Default:``()``.
            dtype (Dtype, optional): Type of the output. Default: ``float32``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The output array random values.
      )pbdoc");
  m.def(
      "normal",
      [](const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         const std::optional<ScalarOrArray>& loc_,
         const std::optional<ScalarOrArray>& scale_,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto dtype = type.value_or(mx::float32);
        auto key = key_ ? key_.value() : default_key().next();
        auto loc =
            loc_ ? std::make_optional(to_array(*loc_, dtype)) : std::nullopt;
        auto scale = scale_ ? std::make_optional(to_array(*scale_, dtype))
                            : std::nullopt;
        return mx::random::normal(shape, dtype, loc, scale, key, s);
      },
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::float32,
      "loc"_a = nb::none(),
      "scale"_a = nb::none(),
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def normal(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, loc: Union[scalar, array, None] = None, scale: Union[scalar, array, None] = None, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate normally distributed random numbers.

        If ``loc`` and ``scale`` are not provided the "standard" normal
        distribution is used. That means $x \sim \mathcal{N}(0, 1)$ for
        real numbers and $\text{Re}(x),\text{Im}(x) \sim \mathcal{N}(0,
        \frac{1}{2})$ for complex numbers.

        Args:
            shape (list(int), optional): Shape of the output. Default: ``()``.
            dtype (Dtype, optional): Type of the output. Default: ``float32``.
            loc (scalar or array, optional): Mean of the distribution.
              Default: ``None``.
            scale (scalar or array, optional): Standard deviation of the
              distribution. Default: ``None``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "multivariate_normal",
      [](const mx::array& mean,
         const mx::array& cov,
         const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return mx::random::multivariate_normal(
            mean, cov, shape, type.value_or(mx::float32), key, s);
      },
      "mean"_a,
      "cov"_a,
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def multivariate_normal(mean: array, cov: array, shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate jointly-normal random samples given a mean and covariance.

        The matrix ``cov`` must be positive semi-definite. The behavior is
        undefined if it is not.  The only supported ``dtype`` is ``float32``.

        Args:
            mean (array): array of shape ``(..., n)``, the mean of the
              distribution.
            cov (array): array  of shape ``(..., n, n)``, the covariance
              matrix of the distribution. The batch shape ``...`` must be
              broadcast-compatible with that of ``mean``.
            shape (list(int), optional): The output shape must be
              broadcast-compatible with ``mean.shape[:-1]`` and ``cov.shape[:-2]``.
              If empty, the result shape is determined by broadcasting the batch
              shapes of ``mean`` and ``cov``. Default: ``[]``.
            dtype (Dtype, optional): The output type. Default: ``float32``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "randint",
      [](const ScalarOrArray& low,
         const ScalarOrArray& high,
         const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return mx::random::randint(
            to_array(low),
            to_array(high),
            shape,
            type.value_or(mx::int32),
            key,
            s);
      },
      "low"_a,
      "high"_a,
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::int32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def randint(low: Union[scalar, array], high: Union[scalar, array], shape: Sequence[int] = [], dtype: Optional[Dtype] = int32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate random integers from the given interval.

        The values are sampled with equal probability from the integers in
        half-open interval ``[low, high)``. The lower and upper bound can be
        scalars or arrays and must be broadcastable to ``shape``.

        Args:
            low (scalar or array): Lower bound of the interval.
            high (scalar or array): Upper bound of the interval.
            shape (list(int), optional): Shape of the output. Default: ``()``.
            dtype (Dtype, optional): Type of the output. Default: ``int32``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The array of random integers.
      )pbdoc");
  m.def(
      "bernoulli",
      [](const ScalarOrArray& p_,
         const std::optional<mx::Shape> shape,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        auto p = to_array(p_);
        if (shape.has_value()) {
          return mx::random::bernoulli(p, shape.value(), key, s);
        } else {
          return mx::random::bernoulli(p, key, s);
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
              distribution. Default: ``0.5``.
            shape (list(int), optional): Shape of the output.
              Default: ``p.shape``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The array of random integers.
      )pbdoc");
  m.def(
      "truncated_normal",
      [](const ScalarOrArray& lower_,
         const ScalarOrArray& upper_,
         const std::optional<mx::Shape> shape_,
         std::optional<mx::Dtype> type,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        auto lower = to_array(lower_);
        auto upper = to_array(upper_);
        auto t = type.value_or(mx::float32);
        if (shape_.has_value()) {
          return mx::random::truncated_normal(
              lower, upper, shape_.value(), t, key, s);
        } else {
          return mx::random::truncated_normal(lower, upper, t, key, s);
        }
      },
      "lower"_a,
      "upper"_a,
      "shape"_a = nb::none(),
      "dtype"_a.none() = mx::float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def truncated_normal(lower: Union[scalar, array], upper: Union[scalar, array], shape: Optional[Sequence[int]] = None, dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate values from a truncated normal distribution.

        The values are sampled from the truncated normal distribution
        on the domain ``(lower, upper)``. The bounds ``lower`` and ``upper``
        can be scalars or arrays and must be broadcastable to ``shape``.

        Args:
            lower (scalar or array): Lower bound of the domain.
            upper (scalar or array): Upper bound of the domain.
            shape (list(int), optional): The shape of the output.
              Default:``()``.
            dtype (Dtype, optional): The data type of the output.
              Default: ``float32``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "gumbel",
      [](const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return mx::random::gumbel(shape, type.value_or(mx::float32), key, s);
      },
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gumbel(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Union[None, Stream, Device] = None, stream: Optional[array] = None) -> array"),
      R"pbdoc(
        Sample from the standard Gumbel distribution.

        The values are sampled from a standard Gumbel distribution
        which CDF ``exp(-exp(-x))``.

        Args:
            shape (list(int)): The shape of the output.
            dtype (Dtype, optional): The data type of the output.
              Default: ``float32``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array:
              The :class:`array` with shape ``shape`` and distributed according
              to the Gumbel distribution.
      )pbdoc");
  m.def(
      "categorical",
      [](const mx::array& logits,
         int axis,
         const std::optional<mx::Shape> shape,
         const std::optional<int> num_samples,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        if (shape.has_value() && num_samples.has_value()) {
          throw std::invalid_argument(
              "[categorical] At most one of shape or num_samples can be specified.");
        } else if (shape.has_value()) {
          return mx::random::categorical(logits, axis, shape.value(), key, s);
        } else if (num_samples.has_value()) {
          return mx::random::categorical(
              logits, axis, num_samples.value(), key, s);
        } else {
          return mx::random::categorical(logits, axis, key, s);
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
               Default: ``-1``.
            shape (list(int), optional): The shape of the output. This must
               be broadcast compatible with ``logits.shape`` with the ``axis``
               dimension removed. Default: ``None``
            num_samples (int, optional): The number of samples to draw from each
              of the categorical distributions in ``logits``. The output will have
              ``num_samples`` in the last dimension. Default: ``None``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The ``shape``-sized output array with type ``uint32``.
      )pbdoc");
  m.def(
      "laplace",
      [](const mx::Shape& shape,
         std::optional<mx::Dtype> type,
         float loc,
         float scale,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return mx::random::laplace(
            shape, type.value_or(mx::float32), loc, scale, key, s);
      },
      "shape"_a = mx::Shape{},
      "dtype"_a.none() = mx::float32,
      "loc"_a = 0.0,
      "scale"_a = 1.0,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def laplace(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, loc: float = 0.0, scale: float = 1.0, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Sample numbers from a Laplace distribution.

        Args:
            shape (list(int), optional): Shape of the output. Default: ``()``.
            dtype (Dtype, optional): Type of the output. Default: ``float32``.
            loc (float, optional): Mean of the distribution. Default: ``0.0``.
            scale (float, optional): The scale "b" of the Laplace distribution.
              Default:``1.0``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array: The output array of random values.
      )pbdoc");
  m.def(
      "permutation",
      [](const std::variant<nb::int_, mx::array>& x,
         int axis,
         const std::optional<mx::array>& key_,
         mx::StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        if (auto pv = std::get_if<nb::int_>(&x); pv) {
          return mx::random::permutation(nb::cast<int>(*pv), key, s);
        } else {
          return mx::random::permutation(std::get<mx::array>(x), axis, key, s);
        }
      },
      "x"_a,
      "axis"_a = 0,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def permutation(x: Union[int, array], axis: int = 0, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate a random permutation or permute the entries of an array.

        Args:
            x (int or array, optional): If an integer is provided a random
              permtuation of ``mx.arange(x)`` is returned. Otherwise the entries
              of ``x`` along the given axis are randomly permuted.
            axis (int, optional): The axis to permute along. Default: ``0``.
            key (array, optional): A PRNG key. Default: ``None``.

        Returns:
            array:
              The generated random permutation or randomly permuted input array.
      )pbdoc");
  // Register static Python object cleanup before the interpreter exits
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() { default_key().release(); }));
}
