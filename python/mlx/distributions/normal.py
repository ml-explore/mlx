import math

import mlx.core as mx

# Pre-computed constant for performance
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def sample(loc: mx.array, scale: mx.array, key: mx.array) -> mx.array:
    r"""Samples from a Normal distribution.

    This function uses the reparameterization trick, making it differentiable
    with respect to ``loc`` and ``scale``.

    Args:
        loc (mx.array): The mean of the distribution(s).
        scale (mx.array): The standard deviation of the distribution(s).
        key (mx.array): The PRNG key for random number generation.

    Returns:
        mx.array: An array of samples from the Normal distribution(s).

    Example:
        >>> import mlx.core as mx
        >>> import mlx.distributions.normal as normal
        >>>
        >>> key = mx.random.key(0)
        >>> loc = mx.array([0.0, 10.0])
        >>> scale = mx.array([1.0, 2.0])
        >>> normal.sample(loc, scale, key)
        array([-0.784766, 11.7129], dtype=float32)
    """
    return loc + scale * mx.random.normal(loc.shape, key=key)


def log_prob(value: mx.array, loc: mx.array, scale: mx.array) -> mx.array:
    r"""Computes the log probability of a value given a Normal distribution.

    Args:
        value (mx.array): The value(s) at which to evaluate the log probability.
        loc (mx.array): The mean of the distribution(s).
        scale (mx.array): The standard deviation of the distribution(s).

    Returns:
        mx.array: The log probability of the given values.

    Example:
        >>> import mlx.core as mx
        >>> import mlx.distributions.normal as normal
        >>>
        >>> # Evaluate the log probability at the mean for two distributions
        >>> loc = mx.array([0.0, 10.0])
        >>> scale = mx.array([1.0, 2.0])
        >>> value = mx.array([0.0, 10.0])
        >>> normal.log_prob(value, loc, scale)
        array([-0.918939, -1.61209], dtype=float32)
    """
    var = scale**2
    log_scale = mx.log(scale)
    return -((value - loc) ** 2) / (2 * var) - log_scale - LOG_SQRT_2PI


def entropy(scale: mx.array) -> mx.array:
    r"""Computes the entropy of the Normal distribution.

    Args:
        scale (mx.array): The standard deviation of the distribution(s).

    Returns:
        mx.array: The entropy of the distribution(s).

    Example:
        >>> import mlx.core as mx
        >>> import mlx.distributions.normal as normal
        >>>
        >>> scale = mx.array([1.0, 2.0])
        >>> normal.entropy(scale)
        array([1.41894, 2.11209], dtype=float32)
    """
    return 0.5 + mx.log(scale) + LOG_SQRT_2PI


def mean(loc: mx.array) -> mx.array:
    return loc


def variance(scale: mx.array) -> mx.array:
    return scale**2
