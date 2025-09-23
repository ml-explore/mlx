from types import SimpleNamespace

import mlx.core as mx

from .transforms import Transform


def TransformedDistribution(base_distribution, transform: Transform) -> SimpleNamespace:
    r"""Creates a new distribution by applying a transform to a base distribution.

    This function is a factory that generates a new distribution module by
    composing a base distribution with an invertible transformation. The
    probability density of the new distribution is calculated using the
    change of variables formula.

    The returned object is a ``SimpleNamespace`` containing ``sample`` and
    ``log_prob`` functions that can be used like any other distribution module
    in this library.

    Args:
        base_distribution (module): The base distribution module (e.g.,
          ``mlx.distributions.normal``).
        transform (Transform): An instance of a ``Transform`` subclass (e.g.,
          ``ExpTransform()``).

    Returns:
        SimpleNamespace: A new distribution-like module with ``sample`` and
        ``log_prob`` methods.

    Example:
        >>> from mlx.distributions.transformed_distribution import TransformedDistribution
        >>> # Create a LogNormal distribution from a Normal distribution
        >>> log_normal = TransformedDistribution(normal, ExpTransform())
        >>>
        >>> # Sample from the new distribution
        >>> key = mx.random.key(0)
        >>> loc = mx.array([0.0])
        >>> scale = mx.array([1.0])
        >>> samples = log_normal.sample(loc=loc, scale=scale, key=key)
        >>> samples
        array([0.813962], dtype=float32)
        >>>
        >>> # Calculate the log probability of a value
        >>> # This is equivalent to stats.lognorm.logpdf(1.0, s=1.0, scale=1.0)
        >>> log_prob_val = log_normal.log_prob(value=1.0, loc=0.0, scale=1.0)
        >>> log_prob_val
        array(-0.918939, dtype=float32)
    """

    def sample(key: mx.array, **kwargs) -> mx.array:
        base_sample = base_distribution.sample(key=key, **kwargs)
        return transform.forward(base_sample)

    def log_prob(value: mx.array, **kwargs) -> mx.array:
        # This is the change of variables formula for probabilities:
        # log p_Y(y) = log p_X(x) - log|det(J)|
        # where y = f(x) and J is the Jacobian of f.

        # Computing x = f^{-1}(y)
        x = transform.inverse(value)

        # Computing the log prob of the origin value x
        base_log_prob = base_distribution.log_prob(value=x, **kwargs)

        # Computing the log determinant of the Jacobian
        log_det = transform.log_abs_det_jacobian(x, value)

        return base_log_prob - log_det

    # Return as new module
    return SimpleNamespace(sample=sample, log_prob=log_prob)
