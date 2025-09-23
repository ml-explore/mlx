from types import SimpleNamespace

import mlx.core as mx
import mlx.distributions.normal as normal  # Used for the example


def Independent(base_distribution, reinterpreted_batch_ndims: int) -> SimpleNamespace:
    r"""Reinterprets batch dimensions of a distribution as event dimensions.

    This function is a factory that creates a new distribution from a base
    distribution with a batch shape. It treats the last
    ``reinterpreted_batch_ndims`` of the batch dimensions as a single event.

    The primary use of this is to compute the joint log probability of a set of
    independent random variables. For example, if a base distribution has a
    batch shape of ``[B]`` and an event shape of ``[E]``, ``Independent`` can
    create a new distribution with a batch shape of ``[]`` and an event shape
    of ``[B, E]``. The ``log_prob`` method of this new distribution will then
    return a single scalar value representing the sum of the log probabilities
    of the independent events.

    Args:
        base_distribution (module): The base distribution module (e.g.,
          ``mlx.distributions.normal``).
        reinterpreted_batch_ndims (int): The number of trailing batch dimensions
          to treat as event dimensions.

    Returns:
        SimpleNamespace: A new distribution-like module with ``sample`` and
        ``log_prob`` methods.

    Example:
        >>> import mlx.core as mx
        >>> import mlx.distributions.normal as normal
        >>>
        >>> # A batch of 3 independent Normal distributions
        >>> loc = mx.array([0.0, 1.0, 2.0])
        >>> scale = mx.array([1.0, 1.0, 1.0])
        >>>
        >>> # Without Independent, log_prob is element-wise
        >>> base_log_probs = normal.log_prob(loc, loc, scale)
        >>> base_log_probs
        array([-0.918939, -0.918939, -0.918939], dtype=float32)
        >>>
        >>> # Use Independent to treat the batch of 3 as a single 3D event
        >>> from mlx.distributions import Independent
        >>> independent_normal = Independent(normal, reinterpreted_batch_ndims=1)
        >>>
        >>> # The new log_prob returns a single scalar value (the sum)
        >>> joint_log_prob = independent_normal.log_prob(loc, loc, scale)
        >>> joint_log_prob
        array(-2.75682, dtype=float32)
    """

    def sample(key: mx.array, **kwargs) -> mx.array:
        return base_distribution.sample(key=key, **kwargs)

    def log_prob(*args, **kwargs) -> mx.array:
        base_log_prob = base_distribution.log_prob(*args, **kwargs)

        # Sum over the dimensions that are being reinterpreted as part of the event.
        # These are the trailing `reinterpreted_batch_ndims` dimensions.
        if reinterpreted_batch_ndims > 0:
            axes_to_sum = tuple(range(-reinterpreted_batch_ndims, 0))
            return mx.sum(base_log_prob, axis=axes_to_sum)
        return base_log_prob

    return SimpleNamespace(sample=sample, log_prob=log_prob)
