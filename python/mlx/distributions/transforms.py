from abc import ABC, abstractmethod

import mlx.core as mx


class Transform(ABC):
    """Abstract base class for invertible, differentiable transformations.

    A ``Transform`` represents a differentiable, invertible mapping, to create
    complex distributions via the ``TransformedDistribution`` factory.
    """

    @abstractmethod
    def forward(self, x: mx.array) -> mx.array:
        """Computes the forward transformation ``y = f(x)``.

        Args:
            x (mx.array): The input array to the transformation.

        Returns:
            mx.array: The transformed output array.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y: mx.array) -> mx.array:
        """Computes the inverse transformation ``x = f^-1(y)``.

        Args:
            y (mx.array): The output array from the forward transformation.

        Returns:
            mx.array: The reconstructed input array.
        """
        raise NotImplementedError

    @abstractmethod
    def log_abs_det_jacobian(self, x: mx.array, y: mx.array) -> mx.array:
        """Computes the log of the absolute value of the determinant of the Jacobian.

        This is a key component of the change of variables formula used to compute
        the log probability of a ``TransformedDistribution``. For a 1-D
        transformation, this simplifies to ``log|dy/dx|``.

        Args:
            x (mx.array): The input array to the forward transformation.
            y (mx.array): The output array from the forward transformation.

        Returns:
            mx.array: The log absolute determinant of the Jacobian.
        """
        raise NotImplementedError


class ExpTransform(Transform):
    r"""A transformation applying the exponential function ``y = exp(x)``.

    This is commonly used to transform a distribution defined on the real line
    (like the Normal distribution) to one defined on the positive real line
    (like the LogNormal distribution).

    Example:
        >>> import mlx.core as mx
        >>> from mlx.distributions.transforms import ExpTransform
        >>>
        >>> transform = ExpTransform()
        >>> x = mx.array([0.0, 1.0, -1.0])
        >>> y = transform.forward(x)
        >>> y
        array([1, 2.71828, 0.367879], dtype=float32)
        >>>
        >>> # Check the inverse operation
        >>> transform.inverse(y)
        array([0, 1, -1], dtype=float32)
    """

    def forward(self, x: mx.array) -> mx.array:
        return mx.exp(x)

    def inverse(self, y: mx.array) -> mx.array:
        return mx.log(y)

    def log_abs_det_jacobian(self, x, y) -> mx.array:
        # The derivative of exp(x) is exp(x).
        # The log of the absolute value of the derivative is log(exp(x))
        # We use x directly as it's often more numerically stable than log(y).
        return x
