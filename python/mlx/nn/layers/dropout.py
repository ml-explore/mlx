# Copyright © 2023 Apple Inc.

import mlx.core as mx
from mlx.nn.layers.base import Module


class Dropout(Module):
    """Randomly zero a portion of the elements during training.

    The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
    :math:`p` is the probability of zeroing an element. This is done so the
    expected value of a given element will remain the same.

    Args:
        p (float): The probability to zero an element
    """

    def __init__(self, p: float = 0.5):
        super().__init__()

        if p < 0 or p >= 1:
            raise ValueError("The dropout probability should be in [0, 1)")

        self._p_1 = 1 - p

    def _extra_repr(self):
        return f"p={1-self._p_1}"

    def __call__(self, x):
        if self._p_1 == 1 or not self.training:
            return x

        mask = mx.random.bernoulli(self._p_1, x.shape)

        return (1 / self._p_1) * mask * x


class Dropout2d(Module):
    """Apply 2D channel-wise dropout during training.

    Randomly zero out entire channels independently with probability :math:`p`.
    This layer expects the channels to be last, i.e. the input shape should be
    ``NWHC`` or ``WHC`` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels

    The remaining channels are scaled by :math:`\frac{1}{1-p}` to
    maintain the expected value of each element. Unlike traditional dropout,
    which zeros individual entries, this layer zeros entire channels. This is
    beneficial for early convolution layers where adjacent pixels are
    correlated. In such case, traditional dropout may not effectively
    regularize activations. For more details, see [1].

    [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
    Efficient Object Localization Using Convolutional Networks. CVPR 2015.

    Args:
        p (float): Probability of zeroing a channel during training.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()

        if p < 0 or p >= 1:
            raise ValueError("The dropout probability should be in [0, 1)")

        self._p_1 = 1 - p

    def _extra_repr(self):
        return f"p={1-self._p_1}"

    def __call__(self, x):
        if x.ndim not in (3, 4):
            raise ValueError(
                f"Received input with {x.ndim} dimensions. Expected 3 or 4 dimensions."
            )

        if self._p_1 == 1 or not self.training:
            return x

        # Dropout is applied on the whole channel
        # 3D input: (1, 1, C)
        # 4D input: (B, 1, 1, C)
        mask_shape = x.shape
        mask_shape[-2] = mask_shape[-3] = 1

        mask = mx.random.bernoulli(p=self._p_1, shape=mask_shape)
        return (1 / self._p_1) * mask * x
