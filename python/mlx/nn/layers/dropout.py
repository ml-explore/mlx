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

        return (1 / self._p_1) * mask.astype(x.dtype) * x
