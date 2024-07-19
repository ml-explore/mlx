import math
from typing import Optional, Tuple, Union

import mlx.core as mx


class Normal:
    def __init__(self, mu: mx.array, sigma: mx.array):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(
        self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None
    ):
        return mx.random.normal(sample_shape, key=key) * self.sigma + self.mu

    def log_prob(self, x: mx.array):
        return (
            -0.5 * math.log(2 * math.pi)
            - mx.log(self.sigma)
            - 0.5 * ((x - self.mu) / self.sigma) ** 2
        )

    def sample_and_log_prob(
        self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None
    ):
        x = self.sample(sample_shape, key=key)
        return x, self.log_prob(x)
