# Copyright © 2023 Apple Inc.

from mlx.nn.layers.base import Module


class Sequential(Module):
    """A layer that calls the passed callables in order.

    We can pass either modules or plain callables to the Sequential module. If
    our functions have learnable parameters they should be implemented as
    ``nn.Module`` instances.

    Args:
        modules (tuple of Callables): The modules to call in order
    """

    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x):
        for m in self.layers:
            x = m(x)
        return x
