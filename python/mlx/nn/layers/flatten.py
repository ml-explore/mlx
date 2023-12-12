import mlx.core as mx
from mlx.nn.layers.base import Module


def _get_new_shape(a: mx.array, start: int, end: int):
    shape = a.shape

    if a.ndim == 0 or (end == start):
        return shape
    end = min(end, a.ndim - 1)
    length = shape[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] == 0 or shape[idx + 1] == 0:
            length = 0
            break

        if shape[idx] == 1:
            continue

        length = length * shape[idx]

    new_shape = (*shape[:start], length, *shape[end + 1:])
    return new_shape


def flatten(a: mx.array, start_dim: int = 0, end_dim: int = -1) -> mx.array:
    r"""
    Flattens a contiguous range of dims into a tensor.

    See :class:`~nn.Flatten` for details.

    Args:
        input: input tensor.
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    """

    start_dim = max(0, start_dim)
    end_dim = a.ndim if end_dim == -1 else min(a.ndim, end_dim)
    
    if start_dim > end_dim:
        raise ValueError("start_dim must be less than or equal to end_dim")

    if start_dim == end_dim and a.ndim != 0:
        return a
    return mx.reshape(a, _get_new_shape(a, start_dim, end_dim))


class Flatten(Module):
    """
    Flattens input by reshaping it into a one-dimensional array. For use with :class:`~nn.Flatten`.
    See :meth:`mx.nn.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim (int, optional): The starting dimension for flattening. Default is 1.
        end_dim (int, optional): The ending dimension for flattening. Default is -1.

    Note:
        If start_dim or end_dim are passed, only dimensions starting with start_dim
        and ending with end_dim are flattened. The order of elements in input is unchanged.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> input = mx.random.normal((32, 3, 224, 224))
        >>> # With default parameters
        >>> m = Flatten()
        >>> output = m(input)
        >>> output.shape
        (32, 150528)
        >>> # With non-default parameters
        >>> m = Flatten(start_dim=0, end_dim=2)
        >>> output = m(input)
        >>> output.shape
        (21504, 224)
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, a: mx.array) -> mx.array:
        return flatten(a, self.start_dim, self.end_dim)
