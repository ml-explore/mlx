import mlx.core as mx
from mlx.nn.layers.base import Module


def flatten(a: mx.array, start_dim: int = 0, end_dim: int = -1) -> mx.array:
    r"""
    Flattens a contiguous range of dims into a tensor.

    See :class:`~nn.Flatten` for details.

    Args:
        input: input tensor.
        start_dim: first dim to flatten (default = 0).
        end_dim: last dim to flatten (default = -1).
    """
    
    return mx.flatten(a, start_dim, end_dim)


class Flatten(Module):
    r"""
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
    
if __name__ == "__main__":
    
    import mlx.core as mx
    import mlx.nn as nn

    # Example 1: With default parameters
    input_tensor = mx.random.normal((32, 3, 224, 224))
    flattener = nn.Flatten()
    output_tensor = flattener(input_tensor)
    print(output_tensor.shape)  # Output: [32, 150528]

    # Example 2: With non-default parameters
    flattener = nn.Flatten(start_dim=0, end_dim=2)
    output_tensor = flattener(input_tensor)
    print(output_tensor.shape)  # Output: [21504, 224]
