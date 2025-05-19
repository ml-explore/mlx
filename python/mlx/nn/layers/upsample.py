# Copyright © 2023-2024 Apple Inc.

import operator
from functools import partial, reduce
from itertools import product
from typing import Callable, Literal, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


def _scaled_indices(N, scale, align_corners, dim, ndims):
    M = int(scale * N)
    if align_corners:
        indices = mx.arange(M, dtype=mx.float32) * ((N - 1) / (M - 1))
    else:
        step = 1 / scale
        start = ((M - 1) * step - N + 1) / 2
        indices = mx.arange(M, dtype=mx.float32) * step - start

    shape = [1] * ndims
    shape[dim] = -1

    return indices.reshape(shape)


def _nearest_indices(N, scale, dim, ndims):
    M = int(scale * N)
    indices = mx.arange(M, dtype=mx.float32)
    if M > N:
        indices = (indices + 0.5) * (N / M) - 0.5
        indices = indices.round()
    else:
        indices = indices * (N / M)
    shape = [1] * ndims
    shape[dim] = -1
    return indices.astype(mx.uint32).reshape(shape)


def _linear_indices(N, scale, align_corners, dim, ndims):
    indices = _scaled_indices(N, scale, align_corners, dim, ndims)
    indices = mx.clip(indices, a_min=0, a_max=N - 1)
    indices_l = mx.floor(indices)
    indices_r = mx.ceil(indices)
    weight = indices - indices_l
    weight = mx.expand_dims(weight, -1)

    return (
        (indices_l.astype(mx.uint32), 1 - weight),
        (indices_r.astype(mx.uint32), weight),
    )


def _cubic_indices(N, scale, align_corners, dim, ndims):
    indices = _scaled_indices(N, scale, align_corners, dim, ndims)
    indices_l1 = mx.floor(indices)
    indices_r1 = mx.floor(indices + 1)
    indices_l2 = indices_l1 - 1
    indices_r2 = indices_r1 + 1

    @partial(mx.compile, shapeless=True)
    def _get_weight(ind, grid, dist):
        # PyTorch uses -0.5 for antialiasing=true (compatibility with PIL)
        # and uses -0.75 for antialiasing=false (compatibility with OpenCV)
        a = -0.75
        x = mx.abs(ind - grid)
        if dist == 1:
            weight = ((a + 2.0) * x - (a + 3.0)) * x * x + 1
        else:
            weight = (((x - 5) * x + 8) * x - 4) * a
        return weight

    weight_l1 = _get_weight(indices, indices_l1, dist=1)[..., None]
    weight_r1 = _get_weight(indices, indices_r1, dist=1)[..., None]
    weight_l2 = _get_weight(indices, indices_l2, dist=2)[..., None]
    weight_r2 = _get_weight(indices, indices_r2, dist=2)[..., None]

    # padding with border value
    indices_l1 = mx.clip(indices_l1, a_min=0, a_max=N - 1)
    indices_r1 = mx.clip(indices_r1, a_min=0, a_max=N - 1)
    indices_l2 = mx.clip(indices_l2, a_min=0, a_max=N - 1)
    indices_r2 = mx.clip(indices_r2, a_min=0, a_max=N - 1)

    return (
        (indices_l1.astype(mx.uint32), weight_l1),
        (indices_r1.astype(mx.uint32), weight_r1),
        (indices_l2.astype(mx.uint32), weight_l2),
        (indices_r2.astype(mx.uint32), weight_r2),
    )


def upsample_nearest(x: mx.array, scale_factor: Tuple):
    dims = x.ndim - 2
    if dims != len(scale_factor):
        raise ValueError("A scale needs to be provided for each spatial dimension")

    # Integer scale_factors means we can simply expand-broadcast and reshape
    if tuple(map(int, scale_factor)) == scale_factor:
        shape = list(x.shape)
        for d in range(dims):
            shape.insert(2 + 2 * d, 1)
        x = x.reshape(shape)
        for d in range(dims):
            shape[2 + 2 * d] = int(scale_factor[d])
        x = mx.broadcast_to(x, shape)
        for d in range(dims):
            shape[d + 1] *= shape[d + 2]
            shape.pop(d + 2)
        x = x.reshape(shape)
        return x

    else:
        B, *N, C = x.shape
        indices = [slice(None)]
        for i, (n, s) in enumerate(zip(N, scale_factor)):
            indices.append(_nearest_indices(n, s, i, dims))
        indices = tuple(indices)

        return x[indices]


def _interpolate(
    x: mx.array, scale_factor: Tuple, indices_fn: Callable, align_corners: bool = False
):
    dims = x.ndim - 2
    if dims != len(scale_factor):
        raise ValueError("A scale needs to be provided for each spatial dimension")

    B, *N, C = x.shape

    # Compute the sampling grid
    indices = []
    for i, (n, s) in enumerate(zip(N, scale_factor)):
        indices.append(indices_fn(n, s, align_corners, i, dims))

    # Sample and compute the weights
    samples = []
    weights = []
    for idx_weight in product(*indices):
        idx, weight = zip(*idx_weight)
        samples.append(x[(slice(None),) + idx])
        weights.append(reduce(operator.mul, weight))

    # Interpolate
    return sum(wi * xi for wi, xi in zip(weights, samples))


def upsample_linear(x: mx.array, scale_factor: Tuple, align_corners: bool = False):
    return _interpolate(
        x=x,
        scale_factor=scale_factor,
        indices_fn=_linear_indices,
        align_corners=align_corners,
    )


def upsample_cubic(x: mx.array, scale_factor: Tuple, align_corners: bool = False):
    return _interpolate(
        x=x,
        scale_factor=scale_factor,
        indices_fn=_cubic_indices,
        align_corners=align_corners,
    )


class Upsample(Module):
    r"""Upsample the input signal spatially.

    The spatial dimensions are by convention dimensions ``1`` to ``x.ndim -
    2``. The first is the batch dimension and the last is the feature
    dimension.

    For example, an audio signal would be 3D with 1 spatial dimension, an image
    4D with 2 and so on and so forth.

    There are three upsampling algorithms implemented nearest neighbor upsampling,
    linear interpolation, and cubic interpolation. All can be applied to any number
    of spatial dimensions. The linear interpolation will be bilinear, trilinear etc
    when applied to more than one spatial dimension. And cubic interpolation will be
    bicubic when there are 2 spatial dimensions.

    .. note::
       When using one of the linear or cubic interpolation modes the ``align_corners``
       argument changes how the corners are treated in the input image. If
       ``align_corners=True`` then the top and left edge of the input and
       output will be matching as will the bottom right edge.

    Parameters:
        scale_factor (float or tuple): The multiplier for the spatial size.
            If a ``float`` is provided, it is the multiplier for all spatial dimensions.
            Otherwise, the number of scale factors provided must match the
            number of spatial dimensions.
        mode (str, optional): The upsampling algorithm, either ``"nearest"``,
            ``"linear"`` or ``"cubic"``. Default: ``"nearest"``.
        align_corners (bool, optional): Changes the way the corners are treated
            during ``"linear"`` and ``"cubic"`` upsampling.  See the note above and the
            examples below for more details.  Default: ``False``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.arange(1, 5).reshape((1, 2, 2, 1))
        >>> x
        array([[[[1],
                 [2]],
                [[3],
                 [4]]]], dtype=int32)
        >>> n = nn.Upsample(scale_factor=2, mode='nearest')
        >>> n(x).squeeze()
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [3, 3, 4, 4],
               [3, 3, 4, 4]], dtype=int32)
        >>> b = nn.Upsample(scale_factor=2, mode='linear')
        >>> b(x).squeeze()
        array([[1, 1.25, 1.75, 2],
               [1.5, 1.75, 2.25, 2.5],
               [2.5, 2.75, 3.25, 3.5],
               [3, 3.25, 3.75, 4]], dtype=float32)
        >>> b = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        >>> b(x).squeeze()
        array([[1, 1.33333, 1.66667, 2],
               [1.66667, 2, 2.33333, 2.66667],
               [2.33333, 2.66667, 3, 3.33333],
               [3, 3.33333, 3.66667, 4]], dtype=float32)
    """

    def __init__(
        self,
        scale_factor: Union[float, Tuple],
        mode: Literal["nearest", "linear", "cubic"] = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        if mode not in ["nearest", "linear", "cubic"]:
            raise ValueError(f"[Upsample] Got unsupported upsampling algorithm: {mode}")
        if isinstance(scale_factor, (list, tuple)):
            self.scale_factor = tuple(map(float, scale_factor))
        else:
            self.scale_factor = float(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def _extra_repr(self) -> str:
        return (
            f"scale_factor={self.scale_factor}, mode={self.mode!r}, "
            f"align_corners={self.align_corners}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        dims = x.ndim - 2
        if dims <= 0:
            raise ValueError(
                f"[Upsample] The input should have at least 1 spatial "
                f"dimension which means it should be at least 3D but "
                f"{x.ndim}D was provided"
            )

        scale_factor = self.scale_factor
        if isinstance(scale_factor, tuple):
            if len(scale_factor) != dims:
                raise ValueError(
                    f"[Upsample] One scale per spatial dimension is required but "
                    f"scale_factor={scale_factor} and the number of spatial "
                    f"dimensions were {dims}"
                )
        else:
            scale_factor = (scale_factor,) * dims

        if self.mode == "nearest":
            return upsample_nearest(x, scale_factor)
        elif self.mode == "linear":
            return upsample_linear(x, scale_factor, self.align_corners)
        elif self.mode == "cubic":
            return upsample_cubic(x, scale_factor, self.align_corners)
        else:
            raise Exception(f"Unknown interpolation mode: {self.mode}")
