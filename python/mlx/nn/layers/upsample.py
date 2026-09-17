# Copyright © 2023-2024 Apple Inc.

import operator
from functools import partial, reduce
from itertools import product
from math import ceil
from typing import Callable, Literal, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


def _scaled_indices(N, scale, align_corners, dim, ndims):
    M = int(scale * N)
    if align_corners:
        indices = mx.arange(M, dtype=mx.float32) * ((N - 1) / max(M - 1, 1))
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


def _aa_indices(N, scale, align_corners, dim, ndims, kernel_fn, kernel_radius):
    """Compute antialiased interpolation indices for a given kernel.

    When downscaling (scale < 1), the kernel support widens by 1/scale to
    act as a low-pass filter, preventing aliasing. Out-of-bounds taps are
    zeroed and weights are renormalized per output pixel.

    For upscaling (scale >= 1), the kernel is applied at its native width
    without widening. This matches PyTorch's F.interpolate(antialias=True)
    behavior where the kernel coefficient (e.g. a=-0.5 for cubic) is used
    for both up and downsampling when antialias=True.

    Args:
        N: input size for this dimension
        scale: scale factor for this dimension
        align_corners: align_corners flag
        dim: which spatial dimension
        ndims: number of spatial dimensions
        kernel_fn: callable(distance) -> weight, operating in normalized
            filter coordinates where |distance| < kernel_radius has support
        kernel_radius: support radius of the kernel in normalized coords
            (1.0 for triangle/linear, 2.0 for cubic)
    """
    indices = _scaled_indices(N, scale, align_corners, dim, ndims)

    # For downscale, widen the filter by 1/scale
    if scale < 1:
        inv_scale = 1.0 / scale
    else:
        inv_scale = 1.0

    support = kernel_radius * inv_scale
    num_taps = ceil(support) + 1

    # Compute per-tap weights, zero out-of-bounds, then normalize
    all_idx = []
    all_w = []
    for k in range(-num_taps + 1, num_taps):
        idx = mx.floor(indices) + k
        # Map distance to normalized filter coordinates
        dist = mx.abs(indices - idx) / inv_scale
        w = kernel_fn(dist)
        # Zero out-of-bounds taps
        w = mx.where((idx >= 0) & (idx < N), w, 0.0)
        all_idx.append(idx)
        all_w.append(w)

    # Normalize so weights sum to 1 per output pixel
    w_sum = sum(all_w)
    w_sum = mx.where(w_sum > 0, w_sum, 1.0)

    result = []
    for idx, w in zip(all_idx, all_w):
        w = mx.expand_dims(w / w_sum, -1)
        idx = mx.clip(idx, a_min=0, a_max=N - 1).astype(mx.uint32)
        result.append((idx, w))

    return tuple(result)


def _triangle_kernel(x):
    """Triangle (linear) filter kernel. Support radius = 1."""
    return mx.maximum(1.0 - x, 0.0)


def _cubic_kernel(x):
    """Keys cubic kernel with a=-0.5 (PIL/Pillow convention).

    This coefficient is used by PyTorch when antialias=True for both
    bilinear and bicubic modes. The non-antialiased cubic path uses
    a=-0.75 (OpenCV convention) -- see ``_cubic_indices``.

    Support radius = 2.
    """
    a = -0.5
    w_inner = ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    w_outer = (((x - 5) * x + 8) * x - 4) * a
    return mx.where(x <= 1.0, w_inner, mx.where(x <= 2.0, w_outer, 0.0))


def _linear_aa_indices(N, scale, align_corners, dim, ndims):
    """Linear interpolation with antialiasing (triangle kernel)."""
    return _aa_indices(
        N,
        scale,
        align_corners,
        dim,
        ndims,
        kernel_fn=_triangle_kernel,
        kernel_radius=1.0,
    )


def _cubic_aa_indices(N, scale, align_corners, dim, ndims):
    """Cubic interpolation with antialiasing (Keys cubic, a=-0.5).

    Note: the non-antialiased cubic path (``_cubic_indices``) uses a=-0.75
    (OpenCV convention). When ``antialias=True``, PyTorch switches to a=-0.5
    (PIL convention). This coefficient change affects the interpolant shape,
    not just the filter width. See ``_cubic_kernel`` for details.
    """
    return _aa_indices(
        N,
        scale,
        align_corners,
        dim,
        ndims,
        kernel_fn=_cubic_kernel,
        kernel_radius=2.0,
    )


def _cubic_indices(N, scale, align_corners, dim, ndims):
    indices = _scaled_indices(N, scale, align_corners, dim, ndims)
    indices_l1 = mx.floor(indices)
    indices_r1 = mx.floor(indices + 1)
    indices_l2 = indices_l1 - 1
    indices_r2 = indices_r1 + 1

    @partial(mx.compile, shapeless=True)
    def _get_weight(ind, grid, dist):
        # a=-0.75 (OpenCV convention) for non-antialiased cubic.
        # When antialias=True, _cubic_aa_indices uses a=-0.5 (PIL convention)
        # via _cubic_kernel instead.
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


def _validate_antialias_align_corners(align_corners, antialias):
    if antialias and align_corners:
        raise ValueError(
            "[Upsample] antialias=True with align_corners=True is not "
            "supported. Use align_corners=False for antialiased interpolation."
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


def _interpolate_separable(
    x: mx.array, scale_factor: Tuple, indices_fn: Callable, align_corners: bool = False
):
    dims = x.ndim - 2
    if dims != len(scale_factor):
        raise ValueError("A scale needs to be provided for each spatial dimension")

    _, *N, _ = x.shape
    out = x

    for i, (n, s) in enumerate(zip(N, scale_factor)):
        axis = i + 1
        samples = []
        for idx, weight in indices_fn(n, s, align_corners, i, dims):
            sample = mx.take(out, idx.reshape(-1), axis=axis)
            samples.append(sample * weight)
        out = sum(samples)

    return out


def upsample_linear(
    x: mx.array,
    scale_factor: Tuple,
    align_corners: bool = False,
    antialias: bool = False,
):
    _validate_antialias_align_corners(align_corners, antialias)
    if antialias:
        return _interpolate_separable(
            x=x,
            scale_factor=scale_factor,
            indices_fn=_linear_aa_indices,
            align_corners=align_corners,
        )
    return _interpolate(
        x=x,
        scale_factor=scale_factor,
        indices_fn=_linear_indices,
        align_corners=align_corners,
    )


def upsample_cubic(
    x: mx.array,
    scale_factor: Tuple,
    align_corners: bool = False,
    antialias: bool = False,
):
    _validate_antialias_align_corners(align_corners, antialias)
    if antialias:
        return _interpolate_separable(
            x=x,
            scale_factor=scale_factor,
            indices_fn=_cubic_aa_indices,
            align_corners=align_corners,
        )
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

    .. note::
       When ``antialias=True`` is used with ``"linear"`` or ``"cubic"`` mode,
       an antialiased filter is applied during downsampling (scale factor < 1),
       producing smoother results by avoiding aliasing artifacts. For 2D
       integer-ratio downscales with ``align_corners=False``, this matches the
       behavior of PyTorch's ``F.interpolate(antialias=True)``. Non-integer
       scale factors are supported but may differ from PyTorch because of
       existing index-selection differences.

       For ``"cubic"`` mode, enabling ``antialias`` also changes the cubic
       kernel coefficient from ``a=-0.75`` (OpenCV convention) to ``a=-0.5``
       (PIL/Pillow convention), matching PyTorch's behavior. This affects the
       interpolant shape, not just the filter width.

       ``antialias=True`` with ``align_corners=True`` is not supported and
       will raise a ``ValueError``.

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
        antialias (bool, optional): If ``True``, apply an antialiasing filter
            when downsampling with ``"linear"`` or ``"cubic"`` mode. For
            ``"cubic"`` mode this also switches the kernel coefficient to
            ``a=-0.5``. Not supported with ``"nearest"`` mode or with
            ``align_corners=True``. Default: ``False``.

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
        antialias: bool = False,
    ):
        super().__init__()
        if mode not in ["nearest", "linear", "cubic"]:
            raise ValueError(f"[Upsample] Got unsupported upsampling algorithm: {mode}")
        if antialias and mode == "nearest":
            raise ValueError(
                "[Upsample] Antialiasing is not supported for nearest neighbor upsampling"
            )
        if isinstance(scale_factor, (list, tuple)):
            scale_factor = tuple(map(float, scale_factor))
        else:
            scale_factor = float(scale_factor)

        _validate_antialias_align_corners(align_corners, antialias)

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def _extra_repr(self) -> str:
        repr_str = (
            f"scale_factor={self.scale_factor}, mode={self.mode!r}, "
            f"align_corners={self.align_corners}"
        )
        if self.antialias:
            repr_str += ", antialias=True"
        return repr_str

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
            return upsample_linear(x, scale_factor, self.align_corners, self.antialias)
        elif self.mode == "cubic":
            return upsample_cubic(x, scale_factor, self.align_corners, self.antialias)
        else:
            raise Exception(f"Unknown interpolation mode: {self.mode}")
