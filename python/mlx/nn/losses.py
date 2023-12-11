# Copyright © 2023 Apple Inc.

import mlx.core as mx
from mlx.nn.layers.base import Module


def _make_loss_module(f):
    def decorator(klass):
        klass.__call__ = lambda self, inputs, targets: f(
            inputs, targets, self.reduction
        )
        return klass

    return decorator


def cross_entropy(
    logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the cross entropy loss between logits and targets.

    Args:
        logits (mx.array): The predicted logits.
        targets (mx.array): The target values.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed cross entropy loss.
    """
    score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)
    loss = mx.logsumexp(logits, axis=axis) - score

    return _reduce(loss, reduction)


def binary_cross_entropy(
    inputs: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    """
    Computes the binary cross entropy loss between inputs and targets.

    Args:
        inputs (mx.array): The predicted inputs (post-sigmoid probabilities).
        targets (mx.array): The target values (binary labels).
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed binary cross entropy loss.
    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> inputs = mx.array([0.1, 0.2, 0.3, 0.4])
        >>> targets = mx.array([0, 0, 1, 1])
        >>> loss = nn.losses.binary_cross_entropy(inputs, targets)
        >>> loss
        array([0.612192])
    """
    loss = -targets * mx.log(inputs) - (1 - targets) * mx.log(1 - inputs)
    return _reduce(loss, reduction)


@_make_loss_module(binary_cross_entropy)
class BCELoss(Module):
    """
    Binary Cross Entropy Loss module.
    It computes the binary cross entropy loss between predicted probabilities (post-sigmoid inputs) and target binary labels.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            - 'none': no reduction (default)
            - 'mean': compute the mean loss
            - 'sum': compute the sum of the loss

    Examples:
        >>> import mlx.core as mx
        >>> from mlx.nn.losses import BCELoss
        >>>
        >>> # Create BCELoss module with default reduction ('none')
        >>> loss_module_none = BCELoss()
        >>> inputs = mx.array([0.5, 0.7, 0.3])
        >>> targets = mx.array([1, 0, 1])
        >>> loss_none = loss_module_none(inputs, targets)
        >>> print(loss_none)
        array([0.693147, 1.20397, 1.20397], dtype=float32)

        >>> # Create BCELoss module with reduction 'mean'
        >>> loss_module_mean = BCELoss(reduction='mean')
        >>> loss_mean = loss_module_mean(inputs, targets)
        >>> print(loss_mean)
        array(1.0337, dtype=float32)

        >>> # Create BCELoss module with reduction 'sum'
        >>> loss_module_sum = BCELoss(reduction='sum')
        >>> loss_sum = loss_module_sum(inputs, targets)
        >>> print(loss_sum)
        array(3.10109, dtype=float32)
    """

    def __init__(self, reduction: str = "none"):
        super().__init__()

        self.reduction = reduction


def l1_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    """
    Computes the L1 loss between predictions and targets.

    Args:
        predictions (mx.array): The predicted values.
        targets (mx.array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed L1 loss.
    """
    loss = mx.mean(mx.abs(predictions - targets))

    return _reduce(loss, reduction)


def mse_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    """
    Computes the mean squared error loss between predictions and targets.

    Args:
        predictions (mx.array): The predicted values.
        targets (mx.array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed mean squared error loss.
    """
    loss = mx.square(predictions - targets)

    return _reduce(loss, reduction)


def nll_loss(
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the negative log likelihood loss between inputs and targets.

    Args:
        inputs (mx.array): The predicted distribution in log space.
        targets (mx.array): The target values.
        axis (int, optional): The distribution axis. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed NLL loss.
    """
    loss = -mx.take_along_axis(inputs, targets[..., None], axis).squeeze(-1)

    return _reduce(loss, reduction)


def kl_div_loss(
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the Kullback-Leibler divergence loss between targets and the
    inputs.

    Computes the following when ``reduction == 'none'``:

    .. code-block:: python

        mx.exp(targets) * (targets - inputs).sum(axis)

    Args:
        inputs (mx.array): Log probabilities for the predicted distribution.
        targets (mx.array): Log probabilities for the target distribution.
        axis (int, optional): The distribution axis. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed Kullback-Leibler divergence loss.
    """
    loss = mx.sum(mx.exp(targets) * (targets - inputs), axis)

    return _reduce(loss, reduction)


def _reduce(loss: mx.array, reduction: str = "none"):
    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")
