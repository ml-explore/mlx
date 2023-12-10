# Copyright © 2023 Apple Inc.

import mlx.core as mx


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
    predictions: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the mean squared error loss between predictions and targets.

    Args:
        predictions (mx.array): The predicted values.
        targets (mx.array): The target values.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed mean squared error loss.
    """
    loss = mx.mean(mx.square(predictions - targets), axis)

    return _reduce(loss, reduction)


def nll_loss(
    logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the negative log likelihood loss between logits and targets.

    Args:
        logits (mx.array): The predicted logits.
        targets (mx.array): The target values.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed NLL loss.
    """
    loss = -mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)

    return _reduce(loss, reduction)


def kl_div_loss(
    logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the Kullback-Leibler divergence loss between logits and targets.

    Args:
        logits (mx.array): Logits for the distribution p.
        targets (mx.array): Log probabilities for the distribution q.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed Kullback-Leibler divergence loss.
    """
    loss = mx.sum(mx.exp(targets) * (targets - logits), axis)

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
