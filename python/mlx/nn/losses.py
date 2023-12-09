# Copyright © 2023 Apple Inc.

import mlx.core as mx


def cross_entropy(
    logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
):
    """
    Computes the cross entropy loss between logits and targets.

    Args:
        logits (mx.array): The predicted logits.
        targets (mx.array): The target values.
        axis (int, optional): The axis over which to compute softmax. Defaults to -1.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied.
                                    'mean': the sum of the output will be divided by the number of elements in the output.
                                    'sum': the output will be summed.
                                    Defaults to 'none'.

    Returns:
        mx.array: The computed cross entropy loss.
    """
    score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)
    loss = mx.logsumexp(logits, axis=axis) - score

    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")


def l1_loss(predictions: mx.array, targets: mx.array):
    """
    Computes the L1 loss between predictions and targets.

    Args:
        predictions (mx.array): The predicted values.
        targets (mx.array): The target values.

    Returns:
        mx.array: The computed L1 loss.
    """
    return mx.mean(mx.abs(predictions - targets))


def mse_loss(predictions: mx.array, targets: mx.array, reduction: str = "none"):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (mx.array): The predicted values.
        targets (mx.array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied.
                                    'mean': the sum of the output will be divided by the number of elements in the output.
                                    'sum': the output will be summed.
                                    Defaults to 'none'.

    Returns:
        mx.array: The computed MSE loss.
    """
    loss = mx.mean(mx.square(predictions - targets), axis=-1)

    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")


def nll_loss(logits: mx.array, targets: mx.array, reduction: str = "none"):
    """
    Computes the Negative Log Likelihood (NLL) loss between logits and targets.

    Args:
        logits (mx.array): The predicted logits.
        targets (mx.array): The target values (assumed to be class labels).
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied.
                                    'mean': the sum of the output will be divided by the number of elements in the output.
                                    'sum': the output will be summed.
                                    Defaults to 'none'.

    Returns:
        mx.array: The computed NLL loss.
    """
    loss = -mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)

    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")


def kl_div_loss(p_logits: mx.array, q_logits: mx.array, reduction: str = "none"):
    """
    Computes the Kullback-Leibler (KL) Divergence loss between two sets of logits, p_logits and q_logits.

    Args:
        p_logits (mx.array): Logits for the distribution p.
        q_logits (mx.array): Logits for the distribution q.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied.
                                    'mean': the sum of the output will be divided by the number of elements in the output.
                                    'sum': the output will be summed.
                                    Defaults to 'none'.

    Returns:
        mx.array: The computed KL Divergence loss.
    """
    p_probs = mx.softmax(p_logits, axis=-1)
    q_probs = mx.softmax(q_logits, axis=-1)

    loss = mx.sum(p_probs * (mx.log(p_probs) - mx.log(q_probs)), axis=-1)

    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")
