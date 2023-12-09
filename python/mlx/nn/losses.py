# Copyright © 2023 Apple Inc.

import mlx.core as mx

def cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = 'mean'):
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
                                    Defaults to 'mean'.

    Returns:
        mx.array: The computed cross entropy loss.
    """
    logits = logits - mx.max(logits, axis=axis, keepdims=True)
    log_probs = mx.log(mx.softmax(logits, axis=axis))
    score = mx.take_along_axis(log_probs, targets[..., None], axis).squeeze(-1)
    
    if reduction == 'mean':
        return -mx.mean(score)
    elif reduction == 'sum':
        return -mx.sum(score)
    elif reduction == 'none':
        return -score
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

