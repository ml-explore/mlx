# Copyright © 2023 Apple Inc.

import mlx.core as mx

def cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1, reduction: str = 'mean'):
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
    return mx.mean(mx.abs(predictions - targets))

