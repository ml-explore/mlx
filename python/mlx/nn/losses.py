# Copyright © 2023 Apple Inc.

import mlx.core as mx

def cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1):
    score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)
    return mx.logsumexp(logits, axis=axis) - score

def MLE_loss(logits: mx.array, targets: mx.array, axis: int = -1):
    logits = logits- mx.max(logits, axis=axis, keepdims=True)
    log_probs = mx.log(mx.softmax(logits, axis=axis))
    score = mx.take_along_axis(log_probs, targets[..., None], axis).squeeze(-1)
    return -mx.mean(score)

def l1_loss(predictions: mx.array, targets: mx.array):
    return mx.mean(mx.abs(predictions - targets))

