# Copyright © 2023 Apple Inc.

import mlx.core as mx


def cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1):
    score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)
    return mx.logsumexp(logits, axis=axis) - score
