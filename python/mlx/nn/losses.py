# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx


def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    weights: mx.array = None,
    axis: int = -1,
    label_smoothing: float = 0.0,
    reduction: str = "none",
) -> mx.array:
    """
    Computes the cross entropy loss.

    Args:
        logits (array): The unnormalized predicted logits.
        targets (array): The target values, as class indices.
        weights (array, optional): Weights for each target. Default: ``None``.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        label_smoothing (float, optional): Label smoothing factor. Default: ``0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed cross entropy loss.
    """
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError(f"Label smoothing must in [0, 1), got {label_smoothing}.")

    score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)
    logsumexp_logits = mx.logsumexp(logits, axis=axis)
    if label_smoothing > 0:
        # Adjust the true class score with label smoothing
        adjusted_score = (1 - label_smoothing) * score

        # Calculate the mean logit across the classes for smoothed loss
        mean_logits = logits.mean(axis=axis)
        smoothed_loss = -mean_logits * label_smoothing

        # Combine the adjusted score and smoothed loss with the logsumexp logits
        loss = logsumexp_logits - adjusted_score + smoothed_loss
    else:
        loss = logsumexp_logits - score

    # Apply weights if provided
    if weights is not None:
        if weights.shape != targets.shape:
            raise ValueError(
                f"Weights with shape {weights.shape} is not the same as "
                f"targets with shape {targets.shape}."
            )
        loss *= weights

    # Apply reduction
    return _reduce(loss, reduction)


def binary_cross_entropy(
    logits: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    """
    Computes the binary cross entropy loss.

    Args:
        logits (array): The unnormalized (pre-sigmoid) predicted logits.
        targets (array): The binary target values in {0, 1}.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed binary cross entropy loss.
    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> inputs = mx.array([0.105361, 0.223144, 1.20397, 0.916291])
        >>> targets = mx.array([0, 0, 1, 1])
        >>> loss = nn.losses.binary_cross_entropy(inputs, targets, "mean")
        >>> loss
        array([0.612192], dtype=float32)
    """
    loss = mx.logaddexp(0.0, logits) - targets * logits
    return _reduce(loss, reduction)


def l1_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "mean"
) -> mx.array:
    """
    Computes the L1 loss.

    Args:
        predictions (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed L1 loss.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"targets shape {targets.shape}."
        )
    loss = mx.abs(predictions - targets)

    return _reduce(loss, reduction)


def mse_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "mean"
) -> mx.array:
    """
    Computes the mean squared error loss.

    Args:
        predictions (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed mean squared error loss.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"targets shape {targets.shape}."
        )

    loss = mx.square(predictions - targets)
    return _reduce(loss, reduction)


def nll_loss(
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the negative log likelihood loss.

    Args:
        inputs (array): The predicted distribution in log space.
        targets (array): The target values.
        axis (int, optional): The distribution axis. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed NLL loss.
    """
    loss = -mx.take_along_axis(inputs, targets[..., None], axis).squeeze(-1)

    return _reduce(loss, reduction)


def kl_div_loss(
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: str = "none"
) -> mx.array:
    """
    Computes the Kullback-Leibler divergence loss.

    Computes the following when ``reduction == 'none'``:

    .. code-block:: python

        mx.exp(targets) * (targets - inputs).sum(axis)

    Args:
        inputs (array): Log probabilities for the predicted distribution.
        targets (array): Log probabilities for the target distribution.
        axis (int, optional): The distribution axis. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed Kullback-Leibler divergence loss.
    """
    loss = mx.sum(mx.exp(targets) * (targets - inputs), axis)

    return _reduce(loss, reduction)


def smooth_l1_loss(
    predictions: mx.array, targets: mx.array, beta: float = 1.0, reduction: str = "mean"
) -> mx.array:
    r"""
    Computes the smooth L1 loss.

    The smooth L1 loss is a variant of the L1 loss which replaces the absolute
    difference with a squared difference when the absolute difference is less
    than ``beta``.

    The formula for the smooth L1 Loss is:

    .. math::

       l =
          \begin{cases}
            0.5 (x - y)^2, & \text{ if } & (x - y) < \beta \\
            |x - y| - 0.5 \beta, &  & \text{otherwise}
          \end{cases}

    Args:
        predictions (array): Predicted values.
        targets (array): Ground truth values.
        beta (float, optional): The threshold after which the loss changes
          from the squared to the absolute difference. Default: ``1.0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed smooth L1 loss.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"targets shape {targets.shape}."
        )

    diff = predictions - targets
    loss = mx.where(
        diff < beta, 0.5 * mx.square(diff) / beta, mx.abs(diff) - 0.5 * beta
    )

    return _reduce(loss, reduction)


def triplet_loss(
    anchors: mx.array,
    positives: mx.array,
    negatives: mx.array,
    axis: int = -1,
    p: int = 2,
    margin: float = 1.0,
    eps: float = 1e-6,
    reduction: str = "none",
) -> mx.array:
    r"""
    Computes the triplet loss for a set of anchor, positive, and negative samples.
    Margin is represented with alpha in the math section.

    .. math::

       L_{\text{triplet}} = \max\left(\|A - P\|_p - \|A - N\|_p + \alpha, 0\right)

    Args:
        anchors (array): The anchor samples.
        positives (array): The positive samples.
        negatives (array): The negative samples.
        axis (int, optional): The distribution axis. Default: ``-1``.
        p (int, optional): The norm degree for pairwise distance. Default: ``2``.
        margin (float, optional): Margin for the triplet loss. Defaults to ``1.0``.
        eps (float, optional): Small positive constant to prevent numerical instability. Defaults to ``1e-6``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: Computed triplet loss. If reduction is "none", returns a tensor of the same shape as input;
                  if reduction is "mean" or "sum", returns a scalar tensor.
    """
    loss = mx.maximum(
        mx.sqrt(mx.power(anchors - positives, p).sum(axis) + eps)
        - mx.sqrt(mx.power(anchors - negatives, p).sum(axis) + eps)
        + margin,
        0,
    )
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


def hinge_loss(
    inputs: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    r"""
    Computes the hinge loss between inputs and targets.

    .. math::

       \text{hinge}(y, y_{\text{pred}}) = \max(0, 1 - y \cdot y_{\text{pred}})


    Args:
        inputs (array): The predicted values.
        targets (array): The target values. They should be -1 or 1.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed hinge loss.
    """
    loss = mx.maximum(1 - inputs * targets, 0)

    return _reduce(loss, reduction)


def huber_loss(
    inputs: mx.array, targets: mx.array, delta: float = 1.0, reduction: str = "none"
) -> mx.array:
    r"""
    Computes the Huber loss between inputs and targets.

    .. math::

        L_{\delta}(a) =
        \left\{ \begin{array}{ll}
            \frac{1}{2} a^2 & \text{for } |a| \leq \delta, \\
            \delta \left( |a| - \frac{1}{2} \delta \right) & \text{otherwise.}
        \end{array} \right.

    Args:
        inputs (array): The predicted values.
        targets (array): The target values.
        delta (float, optional): The threshold at which to change between L1 and L2 loss.
          Default: ``1.0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed Huber loss.
    """
    errors = inputs - targets
    abs_errors = mx.abs(errors)
    quadratic = mx.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    return _reduce(loss, reduction)


def log_cosh_loss(
    inputs: mx.array, targets: mx.array, reduction: str = "none"
) -> mx.array:
    r"""
    Computes the log cosh loss between inputs and targets.

    Logcosh acts like L2 loss for small errors, ensuring stable gradients,
    and like the L1 loss for large errors, reducing sensitivity to outliers. This
    dual behavior offers a balanced, robust approach for regression tasks.

    .. math::

       \text{logcosh}(y_{\text{true}}, y_{\text{pred}}) =
            \frac{1}{n} \sum_{i=1}^{n}
            \log(\cosh(y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}))


    Args:
        inputs (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed log cosh loss.
    """
    errors = inputs - targets
    loss = mx.logaddexp(errors, -errors) - math.log(2)

    return _reduce(loss, reduction)


def focal_loss(
    inputs: mx.array,
    targets: mx.array,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> mx.array:
    r"""
    Computes the Focal Loss between inputs and targets, which is designed to address
    class imbalance by focusing more on hard-to-classify examples.

    .. math::

        FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)

    Args:
        inputs (array): The predicted logits or probabilities.
        targets (array): The ground truth target values.
        alpha (float, optional): The balancing parameter. Default: 0.25.
        gamma (float, optional): The focusing parameter. Default: 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed Focal Loss.
    """
    if gamma < 0:
        raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}.")

    # Calculating the cross-entropy loss
    ce_loss = mx.logaddexp(0.0, inputs) - targets * inputs

    # Calculating the probability
    pt = mx.exp(-ce_loss)

    # Calculating Focal Loss
    focal_loss = -alpha * ((1 - pt) ** gamma) * ce_loss

    return _reduce(focal_loss, reduction)


def dice_loss(
    inputs: mx.array, targets: mx.array, epsilon: float = 1e-6, reduction: str = "none"
) -> mx.array:
    r"""
    Computes the Dice Loss, which is a measure of overlap between two samples.
    This loss is commonly used for binary segmentation tasks.

    .. math::

        \text{Dice Loss} = 1 - \frac{2 \times |X \cap Y|}{|X| + |Y|}

    Args:
        inputs (array): The predicted values.
        targets (array): The ground truth values.
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed Dice Loss.
    """
    intersection = mx.sum(inputs * targets, axis=1)
    cardinality = mx.sum(inputs + targets, axis=1)
    dice_score = (2.0 * intersection + epsilon) / (cardinality + epsilon)
    loss = 1 - dice_score

    return _reduce(loss, reduction)


def iou_loss(
    inputs: mx.array, targets: mx.array, epsilon: float = 1e-6, reduction: str = "none"
) -> mx.array:
    r"""
    Computes the Intersection over Union (IoU) Loss, which is a measure of the
    overlap between two sets, typically used in segmentation tasks.

    .. math::

        \text{IoU Loss} = 1 - \frac{X \cap Y}{X \cup Y}

    Args:
        inputs (array): The predicted values.
        targets (array): The ground truth values.
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed IoU Loss.
    """
    intersection = mx.sum(inputs * targets, axis=1)
    union = mx.sum(inputs + targets - inputs * targets, axis=1)
    iou_score = (intersection + epsilon) / (union + epsilon)
    loss = 1 - iou_score

    return _reduce(loss, reduction)


def contrastive_loss(
    anchors: mx.array,
    positives: mx.array,
    negatives: mx.array,
    margin: float = 1.0,
    p: int = 2,
    reduction: str = "none",
) -> mx.array:
    r"""
    Computes the Contrastive Loss for a set of anchor, positive, and negative samples.

    .. math::

       L_{\text{contrastive}} = \max\left(\|A - P\|_p - \|A - N\|_p + \text{margin}, 0\right)

    Args:
        anchors (array): The anchor samples.
        positives (array): The positive samples.
        negatives (array): The negative samples.
        margin (float, optional): Margin for the contrastive loss. Defaults to ``1.0``.
        p (int, optional): The norm degree for pairwise distance. Default: ``2``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: Computed contrastive loss.
    """
    positive_distance = mx.sqrt(mx.power(anchors - positives, p).sum(axis=1))
    negative_distance = mx.sqrt(mx.power(anchors - negatives, p).sum(axis=1))
    loss = mx.maximum(positive_distance - negative_distance + margin, 0)

    return _reduce(loss, reduction)


def tversky_loss(
    inputs: mx.array,
    targets: mx.array,
    alpha: float = 0.5,
    beta: float = 0.5,
    epsilon: float = 1e-6,
    reduction: str = "none",
) -> mx.array:
    r"""
    Computes the Tversky Loss, a generalization of the Dice Loss, allowing more control over false
    positives and false negatives. It is particularly useful in segmentation tasks with imbalanced datasets.

    .. math::

        \text{Tversky Loss} = 1 - \frac{|X \cap Y|}{|X \cap Y| + \alpha |X \backslash Y| + \beta |Y \backslash X|}

    Args:
        inputs (array): The predicted values.
        targets (array): The ground truth values.
        alpha (float, optional): Controls the penalty for false negatives. Default: 0.5.
        beta (float, optional): Controls the penalty for false positives. Default: 0.5.
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed Tversky Loss.
    """
    intersection = mx.sum(inputs * targets, axis=1)
    false_negatives = mx.sum(inputs * (1 - targets), axis=1)
    false_positives = mx.sum((1 - inputs) * targets, axis=1)
    tversky_index = (intersection + epsilon) / (
        intersection + alpha * false_negatives + beta * false_positives + epsilon
    )
    loss = 1 - tversky_index

    return _reduce(loss, reduction)
