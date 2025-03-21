# Copyright © 2023 Apple Inc.

import math
from typing import Literal, Optional, get_args

import mlx.core as mx

Reduction = Literal["none", "mean", "sum"]


def _reduce(loss: mx.array, reduction: Reduction = "none"):
    if reduction not in get_args(Reduction):
        raise ValueError(f"Invalid reduction. Must be one of {get_args(Reduction)}.")

    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    elif reduction == "none":
        return loss


def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    weights: Optional[mx.array] = None,
    axis: int = -1,
    label_smoothing: float = 0.0,
    reduction: Reduction = "none",
) -> mx.array:
    """
    Computes the cross entropy loss.

    Args:
        logits (array): The unnormalized logits.
        targets (array): The ground truth values. These can be class indices or
            probabilities for each class. If the ``targets`` are class indices,
            then ``targets`` shape should match the ``logits`` shape with
            the ``axis`` dimension removed. If the ``targets`` are probabilities
            (or one-hot encoded), then the ``targets`` shape should be the same as
            the ``logits`` shape.
        weights (array, optional): Optional weights for each target. Default: ``None``.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        label_smoothing (float, optional): Label smoothing factor. Default: ``0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed cross entropy loss.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>>
        >>> # Class indices as targets
        >>> logits = mx.array([[2.0, -1.0], [-1.0, 2.0]])
        >>> targets = mx.array([0, 1])
        >>> nn.losses.cross_entropy(logits, targets)
        array([0.0485873, 0.0485873], dtype=float32)
        >>>
        >>> # Probabilities (or one-hot vectors) as targets
        >>> logits = mx.array([[2.0, -1.0], [-1.0, 2.0]])
        >>> targets = mx.array([[0.9, 0.1], [0.1, 0.9]])
        >>> nn.losses.cross_entropy(logits, targets)
        array([0.348587, 0.348587], dtype=float32)
    """
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError(f"Label smoothing must in [0, 1), got {label_smoothing}.")

    # Whether targets are class indices or probabilities
    targets_as_probs = targets.ndim == logits.ndim

    def _drop_dim(shape, axis):
        shape = list(shape)
        shape.pop(axis)
        return tuple(shape)

    # Check shapes in two cases: targets as class indices and targets as probabilities
    if (targets_as_probs and targets.shape != logits.shape) or (
        not targets_as_probs and targets.shape != _drop_dim(logits.shape, axis)
    ):
        raise ValueError(
            f"Targets shape {targets.shape} does not match logits shape {logits.shape}."
        )

    if targets_as_probs:
        score = mx.sum(logits * targets, axis=axis)
    else:
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
        if weights.shape != loss.shape:
            raise ValueError(
                f"Weights with shape {weights.shape} is not the same as "
                f"output loss with shape {loss.shape}."
            )
        loss *= weights

    # Apply reduction
    return _reduce(loss, reduction)


def binary_cross_entropy(
    inputs: mx.array,
    targets: mx.array,
    weights: Optional[mx.array] = None,
    with_logits: bool = True,
    reduction: Reduction = "mean",
) -> mx.array:
    """
    Computes the binary cross entropy loss.

    By default, this function takes the pre-sigmoid logits, which results in a faster
    and more precise loss. For improved numerical stability when ``with_logits=False``,
    the loss calculation clips the input probabilities (in log-space) to a minimum value
    of ``-100``.

    Args:
        inputs (array): The predicted values. If ``with_logits`` is ``True``, then
            ``inputs`` are unnormalized logits. Otherwise, ``inputs`` are probabilities.
        targets (array): The binary target values in {0, 1}.
        with_logits (bool, optional): Whether ``inputs`` are logits. Default: ``True``.
        weights (array, optional): Optional weights for each target. Default: ``None``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed binary cross entropy loss.
    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn

        >>> logits = mx.array([0.105361, 0.223144, 1.20397, 0.916291])
        >>> targets = mx.array([0, 0, 1, 1])
        >>> loss = nn.losses.binary_cross_entropy(logits, targets, reduction="mean")
        >>> loss
        array(0.539245, dtype=float32)

        >>> probs = mx.array([0.1, 0.1, 0.4, 0.4])
        >>> targets = mx.array([0, 0, 1, 1])
        >>> loss = nn.losses.binary_cross_entropy(probs, targets, with_logits=False, reduction="mean")
        >>> loss
        array(0.510826, dtype=float32)
    """
    if inputs.shape != targets.shape:
        raise ValueError(
            f"Inputs shape {inputs.shape} does not match targets shape {targets.shape}."
        )

    if with_logits:
        loss = mx.logaddexp(0.0, inputs) - inputs * targets
    else:
        log_inputs_clip = mx.clip(mx.log(inputs), a_min=-100, a_max=None)
        log_inputs_inv_clip = mx.clip(mx.log(1 - inputs), a_min=-100, a_max=None)
        loss = -(targets * log_inputs_clip + (1 - targets) * log_inputs_inv_clip)

    # Apply weights if provided
    if weights is not None:
        if weights.shape != loss.shape:
            raise ValueError(
                f"Weights with shape {weights.shape} is not the same as "
                f"output loss with shape {loss.shape}."
            )
        loss *= weights

    return _reduce(loss, reduction)


def l1_loss(
    predictions: mx.array, targets: mx.array, reduction: Reduction = "mean"
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
    predictions: mx.array, targets: mx.array, reduction: Reduction = "mean"
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
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: Reduction = "none"
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


def gaussian_nll_loss(
    inputs: mx.array,
    targets: mx.array,
    vars: mx.array,
    full: bool = False,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> mx.array:
    r"""
    Computes the negative log likelihood loss for a Gaussian distribution.

    The loss is given by:

    .. math::
        \frac{1}{2}\left(\log\left(\max\left(\text{vars},
        \ \epsilon\right)\right) + \frac{\left(\text{inputs} - \text{targets} \right)^2}
        {\max\left(\text{vars}, \ \epsilon \right)}\right) + \text{const.}

    where ``inputs`` are the predicted means and ``vars`` are the the
    predicted variances.

    Args:
        inputs (array): The predicted expectation of the Gaussian distribution.
        targets (array): The target values (samples from the Gaussian distribution).
        vars (array): The predicted variance of the Gaussian distribution.
        full (bool, optional): Whether to include the constant term in the loss calculation.
            Default: ``False``.
        eps (float, optional): Small positive constant for numerical stability.
            Default: ``1e-6``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The Gaussian NLL loss.
    """
    if inputs.shape != targets.shape:
        raise ValueError(
            f"Inputs shape {inputs.shape} does not match targets shape {targets.shape}."
        )

    if inputs.shape != vars.shape:
        raise ValueError(
            f"Inputs shape {inputs.shape} does not match vars shape {vars.shape}."
        )

    # For stability
    vars = mx.maximum(vars, eps)
    loss = 0.5 * (mx.log(vars) + mx.square(targets - inputs) / vars)

    if full:
        loss += 0.5 * math.log(2 * math.pi)

    return _reduce(loss, reduction)


def kl_div_loss(
    inputs: mx.array, targets: mx.array, axis: int = -1, reduction: Reduction = "none"
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
    predictions: mx.array,
    targets: mx.array,
    beta: float = 1.0,
    reduction: Reduction = "mean",
) -> mx.array:
    r"""
    Computes the smooth L1 loss.

    The smooth L1 loss is a variant of the L1 loss which replaces the absolute
    difference with a squared difference when the absolute difference is less
    than ``beta``.

    The formula for the smooth L1 Loss is:

    .. math::

      l = \begin{cases}
            0.5 (x - y)^2 / \beta, & \text{if } |x - y| < \beta \\
            |x - y| - 0.5 \beta, & \text{otherwise}
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

    diff = mx.abs(predictions - targets)
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
    reduction: Reduction = "none",
) -> mx.array:
    r"""
    Computes the triplet loss for a set of anchor, positive, and negative samples.
    Margin is represented with alpha in the math section.

    .. math::

       \max\left(\|A - P\|_p - \|A - N\|_p + \alpha, 0\right)

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


def hinge_loss(
    inputs: mx.array, targets: mx.array, reduction: Reduction = "none"
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
    inputs: mx.array,
    targets: mx.array,
    delta: float = 1.0,
    reduction: Reduction = "none",
) -> mx.array:
    r"""
    Computes the Huber loss between inputs and targets.

    .. math::

        l_{\delta}(a) =
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
    inputs: mx.array, targets: mx.array, reduction: Reduction = "none"
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


def cosine_similarity_loss(
    x1: mx.array,
    x2: mx.array,
    axis: int = 1,
    eps: float = 1e-8,
    reduction: Reduction = "none",
) -> mx.array:
    r"""
    Computes the cosine similarity between the two inputs.

    The cosine similarity loss is given by

    .. math::

        \frac{x_1 \cdot x_2}{\max(\|x_1\|  \cdot \|x_2\|, \epsilon)}

    Args:
        x1 (mx.array): The first set of inputs.
        x2 (mx.array): The second set of inputs.
        axis (int, optional): The embedding axis. Default: ``1``.
        eps (float, optional): The minimum value of the denominator used for
          numerical stability. Default: ``1e-8``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        mx.array: The computed cosine similarity loss.
    """
    x1_norm = mx.linalg.norm(x1, axis=axis)
    x2_norm = mx.linalg.norm(x2, axis=axis)

    loss = mx.sum(x1 * x2, axis=axis) / mx.maximum(x1_norm * x2_norm, eps)

    return _reduce(loss, reduction)


def margin_ranking_loss(
    inputs1: mx.array,
    inputs2: mx.array,
    targets: mx.array,
    margin: float = 0.0,
    reduction: Reduction = "none",
) -> mx.array:
    r"""
    Calculate the margin ranking loss that loss given inputs :math:`x_1`, :math:`x_2` and a label
    :math:`y` (containing 1 or -1).

    The loss is given by:

    .. math::
        \text{loss} = \max (0, -y * (x_1 - x_2) + \text{margin})

    Where :math:`y` represents ``targets``, :math:`x_1` represents ``inputs1`` and :math:`x_2`
    represents ``inputs2``.

    Args:
        inputs1 (array): Scores for the first input.
        inputs2 (array): Scores for the second input.
        targets (array): Labels indicating whether samples in ``inputs1`` should be ranked higher
            than samples in ``inputs2``. Values should be 1 or -1.
        margin (float, optional): The margin by which the scores should be separated.
            Default: ``0.0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed margin ranking loss.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> targets = mx.array([1, 1, -1])
        >>> inputs1 = mx.array([-0.573409, -0.765166, -0.0638])
        >>> inputs2 = mx.array([0.75596, 0.225763, 0.256995])
        >>> loss = nn.losses.margin_ranking_loss(inputs1, inputs2, targets)
        >>> loss
        array(0.773433, dtype=float32)
    """
    if not (inputs1.shape == inputs2.shape == targets.shape):
        raise ValueError(
            f"The shapes of the arguments do not match. The provided shapes are "
            f"inputs1.shape={inputs1.shape}, inputs2.shape={inputs2.shape}, and "
            f"targets.shape={targets.shape}."
        )

    differences = inputs1 - inputs2
    loss = mx.maximum(0, -targets * differences + margin)

    return _reduce(loss, reduction)
