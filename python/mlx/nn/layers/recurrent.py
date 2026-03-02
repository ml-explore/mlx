# Copyright © 2024 Apple Inc.

import math
import os
from typing import Callable, Optional

import mlx.core as mx
from mlx.nn.layers.activations import tanh
from mlx.nn.layers.base import Module

# Versioned RNN implementation: set MLX_RNN_IMPL to revert or try improvements.
# - "legacy": Python-only GRU/LSTM (no Metal fast_cell); use to compare or revert.
# - "fast": Use mx.fast.gru_cell / mx.fast.lstm_cell when available (default).
# - "fast_v2": Same as fast; reserved for future (e.g. precomputed input projection layout).
_RNN_IMPL = os.environ.get("MLX_RNN_IMPL", "fast").strip().lower()
if _RNN_IMPL not in ("legacy", "fast", "fast_v2"):
    _RNN_IMPL = "fast"


def get_rnn_implementation() -> str:
    """Return current RNN implementation mode: 'legacy', 'fast', or 'fast_v2'."""
    return _RNN_IMPL


class RNN(Module):
    r"""An Elman recurrent layer.

    The input is a sequence of shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element along the sequence length axis, this
    layer applies the function:

    .. math::

        h_{t + 1} = \text{tanh} (W_{ih}x_t + W_{hh}h_t + b)

    The hidden state :math:`h` has shape ``NH`` or ``H``, depending on
    whether the input is batched or not. Returns the hidden state at each
    time step, of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool, optional): Whether to use a bias. Default: ``True``.
        nonlinearity (callable, optional): Non-linearity to use. If ``None``,
            then func:`tanh` is used. Default: ``None``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Optional[Callable] = None,
    ):
        super().__init__()

        self.nonlinearity = nonlinearity or tanh
        if not callable(self.nonlinearity):
            raise ValueError(
                f"Nonlinearity must be callable. Current value: {nonlinearity}."
            )

        scale = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.Wxh = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, input_size)
        )
        self.Whh = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, hidden_size)
        )
        self.bias = (
            mx.random.uniform(low=-scale, high=scale, shape=(hidden_size,))
            if bias
            else None
        )

    def _extra_repr(self):
        return (
            f"input_dims={self.Wxh.shape[1]}, "
            f"hidden_size={self.hidden_size}, "
            f"nonlinearity={self.nonlinearity}, bias={self.bias is not None}"
        )

    def __call__(self, x, hidden=None):
        if self.bias is not None:
            x = mx.addmm(self.bias, x, self.Wxh.T)
        else:
            x = x @ self.Wxh.T

        all_hidden = []
        for idx in range(x.shape[-2]):
            if hidden is not None:
                hidden = mx.addmm(x[..., idx, :], hidden, self.Whh.T)
            else:
                hidden = x[..., idx, :]
            hidden = self.nonlinearity(hidden)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class GRU(Module):
    r"""A gated recurrent unit (GRU) RNN layer.

    The input has shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, this layer computes:

    .. math::

        \begin{aligned}
        r_t &= \sigma (W_{xr}x_t + W_{hr}h_t + b_{r}) \\
        z_t &= \sigma (W_{xz}x_t + W_{hz}h_t + b_{z}) \\
        n_t &= \text{tanh}(W_{xn}x_t + b_{n} + r_t \odot (W_{hn}h_t + b_{hn})) \\
        h_{t + 1} &= (1 - z_t) \odot n_t + z_t \odot h_t
        \end{aligned}

    The hidden state :math:`h` has shape ``NH`` or ``H`` depending on
    whether the input is batched or not. Returns the hidden state at each
    time step of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool): Whether to use biases or not. Default: ``True``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        scale = 1.0 / math.sqrt(hidden_size)
        self.Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, input_size)
        )
        self.Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, hidden_size)
        )
        self.b = (
            mx.random.uniform(low=-scale, high=scale, shape=(3 * hidden_size,))
            if bias
            else None
        )
        self.bhn = (
            mx.random.uniform(low=-scale, high=scale, shape=(hidden_size,))
            if bias
            else None
        )

    def _extra_repr(self):
        return (
            f"input_dims={self.Wx.shape[1]}, "
            f"hidden_size={self.hidden_size}, bias={self.b is not None}"
        )

    def __call__(self, x, hidden=None):
        if self.b is not None:
            x = mx.addmm(self.b, x, self.Wx.T)
        else:
            x = x @ self.Wx.T

        x_rz = x[..., : -self.hidden_size]
        x_n = x[..., -self.hidden_size :]
        all_hidden = []
        # legacy = Python-only; fast / fast_v2 = use Metal kernel when on GPU only
        use_fast_cell = (
            _RNN_IMPL != "legacy"
            and hasattr(mx.fast, "gru_cell")
            and mx.default_device() == mx.gpu
        )
        for idx in range(x.shape[-2]):
            # Metal-accelerated path (GRUCell-style): fused kernel when hidden is set.
            # Pass bhn to kernel when bias is used so h_proj stays contiguous (no per-step copy).
            if hidden is not None and use_fast_cell:
                input_proj = x[..., idx, :]
                h_proj = hidden @ self.Wh.T
                hidden = mx.fast.gru_cell(input_proj, h_proj, hidden, bhn=self.bhn)
                all_hidden.append(hidden)
                continue

            rz = x_rz[..., idx, :]
            if hidden is not None:
                h_proj = hidden @ self.Wh.T
                h_proj_rz = h_proj[..., : -self.hidden_size]
                h_proj_n = h_proj[..., -self.hidden_size :]

                if self.bhn is not None:
                    h_proj_n += self.bhn

                rz = rz + h_proj_rz

            rz = mx.sigmoid(rz)

            r, z = mx.split(rz, 2, axis=-1)

            n = x_n[..., idx, :]

            if hidden is not None:
                n = n + r * h_proj_n
            n = mx.tanh(n)

            if hidden is not None:
                hidden = (1 - z) * n + z * hidden
            else:
                hidden = (1 - z) * n

            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class GRUResetAfter(Module):
    r"""Keras-compatible GRU with reset-after formulation (exact gate math).

    Matches TensorFlow/Keras `layers.GRU(reset_after=True)` numerics:
    - Gate order stored as [update, reset, new]
    - Separate input/recurrent biases (``b_i``, ``b_r``)
    - Candidate computed as ``tanh(x_h + r * (h @ U_h + b_{rh}))``

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool): Whether to use biases. Default: ``True``.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size

        scale = 1.0 / math.sqrt(hidden_size)
        # Store weights in Keras layout: (input_dim, 3 * hidden_size)
        self.W = mx.random.uniform(
            low=-scale, high=scale, shape=(input_size, 3 * hidden_size)
        )
        self.U = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, 3 * hidden_size)
        )

        if bias:
            self.b_i = mx.random.uniform(
                low=-scale, high=scale, shape=(3 * hidden_size,)
            )
            self.b_r = mx.random.uniform(
                low=-scale, high=scale, shape=(3 * hidden_size,)
            )
        else:
            self.b_i = None
            self.b_r = None

    def _extra_repr(self):
        return (
            f"input_dims={self.W.shape[0]}, "
            f"hidden_size={self.hidden_size}, bias={self.b_i is not None}"
        )

    def __call__(self, x, hidden=None):
        squeeze_batch = False
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=0)
            squeeze_batch = True

        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, seq, input), got {x.shape}"
            )

        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.W.shape[0]:
            raise ValueError(
                f"Input dim mismatch: expected {self.W.shape[0]}, got {input_dim}"
            )

        if hidden is None:
            hidden = mx.zeros((batch_size, self.hidden_size), dtype=x.dtype)
        else:
            if hidden.ndim == 1:
                hidden = mx.expand_dims(hidden, axis=0)
            if hidden.shape[0] != batch_size:
                raise ValueError(
                    f"Hidden batch {hidden.shape[0]} != input batch {batch_size}"
                )

        x_flat = mx.reshape(x, (-1, input_dim))
        x_proj = mx.matmul(x_flat, self.W)
        if self.b_i is not None:
            x_proj = x_proj + self.b_i
        x_proj = mx.reshape(x_proj, (batch_size, seq_len, 3 * self.hidden_size))

        outputs = []
        for t in range(seq_len):
            x_t = x_proj[:, t, :]
            x_z, x_r, x_h = mx.split(x_t, 3, axis=-1)

            h_proj = mx.matmul(hidden, self.U)
            if self.b_r is not None:
                h_proj = h_proj + self.b_r
            h_z, h_r, h_h = mx.split(h_proj, 3, axis=-1)

            z = mx.sigmoid(x_z + h_z)
            r = mx.sigmoid(x_r + h_r)
            n = mx.tanh(x_h + r * h_h)

            hidden = (1.0 - z) * n + z * hidden
            outputs.append(hidden)

        out = mx.stack(outputs, axis=1)
        if squeeze_batch:
            out = mx.squeeze(out, axis=0)
        return out


class LSTM(Module):
    r"""An LSTM recurrent layer.

    The input has shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, this layer computes:

    .. math::
        \begin{aligned}
        i_t &= \sigma (W_{xi}x_t + W_{hi}h_t + b_{i}) \\
        f_t &= \sigma (W_{xf}x_t + W_{hf}h_t + b_{f}) \\
        g_t &= \text{tanh} (W_{xg}x_t + W_{hg}h_t + b_{g}) \\
        o_t &= \sigma (W_{xo}x_t + W_{ho}h_t + b_{o}) \\
        c_{t + 1} &= f_t \odot c_t + i_t \odot g_t \\
        h_{t + 1} &= o_t \text{tanh}(c_{t + 1})
        \end{aligned}

    The hidden state :math:`h` and cell state :math:`c` have shape ``NH``
    or ``H``, depending on whether the input is batched or not.

    The layer returns two arrays, the hidden state and the cell state at
    each time step, both of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool): Whether to use biases or not. Default: ``True``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        scale = 1.0 / math.sqrt(hidden_size)
        self.Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, input_size)
        )
        self.Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, hidden_size)
        )
        self.bias = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )

    def _extra_repr(self):
        return (
            f"input_dims={self.Wx.shape[1]}, "
            f"hidden_size={self.hidden_size}, bias={self.bias is not None}"
        )

    def __call__(self, x, hidden=None, cell=None):
        if self.bias is not None:
            x = mx.addmm(self.bias, x, self.Wx.T)
        else:
            x = x @ self.Wx.T

        all_hidden = []
        all_cell = []
        # legacy = Python-only; fast / fast_v2 = use Metal kernel when on GPU only
        use_fast_cell = (
            _RNN_IMPL != "legacy"
            and hasattr(mx.fast, "lstm_cell")
            and mx.default_device() == mx.gpu
        )

        for idx in range(x.shape[-2]):
            input_proj = x[..., idx, :]
            if hidden is not None and cell is not None and use_fast_cell:
                hidden_proj = hidden @ self.Wh.T
                cell, hidden = mx.fast.lstm_cell(input_proj, hidden_proj, cell, hidden)
                all_cell.append(cell)
                all_hidden.append(hidden)
                continue

            ifgo = input_proj
            if hidden is not None:
                ifgo = mx.addmm(ifgo, hidden, self.Wh.T)
            i, f, g, o = mx.split(ifgo, 4, axis=-1)

            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            if cell is not None:
                cell = f * cell + i * g
            else:
                cell = i * g
            hidden = o * mx.tanh(cell)

            all_cell.append(cell)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2), mx.stack(all_cell, axis=-2)
