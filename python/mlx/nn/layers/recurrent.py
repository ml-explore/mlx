# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers import Dropout
from mlx.nn.layers.base import Module


class RecurrentCellBase(Module):
    r"""Base class for Recurrent cells. Helper for shared
    attributes and weights initialization.

        Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state(s)
        bias (bool): Whether to use biases or not
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.hidden_size)
        Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(self.input_size, self.hidden_size)
        )
        Wh = self.Wxh = mx.random.uniform(
            low=-scale, high=scale, shape=(self.hidden_size, self.hidden_size)
        )

        if self.bias:
            bx = mx.random.uniform(low=-scale, high=scale, shape=(self.hidden_size,))
            bh = mx.random.uniform(low=-scale, high=scale, shape=(self.hidden_size,))
        else:
            bx = bh = None

        return Wx, Wh, bx, bh


class RNNCell(RecurrentCellBase):
    r"""Apply one step of Elman recurrent unit with tanh or relu
    non-linearity.

    Concretely:

    .. math::

        h' = \text{tanh} (W_{xh}x + b_{xh} + W_{hh}h + b_{hh})

    The input :math:`x` has shape ``ND` or ``D`` and the hidden
    state :math:`h` has shape ``NH`` or ``H``, where:
        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    If `nonlinearity` is `relu`, then ReLU is used instead of tanh.

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        nonlinearity (str): Non-linearity to use. Valid values: `tanh`
        or `relu`. Default: `tanh`.
        bias (bool): Whether to use biases or not. Default: `True`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ):
        super().__init__(input_size, hidden_size, bias)
        self.input_size = input_size

        self.Wxh, self.Whh, self.bh, self.bi = self._init_weights()

        if nonlinearity == "tanh":
            self.activation = mx.tanh
        elif nonlinearity == "relu":
            self.activation = lambda x: mx.maximum(x, 0)
        else:
            raise ValueError(
                f"Unsupported nonlinearity `{nonlinearity}`. Use 'tanh' or 'relu'."
            )

    def __call__(self, x, h_prev):
        out = x @ self.Wxh
        out += h_prev @ self.Whh

        if self.bias:
            out += self.bh + self.bi

        return self.activation(out)


class GRUCell(RecurrentCellBase):
    r"""Apply one step of GRU recurrent unit.

    Concretely:

    .. math::
        r = \sigma (W_{xr}x + b_{xr} + W_{hr}h + b_{hr})
        z = \sigma (W_{xz}x + b_{xz} + W_{hz}h + b_{hz})
        n = \text{tanh}(W_{xn}x + b_{xn} + r * (W_{hn}h + b_{hn}))
        h' = (1 - z) * n + z * h

    The input :math:`x` has shape ``ND` or ``D`` and the hidden
    state :math:`h` has shape ``NH`` or ``H``, where:
        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        bias (bool): Whether to use biases or not. Default: `True`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__(input_size, hidden_size, bias)
        self.input_size = input_size
        # r
        self.Wxr, self.Whr, self.bxr, self.bhr = self._init_weights()
        # n
        self.Wxn, self.Whn, self.bxn, self.bhn = self._init_weights()
        # z
        self.Wxz, self.Whz, self.bxz, self.bhz = self._init_weights()

    def __call__(self, x, h_prev):
        r = x @ self.Wxr + h_prev @ self.Whr
        if self.bias:
            r += self.bxr + self.bhr
        r = mx.sigmoid(r)

        z = x @ self.Wxz + h_prev @ self.Whz
        if self.bias:
            z += self.bxz + self.bhz
        z = mx.sigmoid(z)

        n_x = x @ self.Wxn
        if self.bias:
            n_x += self.bxn

        n_h = h_prev @ self.Whn
        if self.bias:
            n_h += self.bhn

        n = n_x + r * n_h
        n = mx.tanh(n)

        h_next = (1 - z) * n + z * h_prev
        return h_next


class LSTMCell(RecurrentCellBase):
    r"""Apply one step of LSTM recurrent unit.

    Concretely:

    .. math::
        i = \sigma (W_{xi}x + b_{xi} + W_{hi}h + b_{hi})
        f = \sigma (W_{xf}x + b_{xf} + W_{hf}h + b_{hf})
        g = \text{tanh} (W_{xg}x + b_{xg} + W_{hg}h + b_{hg})
        o = \sigma (W_{xo}x + b_{xo} + W_{ho}h + b_{ho})
        c' = f * c + i * g
        h' = o * \text{tanh}(c')

    The input :math:`x` has shape ``ND` or ``D`` the hidden
    state :math:`h` has shape ``NH`` or ``H``, and the cell state
    :math:`c` has shape ``NH`` or ``H`` where:
        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        bias (bool): Whether to use biases or not. Default: `True`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__(input_size, hidden_size, bias)
        # i
        self.Wxi, self.Whi, self.bxi, self.bhi = self._init_weights()
        # f
        self.Wxf, self.Whf, self.bxf, self.bhf = self._init_weights()
        # g
        self.Wxg, self.Whg, self.bxg, self.bhg = self._init_weights()
        # o
        self.Wxo, self.Who, self.bxo, self.bho = self._init_weights()

    def __call__(self, x, h_prev, c_prev):
        i = x @ self.Wxi + h_prev @ self.Whi
        if self.bias:
            i += self.bxi + self.bhi
        i = mx.sigmoid(i)

        f = x @ self.Wxf + h_prev @ self.Whf
        if self.bias:
            f += self.bxf + self.bhf
        f = mx.sigmoid(f)

        g = x @ self.Wxg + h_prev @ self.Whg
        if self.bias:
            g += self.bxg + self.bhg
        g = mx.tanh(g)

        o = x @ self.Wxo + h_prev @ self.Who
        if self.bias:
            o += self.bxo + self.bho
        o = mx.sigmoid(o)

        c_new = f * c_prev + i * g
        h_new = o * mx.tanh(c_new)

        return h_new, c_new


class RecurrentBase(Module):
    r"""Apply one or multiple recurrent units to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    The exact equation implemented by the layer depends on which cell
    is applied. This class is not meant to be used as is, rather as a
    super-class to recurrent layers such as Elman RNN, GRU and LSTM.

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        num_layers (int): Number of layers to apply
        bias (bool): Whether to use biases or not. Default: `True`.
        dropout (float): If larger than zero, applies dropout after all
        layers, except the last.
        bidirectional (bool): If `True`, becomes bidirectional. Default:
        `False`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"Need at least one layer. Given: `{num_layers}`.")

        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"Dropout rate should be in [0, 1). Current value: `{dropout}`"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.layers_forward = [
            self._make_cell(is_first_cell=idx == 0) for idx in range(num_layers)
        ]
        if bidirectional:
            self.layers_backward = [
                self._make_cell(is_first_cell=idx == 0) for idx in range(num_layers)
            ]
        else:
            self.layers_backward = None

        self.do = Dropout(dropout)

    def _init_carry(self, input_shape):
        raise NotImplementedError

    def _make_cell(self, is_first_cell):
        raise NotImplementedError

    def _one_layer_apply(self, x, layer, forward):
        # x has shape (N, L, D) or (L, D)
        if x.ndim == 3:
            carry_shape = (x.shape[0], self.hidden_size)
        else:
            carry_shape = (self.hidden_size,)

        carry = self._init_carry(carry_shape)

        if forward:
            start = 0
            end = x.shape[-2]
            step = 1
        else:
            start = x.shape[-2] - 1
            end = -1
            step = -1

        carries = [None for _ in range(x.shape[-2])]

        for idx in range(start, end, step):
            x_t = x[..., idx, :]
            carry = layer(x_t, *carry)
            # Needed for RNN and GRU, as they return a single array,
            # unlike LSTM that has hidden and cell states.
            if isinstance(carry, mx.array):
                carry = (carry,)

            carries[idx] = carry

        carries = zip(*carries)
        carries = [mx.stack(carry_seq, axis=-2) for carry_seq in carries]
        return carries

    def __call__(self, x):
        if x.ndim not in (2, 3):
            raise ValueError(
                f"Input is expected to have 2 or 3 dimensions. Input has {x.ndim} dimensions."
            )
        x_t = x
        all_last_carries = []

        for idx in range(self.num_layers):
            layer_forward = self.layers_forward[idx]
            forward_carries = self._one_layer_apply(x_t, layer_forward, forward=True)
            # carries each have shape (N, L, H) or (L, H); take carry from last time step
            last_forward_carries = (carries[..., -1, :] for carries in forward_carries)
            all_last_carries.append(last_forward_carries)

            if self.bidirectional:
                layer_backward = self.layers_backward[idx]
                backward_carries = self._one_layer_apply(
                    x_t, layer_backward, forward=False
                )
                # Assumes first carry is used as input for next layer
                x_t = mx.concatenate([forward_carries[0], backward_carries[0]], axis=-1)
                last_backward_carries = [
                    carries[..., -1, :] for carries in backward_carries
                ]
                all_last_carries.append(last_backward_carries)
            else:
                x_t = forward_carries[0]

            if self.dropout > 0.0 and idx < self.num_layers - 1:
                # No dropout on last layer
                x_t = self.do(x_t)

        # concatenate all last carries into arrays of shape (N, num_layers * s, H)
        # where s = 2 if bidirectional, 1 otherwise (just like torch)
        all_last_carries = [
            mx.stack(carry_last, axis=-2) for carry_last in zip(*all_last_carries)
        ]
        return x_t, *all_last_carries


class RNN(RecurrentBase):
    r"""Apply one or multiple Elman recurrent units to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, compute:

    .. math::

        h_{t + 1} = \text{tanh} (W_{ih}x_t + b_{ih} + W_{hh}h_t + b_{hh})

    The input :math:`x` has shape ``NLD` or ``LD`` and the hidden
    state :math:`h` has shape ``NH`` or ``H``, where:
        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    If `nonlinearity` is `relu`, then ReLU is used instead of tanh. The hidden
    state is initialized as zero. Returns two array, :math:`h_{L-1}` and the
    last hidden state of each layer, of shape ``NKH`` where ``K`` is
    ``2 * num_layers`` if `bidirectional=True`, else ``num_layers``.

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        num_layers (int): Number of RNN layers to apply
        nonlinearity (str): Non-linearity to use. Valid values: `tanh`
        or `relu`. Default: `tanh`.
        bias (bool): Whether to use biases or not. Default: `True`.
        dropout (float): If larger than zero, applies dropout after all
        layers, except the last.
        bidirectional (bool): If `True`, becomes a bidirectional RNN. Default:
        `False`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        self.nonlinearity = nonlinearity
        super().__init__(
            input_size, hidden_size, num_layers, bias, dropout, bidirectional
        )

    def _make_cell(self, is_first_cell):
        if is_first_cell:
            return RNNCell(
                self.input_size, self.hidden_size, self.nonlinearity, self.bias
            )
        else:
            s = 2 if self.bidirectional else 1
            return RNNCell(
                self.hidden_size * s, self.hidden_size, self.nonlinearity, self.bias
            )

    def _init_carry(self, input_shape):
        return (mx.zeros(shape=input_shape),)


class GRU(RecurrentBase):
    r"""Apply one or multiple GRU recurrent unit to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, computes:

    .. math::

        r_t = \sigma (W_{xr}x_t + b_{xr} + W_{hr}h_t + b_{hr})
        z_t = \sigma (W_{xz}x_t + b_{xz} + W_{hz}h_t + b_{hz})
        n_t = \text{tanh}(W_{xn}x_t + b_{xn} + r_t * (W_{hn}h_t + b_{hn}))
        h_{t + 1} = (1 - z_t) * n_t + z_t * h_t

    The input :math:`x` has shape ``ND` or ``D`` and the hidden
    state :math:`h` has shape ``NH`` or ``H``, where:

        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    Returns two array, :math:`h_{L-1}` and the
    last hidden state of each layer, of shape ``NKH`` where ``K`` is
    ``2 * num_layers`` if `bidirectional=True`, else ``num_layers``.

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        num_layers (int): Number of layers to apply
        bias (bool): Whether to use biases or not. Default: `True`.
        dropout (float): If larger than zero, applies dropout after all
        layers, except the last.
        bidirectional (bool): If `True`, becomes bidirectional. Default:
        `False`.
    """

    def _make_cell(self, is_first_cell):
        if is_first_cell:
            return GRUCell(self.input_size, self.hidden_size, self.bias)
        else:
            s = 2 if self.bidirectional else 1
            return GRUCell(self.hidden_size * s, self.hidden_size, self.bias)

    def _init_carry(self, input_shape):
        return (mx.zeros(shape=input_shape),)


class LSTM(RecurrentBase):
    r"""Apply one or multiple LSTM recurrent unit to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, computes:

    .. math::
        i_t = \sigma (W_{xi}x_t + b_{xi} + W_{hi}h_t + b_{hi})
        f_t = \sigma (W_{xf}x_t + b_{xf} + W_{hf}h_t + b_{hf})
        g_t = \text{tanh} (W_{xg}x_t + b_{xg} + W_{hg}h_t + b_{hg})
        o_t = \sigma (W_{xo}x_t + b_{xo} + W_{ho}h_t + b_{ho})
        c_{t + 1} = f_t * c_t + i_t * g_t
        h_{t + 1} = o_t \text{tanh}(c_{t + 1})

    The input :math:`x` has shape ``ND` or ``D`` and the hidden
    state :math:`h` has shape ``NH`` or ``H``, where:
        - ``N`` is the batch dimension
        - ``D`` is the input's features dimension
        - ``H`` is the hidden dimension

    Returns three arrays, :math:`h_{L-1}`, the last hidden state of
    each layer, of shape ``NKH`` and the last cell state of each layer,
    of shape ``NKH`, where ``K`` is ``2 * num_layers`` if
    `bidirectional=True`, else ``num_layers``.

    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        num_layers (int): Number of layers to apply
        bias (bool): Whether to use biases or not. Default: `True`.
        dropout (float): If larger than zero, applies dropout after all
        layers, except the last.
        bidirectional (bool): If `True`, becomes bidirectional. Default:
        `False`.
    """

    def _make_cell(self, is_first_cell):
        if is_first_cell:
            return LSTMCell(self.input_size, self.hidden_size, self.bias)
        else:
            s = 2 if self.bidirectional else 1
            return LSTMCell(self.hidden_size * s, self.hidden_size, self.bias)

    def _init_carry(self, input_shape):
        return (mx.zeros(shape=input_shape), mx.zeros(shape=input_shape))
