import math

import mlx.core as mx
from mlx.nn.layers.base import Module


class RNN(Module):
    r"""Apply one Elman recurrent units to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the optional batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element along the sequence length axis, compute:

    .. math::

        h_{t + 1} = \text{tanh} (W_{ih}x_t + b_{ih} + W_{hh}h_t + b_{hh})

    The hidden state :math:`h` has shape ``NH`` or ``H``, depending
    whether the input is batched or not. Returns the hidden state at each
    time step, of shape ``NLH`` or ``LH``.


    Args:
        input_size (int): Dimension of the input :math:`x`
        hidden_size (int): Dimension of the hidden state :math:`h`
        nonlinearity (callable): Non-linearity to use. Default: `mx.tanh`.
        bias (bool): Whether to use biases or not. Default: `True`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: callable = mx.tanh,
        bias: bool = True,
    ):
        super().__init__()

        if not callable(nonlinearity):
            raise ValueError(
                f"Nonlinearity must be callable. Current value: {nonlinearity}"
            )

        scale = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.Wxh = mx.random.uniform(
            low=-scale, high=scale, shape=(input_size, hidden_size)
        )
        self.Whh = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, hidden_size)
        )
        self.bh = (
            mx.random.uniform(low=-scale, high=scale, shape=(hidden_size,))
            if bias
            else None
        )
        self.nonlinearity = nonlinearity

    def _extra_repr(self):
        return f"input_dims={self.Wxh.shape[0]}, hidden_size={self.hidden_size}, nonlinearity={self.nonlinearity}, bias={self.bh is not None}"

    def __call__(self, x, hidden=None):
        x_proj = x @ self.Wxh

        if self.bh is not None:
            x_proj += x_proj

        if hidden is None:
            hidden = mx.zeros(shape=(self.hidden_size,))
        all_hidden = []

        for idx in range(x.shape[-2]):
            hidden = x_proj[..., idx, :] + hidden @ self.Whh
            hidden = self.nonlinearity(hidden)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class GRU(Module):
    r"""Apply one GRU recurrent unit to a sequence of shape
    ``NLD`` or ``LD``, where:
        - ``N`` is the optional batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, computes:

    .. math::

        r_t = \sigma (W_{xr}x_t + b_{xr} + W_{hr}h_t + b_{hr})
        z_t = \sigma (W_{xz}x_t + b_{xz} + W_{hz}h_t + b_{hz})
        n_t = \text{tanh}(W_{xn}x_t + b_{xn} + r_t \odot (W_{hn}h_t + b_{hn}))
        h_{t + 1} = (1 - z_t) \odot n_t + z_t \odot h_t

    The hidden state :math:`h` has shape ``NH`` or ``H``, depending
    whether the input is batched or not. Returns the hidden state at each
    time step, of shape ``NLH`` or ``LH``.

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
        super().__init__()

        self.hidden_size = hidden_size
        scale = 1.0 / math.sqrt(hidden_size)
        self.Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(input_size, 3 * hidden_size)
        )
        self.Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, 3 * hidden_size)
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
        return f"input_dims={self.Wx.shape[0]}, hidden_size={self.hidden_size}, bias={self.b is not None}"

    def __call__(self, x, hidden=None):
        x_proj = x @ self.Wx

        if self.b is not None:
            x_proj += self.b

        x_proj_rz = x_proj[..., : -self.hidden_size]
        x_proj_n = x_proj[..., -self.hidden_size :]

        all_hidden = []

        if hidden is None:
            hidden = mx.zeros(shape=(self.hidden_size,))

        for idx in range(x.shape[-2]):
            h_proj = hidden @ self.Wh
            h_proj_rz = h_proj[..., : -self.hidden_size]
            h_proj_n = h_proj[..., -self.hidden_size :]

            if self.bhn is not None:
                h_proj_n += self.bhn

            # Note bias in r, z, n is added through x already
            rz = x_proj_rz[..., idx, :] + h_proj_rz
            rz = mx.sigmoid(rz)

            r, z = mx.split(rz, 2, axis=-1)

            n_x = x_proj_n[..., idx, :]

            n = n_x + r * h_proj_n
            n = mx.tanh(n)

            hidden = (1 - z) * n + z * hidden
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class LSTM(Module):
    r"""Apply one LSTM recurrent unit to a sequence of
    shape ``NLD`` or ``LD``, where:
        - ``N`` is the optional batch dimension
        - ``L`` is the sequence length
        - ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, computes:

    .. math::
        i_t = \sigma (W_{xi}x_t + b_{xi} + W_{hi}h_t + b_{hi})
        f_t = \sigma (W_{xf}x_t + b_{xf} + W_{hf}h_t + b_{hf})
        g_t = \text{tanh} (W_{xg}x_t + b_{xg} + W_{hg}h_t + b_{hg})
        o_t = \sigma (W_{xo}x_t + b_{xo} + W_{ho}h_t + b_{ho})
        c_{t + 1} = f_t \odot c_t + i_t \odot g_t
        h_{t + 1} = o_t \text{tanh}(c_{t + 1})

    The hidden state :math:`h` and cell state :math:`c` have shape ``NH``
    or ``H``, depending whether the input is batched or not. Returns two
    arrays: the hidden state and the cell state at each time step, both
    of shape ``NLH`` or ``LH``.

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
        super().__init__()

        self.hidden_size = hidden_size
        scale = 1.0 / math.sqrt(hidden_size)
        self.Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(input_size, 4 * hidden_size)
        )
        self.Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(hidden_size, 4 * hidden_size)
        )
        self.b = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )

    def _extra_repr(self):
        return f"input_dims={self.Wx.shape[0]}, hidden_size={self.hidden_size}, bias={self.b is not None}"

    def __call__(self, x, hidden=None, cell=None):
        x_proj = x @ self.Wx

        if self.b is not None:
            x_proj += self.b

        if hidden is None:
            hidden = mx.zeros(shape=(self.hidden_size,))

        if cell is None:
            cell = mx.zeros(shape=(self.hidden_size,))

        all_hidden = []
        all_cell = []

        for idx in range(x.shape[-2]):
            h_proj = hidden @ self.Wh

            ifgo = x_proj[..., idx, :] + h_proj
            i, f, g, o = mx.split(ifgo, 4, axis=-1)

            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            cell = f * cell + i * g
            hidden = o * mx.tanh(cell)

            all_cell.append(cell)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2), mx.stack(all_cell, axis=-2)
