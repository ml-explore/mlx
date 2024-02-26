import math

import mlx.core as mx
from mlx.nn.layers.base import Module


def _weight_init(input_size: int, hidden_size: int):
    scale = 1.0 / math.sqrt(hidden_size)
    return mx.random.uniform(low=-scale, high=scale, shape=(input_size, hidden_size))


def _bias_init(hidden_size: int, use_bias: bool):
    if not use_bias:
        return None

    scale = 1.0 / math.sqrt(hidden_size)
    return mx.random.uniform(low=-scale, high=scale, shape=(hidden_size,))


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

        self.Wxh = _weight_init(input_size, hidden_size)
        self.Whh = _weight_init(hidden_size, hidden_size)
        self.bh = _bias_init(hidden_size, bias)
        self.nonlinearity = nonlinearity

    def _extra_repr(self):
        return f"input_dims={self.Wxh.shape[0]}, hidden_size={self.Wxh.shape[-1]}, nonlinearity={self.nonlinearity}, bias={self.bh is not None}"

    def __call__(self, x):
        x_proj = x @ self.Wxh

        if self.bh is not None:
            x_proj += x_proj

        curr_hidden = mx.zeros(shape=(self.Whh.shape[-1],))
        all_hidden = []

        for idx in range(x.shape[-2]):
            curr_hidden = x_proj[..., idx, :] + curr_hidden @ self.Whh
            curr_hidden = self.nonlinearity(curr_hidden)
            all_hidden.append(curr_hidden)

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
        self.bias = bias
        # r
        self.Wxr = _weight_init(input_size, hidden_size)
        self.Whr = _weight_init(hidden_size, hidden_size)
        self.br = _bias_init(hidden_size, bias)
        # n
        self.Wxn = _weight_init(input_size, hidden_size)
        self.Whn = _weight_init(hidden_size, hidden_size)
        self.bn = _bias_init(hidden_size, bias)
        self.bhn = _bias_init(hidden_size, bias)
        # z
        self.Wxz = _weight_init(input_size, hidden_size)
        self.Whz = _weight_init(hidden_size, hidden_size)
        self.bz = _bias_init(hidden_size, bias)

    def _extra_repr(self):
        return f"input_dims={self.Wxr.shape[0]}, hidden_size={self.Wxr.shape[-1]}, bias={self.bias}"

    def __call__(self, x):
        x_proj_r = x @ self.Wxr
        x_proj_n = x @ self.Wxn
        x_proj_z = x @ self.Wxz

        if self.bias:
            x_proj_r += self.br
            x_proj_n += self.bn
            x_proj_z += self.bz

        all_hidden = []
        curr_hidden = mx.zeros(shape=(self.Whr.shape[0],))

        for idx in range(x.shape[-2]):
            r = x_proj_r[..., idx, :] + curr_hidden @ self.Whr
            r = mx.sigmoid(r)

            z = x_proj_z[..., idx, :] + curr_hidden @ self.Whz
            z = mx.sigmoid(z)

            n_x = x_proj_n[..., idx, :]

            n_h = curr_hidden @ self.Whn
            if self.bias:
                n_h += self.bhn

            n = n_x + r * n_h
            n = mx.tanh(n)

            curr_hidden = (1 - z) * n + z * curr_hidden
            all_hidden.append(curr_hidden)

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
        self.bias = bias
        # i
        self.Wxi = _weight_init(input_size, hidden_size)
        self.Whi = _weight_init(hidden_size, hidden_size)
        self.bi = _bias_init(hidden_size, bias)
        # f
        self.Wxf = _weight_init(input_size, hidden_size)
        self.Whf = _weight_init(hidden_size, hidden_size)
        self.bf = _bias_init(hidden_size, bias)
        # g
        self.Wxg = _weight_init(input_size, hidden_size)
        self.Whg = _weight_init(hidden_size, hidden_size)
        self.bg = _bias_init(hidden_size, bias)
        # o
        self.Wxo = _weight_init(input_size, hidden_size)
        self.Who = _weight_init(hidden_size, hidden_size)
        self.bo = _bias_init(hidden_size, bias)

    def _extra_repr(self):
        return f"input_dims={self.Wxi.shape[0]}, hidden_size={self.Whi.shape[0]}, bias={self.bias}"

    def __call__(self, x):
        x_proj_i = x @ self.Wxi
        x_proj_f = x @ self.Wxf
        x_proj_g = x @ self.Wxg
        x_proj_o = x @ self.Wxo

        if self.bias:
            x_proj_i += self.bi
            x_proj_f += self.bf
            x_proj_g += self.bg
            x_proj_o += self.bo

        all_hidden = []
        all_cell = []
        curr_hidden = mx.zeros(shape=(self.Whi.shape[0],))
        curr_cell = mx.zeros(shape=(self.Whi.shape[0],))

        for idx in range(x.shape[-2]):
            i = x_proj_i[..., idx, :] + curr_hidden @ self.Whi
            i = mx.sigmoid(i)

            f = x_proj_f[..., idx, :] + curr_hidden @ self.Whf
            f = mx.sigmoid(f)

            g = x_proj_g[..., idx, :] + curr_hidden @ self.Whg
            g = mx.tanh(g)

            o = x_proj_o[..., idx, :] + curr_hidden @ self.Who
            o = mx.sigmoid(o)

            curr_cell = f * curr_cell + i * g
            curr_hidden = o * mx.tanh(curr_cell)

            all_cell.append(curr_cell)
            all_hidden.append(curr_hidden)

        return mx.stack(all_hidden, axis=-2), mx.stack(all_cell, axis=-2)
