import numpy as np


def _tile(x, dims):
    s = x.shape
    if len(dims) > len(s):
        s = tuple(1 for _ in range(len(dims) - len(s))) + s
    # print(s)
    expand_shape = []
    broad_shape = []
    final_shape = []

    odims = len(dims) - len(s)
    for i in range(len(s)):
        if dims[i] != 1:
            expand_shape.append(1)
            broad_shape.append(dims[i])
        expand_shape.append(s[i])
        broad_shape.append(s[i])
        final_shape.append(dims[i])
        if odims > 0:
            odims -= 1
        else:
            final_shape[-1] *= s[i]

    # print(expand_shape)
    # print(broad_shape)
    # print(final_shape)

    x = np.reshape(x, expand_shape)
    x = np.broadcast_to(x, broad_shape)
    return np.reshape(x, final_shape)


x = np.array([1, 2, 3])
np.testing.assert_allclose(np.tile(x, [2, 2, 2]), _tile(x, [2, 2, 2]))
print(_tile(x, [2, 2, 2]).shape)
