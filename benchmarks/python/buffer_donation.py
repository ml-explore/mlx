import mlx.core as mx
from time_utils import time_fn


def time_unary_inplace():
    n = 8192
    x = mx.zeros((n, n, 16))
    def abs_inplace(x):
        for _ in range(20):
            x = mx.abs(x)
        return x

    time_fn(abs_inplace, x)

def time_binary_inplace():
    n = 8192
    x = mx.random.normal((n, n))
    y = mx.random.normal((n, n))

    def add_donate_first(x, y):
        for _ in range(20):
            x = x + y
        return x

    time_fn(add_donate_first, x, y)

    def add_donate_second(x, y):
        for _ in range(20):
            y = x + y
        return y

    time_fn(add_donate_second, x, y)

    def add_donate_first_second_transpose(x, y):
        for _ in range(20):
            x = x + y.T
        return x

    time_fn(add_donate_first_second_transpose, x, y)

    def add_first_transpose_donate_second(x, y):
        for _ in range(20):
            y = x.T + y
        return y

    time_fn(add_first_transpose_donate_second, x, y)


if __name__ == "__main__":
    time_unary_inplace()
    # time_binary_inplace()
