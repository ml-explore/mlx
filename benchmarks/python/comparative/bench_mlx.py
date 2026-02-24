# Copyright © 2023 Apple Inc.

import argparse
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn


def int_or_list(x):
    try:
        return int(x)
    except ValueError:
        return [int(xi) for xi in x.split(",")]


def none_or_list(x):
    if x == "":
        return None
    else:
        return [int(xi) for xi in x.split(",")]


def dtype_from_str(x):
    if x == "":
        return mx.float32
    else:
        dt = getattr(mx, x)
        if not isinstance(dt, mx.Dtype):
            raise ValueError(f"{x} is not an mlx dtype")
        return dt


def bench(f, *args):
    for i in range(10):
        f(*args)

    s = time.perf_counter()
    for i in range(100):
        f(*args)
    e = time.perf_counter()
    return e - s


def matmul_square(x):
    y = x
    for i in range(10):
        y = y @ x
    mx.eval(y)
    return y


def matmul(x, y):
    ys = []
    for i in range(10):
        ys.append(x @ y)
    mx.eval(ys)


def _quant_matmul(x, w, s, b, transpose, group_size, bits):
    ys = []
    for i in range(10):
        ys.append(
            mx.quantized_matmul(
                x, w, s, b, transpose=transpose, group_size=group_size, bits=bits
            )
        )
    mx.eval(ys)


quant_matmul = {
    "quant_matmul_32_1": partial(_quant_matmul, transpose=False, group_size=32, bits=1),
    "quant_matmul_32_2": partial(_quant_matmul, transpose=False, group_size=32, bits=2),
    "quant_matmul_32_4": partial(_quant_matmul, transpose=False, group_size=32, bits=4),
    "quant_matmul_32_8": partial(_quant_matmul, transpose=False, group_size=32, bits=8),
    "quant_matmul_64_1": partial(_quant_matmul, transpose=False, group_size=64, bits=1),
    "quant_matmul_64_2": partial(_quant_matmul, transpose=False, group_size=64, bits=2),
    "quant_matmul_64_4": partial(_quant_matmul, transpose=False, group_size=64, bits=4),
    "quant_matmul_64_8": partial(_quant_matmul, transpose=False, group_size=64, bits=8),
    "quant_matmul_128_1": partial(
        _quant_matmul, transpose=False, group_size=128, bits=1
    ),
    "quant_matmul_128_2": partial(
        _quant_matmul, transpose=False, group_size=128, bits=2
    ),
    "quant_matmul_128_4": partial(
        _quant_matmul, transpose=False, group_size=128, bits=4
    ),
    "quant_matmul_128_8": partial(
        _quant_matmul, transpose=False, group_size=128, bits=8
    ),
    "quant_matmul_t_32_1": partial(
        _quant_matmul, transpose=True, group_size=32, bits=1
    ),
    "quant_matmul_t_32_2": partial(
        _quant_matmul, transpose=True, group_size=32, bits=2
    ),
    "quant_matmul_t_32_4": partial(
        _quant_matmul, transpose=True, group_size=32, bits=4
    ),
    "quant_matmul_t_32_8": partial(
        _quant_matmul, transpose=True, group_size=32, bits=8
    ),
    "quant_matmul_t_64_1": partial(
        _quant_matmul, transpose=True, group_size=64, bits=1
    ),
    "quant_matmul_t_64_2": partial(
        _quant_matmul, transpose=True, group_size=64, bits=2
    ),
    "quant_matmul_t_64_4": partial(
        _quant_matmul, transpose=True, group_size=64, bits=4
    ),
    "quant_matmul_t_64_8": partial(
        _quant_matmul, transpose=True, group_size=64, bits=8
    ),
    "quant_matmul_t_128_1": partial(
        _quant_matmul, transpose=True, group_size=128, bits=1
    ),
    "quant_matmul_t_128_2": partial(
        _quant_matmul, transpose=True, group_size=128, bits=2
    ),
    "quant_matmul_t_128_4": partial(
        _quant_matmul, transpose=True, group_size=128, bits=4
    ),
    "quant_matmul_t_128_8": partial(
        _quant_matmul, transpose=True, group_size=128, bits=8
    ),
}


def conv1d(x, y):
    ys = []
    for i in range(10):
        ys.append(mx.conv1d(x, y))
    mx.eval(ys)


def conv2d(x, y):
    ys = []
    for i in range(10):
        ys.append(mx.conv2d(x, y))
    mx.eval(ys)


def binary(op, x, y):
    for i in range(100):
        y = getattr(mx, op)(x, y)
    mx.eval(y)


def reduction(op, axis, x):
    ys = []
    for i in range(100):
        ys.append(getattr(mx, op)(x, axis=axis))
    mx.eval(ys)


def sum_and_add(axis, x, y):
    z = x.sum(axis=axis, keepdims=True)
    for i in range(50):
        z = (z + y).sum(axis=axis, keepdims=True)
    mx.eval(z)


def softmax(axis, x):
    ys = []
    for i in range(100):
        ex = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
        y = ex / mx.sum(ex, axis=axis, keepdims=True)
        ys.append(y)
    mx.eval(ys)


def softmax_fused(axis, x):
    ys = []
    for i in range(100):
        y = mx.softmax(x, axis=axis)
        ys.append(y)
    mx.eval(ys)


def relu(x):
    y = x
    for i in range(100):
        y = nn.relu(y)
    mx.eval(y)


def leaky_relu(x: mx.array):
    y = x
    for i in range(100):
        y = nn.leaky_relu(y)
    mx.eval(y)


def prelu(x: mx.array):
    y = x
    for i in range(100):
        y = nn.prelu(y, mx.ones(1))
    mx.eval(y)


def softplus(x: mx.array):
    y = x
    for i in range(100):
        y = nn.softplus(y)
    mx.eval(y)


def mish(x: mx.array):
    y = x
    for i in range(100):
        y = nn.mish(y)
    mx.eval(y)


def leaky_relu(x):
    y = x
    for i in range(100):
        y = nn.leaky_relu(y)
    mx.eval(y)


def elu(x):
    y = x
    for i in range(100):
        y = nn.elu(y)
    mx.eval(y)


def relu6(x):
    y = x
    for i in range(100):
        y = nn.relu6(y)
    mx.eval(y)


def softplus(x):
    y = x
    for i in range(100):
        y = nn.softplus(y)
    mx.eval(y)


def celu(x):
    y = x
    for i in range(100):
        y = nn.celu(y)
    mx.eval(y)


def log_sigmoid(x):
    y = x
    for i in range(100):
        y = nn.log_sigmoid(y)
    mx.eval(y)


def scalar_mult(x):
    y = x
    for i in range(100):
        y = y * (1.0 / (1 + i))
    mx.eval(y)


def cross_entropy(targets, x):
    ys = []
    for i in range(100):
        y = mx.logsumexp(x, axis=-1, keepdims=True) - mx.take_along_axis(
            x, mx.reshape(targets, (-1, 1)), axis=-1
        )
        ys.append(mx.mean(y))
    mx.eval(ys)


def logsumexp(axis, x):
    ys = []
    for i in range(100):
        ys.append(mx.logsumexp(x, axis=axis))
    mx.eval(ys)


def linear(w, b, x):
    ys = []
    for i in range(10):
        ys.append(x @ mx.transpose(w, (1, 0)) + b)
    mx.eval(ys)


def linear_fused(w, b, x):
    ys = []
    for i in range(10):
        ys.append(mx.addmm(b, x, mx.transpose(w, (1, 0))))
    mx.eval(ys)


def rope(x):
    *_, N, D = x.shape
    ys = []
    for i in range(10):
        shape = x.shape
        x = mx.reshape(x, (-1, N, D))
        positions = mx.arange(N)
        freqs = mx.exp(mx.arange(0.0, D // 2) / math.log(10000 / (D // 2 - 1)))
        theta = mx.reshape(positions, (-1, 1)) * mx.reshape(freqs, (1, -1))
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        y = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1)
        y = mx.reshape(y, (-1, N, D))
        ys.append(y)
    mx.eval(ys)


def concatenate(axis, x, y):
    ys = []
    for i in range(10):
        ys.append(mx.concatenate([x, y], axis=axis))
    mx.eval(ys)


def cumsum(axis, x):
    ys = []
    for i in range(10):
        ys.append(mx.cumsum(x, axis))
    mx.eval(ys)


def sort(axis, x):
    ys = []
    for i in range(10):
        ys.append(mx.sort(x, axis))
    mx.eval(ys)


def topk(axis, x):
    k = x.shape[axis] // 3
    ys = []
    for i in range(10):
        ys.append(mx.topk(x, k, axis))
    mx.eval(ys)


def step_function(x):
    y = x
    for i in range(100):
        y = nn.step(x)
    mx.eval(y)


def selu(x):
    y = x
    for i in range(100):
        y = nn.selu(x)
    mx.eval(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", help="Choose the benchmark to run")
    parser.add_argument(
        "--size",
        default=[(1024, 1024)],
        type=lambda x: list(map(int, x.split("x"))),
        help="Set the matrix size",
        action="append",
    )
    parser.add_argument(
        "--axis",
        default=[1],
        type=int_or_list,
        help="Set a reduction axis",
        action="append",
    )
    parser.add_argument(
        "--transpose",
        type=none_or_list,
        default=[],
        help="Permute the matrix",
        action="append",
    )
    parser.add_argument(
        "--print-pid", action="store_true", help="Print the PID and pause"
    )
    parser.add_argument("--cpu", action="store_true", help="Use the CPU")
    parser.add_argument(
        "--fused", action="store_true", help="Use fused functions where possible"
    )
    parser.add_argument("--dtype", type=dtype_from_str, default=[], action="append")

    args = parser.parse_args()

    if len(args.size) > 1:
        args.size.pop(0)
    if len(args.axis) > 1:
        args.axis.pop(0)

    if args.cpu:
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    types = args.dtype
    if not types:
        types = [mx.float32]
    if len(types) < len(args.size):
        types = types + [types[0]] * (len(args.size) - len(types))

    xs = []
    for size, dtype in zip(args.size, types):
        xs.append(mx.random.normal(size).astype(dtype))
    for i, t in enumerate(args.transpose):
        if t is None:
            continue
        xs[i] = mx.transpose(xs[i], t)
    mx.eval(xs)
    x = xs[0]
    axis = args.axis[0]

    if args.print_pid:
        print(os.getpid())
        input("Press enter to run")

    if args.benchmark == "matmul_square":
        print(bench(matmul_square, x))

    elif args.benchmark == "matmul":
        print(bench(matmul, *xs))

    elif args.benchmark.startswith("quant_matmul"):
        # Parse group_size and bits from the benchmark name, e.g.
        # "quant_matmul_128_4" or "quant_matmul_t_128_4"
        fn = quant_matmul[args.benchmark]
        gs = fn.keywords["group_size"]
        bits = fn.keywords["bits"]
        transpose = fn.keywords["transpose"]

        # xs[0] = activation x, xs[1] = original (float) weight matrix
        # Quantize the weight internally so the caller only needs:
        #   --size MxK --size NxK  (transpose=True)  or  --size MxK --size KxN
        w_float = xs[1].astype(mx.float16)
        w_q, scales, biases = mx.quantize(w_float, group_size=gs, bits=bits)
        mx.eval(w_q, scales, biases)
        x_input = xs[0].astype(mx.float16)
        mx.eval(x_input)
        print(bench(_quant_matmul, x_input, w_q, scales, biases, transpose, gs, bits))

    elif args.benchmark == "linear":
        if args.fused:
            print(bench(linear_fused, *xs))
        else:
            print(bench(linear, *xs))

    elif args.benchmark == "sum_axis":
        print(bench(reduction, "sum", axis, x))

    elif args.benchmark == "sum_all":
        print(bench(reduction, "sum", None, x))

    elif args.benchmark == "argmax":
        print(bench(reduction, "argmax", axis, x))

    elif args.benchmark == "add":
        print(bench(binary, "add", *xs))

    elif args.benchmark == "mul":
        print(bench(binary, "multiply", *xs))

    elif args.benchmark == "softmax":
        if args.fused:
            print(bench(softmax_fused, axis, x))
        else:
            print(bench(softmax, axis, x))

    elif args.benchmark == "relu":
        print(bench(relu, x))

    elif args.benchmark == "elu":
        print(bench(elu, x))

    elif args.benchmark == "relu6":
        print(bench(relu6, x))

    elif args.benchmark == "celu":
        print(bench(celu, x))

    elif args.benchmark == "log_sigmoid":
        print(bench(log_sigmoid, x))

    elif args.benchmark == "leaky_relu":
        print(bench(leaky_relu, x))
    elif args.benchmark == "prelu":
        print(bench(prelu, x))
    elif args.benchmark == "softplus":
        print(bench(softplus, x))
    elif args.benchmark == "mish":
        print(bench(mish, x))
    elif args.benchmark == "scalar_mul":
        print(bench(scalar_mult, x))

    elif args.benchmark == "cross_entropy":
        if len(size) != 2:
            raise ValueError("Error: [cross_entropy] benchmark requires a 2 dim size")

        targets = mx.zeros((len(x),), dtype=mx.uint32)
        print(bench(cross_entropy, targets, x))

    elif args.benchmark == "logsumexp":
        print(bench(logsumexp, axis, x))

    elif args.benchmark == "rope":
        print(bench(rope, x))

    elif args.benchmark == "concatenate":
        print(bench(concatenate, axis, *xs))

    elif args.benchmark == "cumsum":
        print(bench(cumsum, axis, *xs))

    elif args.benchmark == "conv1d":
        print(bench(conv1d, *xs))

    elif args.benchmark == "conv2d":
        print(bench(conv2d, *xs))

    elif args.benchmark == "sort":
        print(bench(sort, axis, x))

    elif args.benchmark == "topk":
        print(bench(topk, axis, x))

    elif args.benchmark == "step":
        print(bench(step_function, x))

    elif args.benchmark == "selu":
        print(bench(selu, x))

    elif args.benchmark == "sum_and_add":
        print(bench(sum_and_add, axis, *xs))

    else:
        raise ValueError("Unknown benchmark")
