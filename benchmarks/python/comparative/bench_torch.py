# Copyright © 2023 Apple Inc.

import argparse
import os
import time

import torch
import torch.mps


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
        return torch.float32
    else:
        dt = getattr(torch, x)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{x} is not a torch dtype")
        return dt


def bench(f, *args):
    for i in range(10):
        f(*args)

    s = time.time()
    for i in range(100):
        f(*args)
    e = time.time()
    return e - s


def sync_if_needed(x):
    if x.device != torch.device("cpu"):
        torch.mps.synchronize()


@torch.no_grad()
def matmul_square(x):
    y = x
    for i in range(10):
        y = y @ x
    sync_if_needed(x)


@torch.no_grad()
def matmul(x, y):
    ys = []
    for i in range(10):
        ys.append(x @ y)
    sync_if_needed(x)


@torch.no_grad()
def conv1d(x, y):
    x = torch.transpose(x, -1, -2)
    y = torch.transpose(y, -1, -2)
    ys = []
    for i in range(10):
        ys.append(torch.nn.functional.conv1d(x, y))
    sync_if_needed(x)


@torch.no_grad()
def conv2d(x, y):
    x = torch.permute(x, (0, 3, 1, 2))
    y = torch.permute(y, (0, 3, 1, 2))
    ys = []
    for i in range(10):
        ys.append(torch.nn.functional.conv2d(x, y))
    sync_if_needed(x)


@torch.no_grad()
def binary(op, x, y):
    for i in range(100):
        y = getattr(torch, op)(x, y)
    sync_if_needed(x)


@torch.no_grad()
def reduction(op, axis, x):
    ys = []
    for i in range(100):
        ys.append(getattr(x, op)(axis))
    sync_if_needed(x)


@torch.no_grad()
def softmax(axis, x):
    ys = []
    for i in range(100):
        ex = torch.exp(x - torch.max(x, dim=axis, keepdims=True).values)
        y = ex / torch.sum(ex, dim=axis, keepdims=True)
        ys.append(y)
    sync_if_needed(x)


@torch.no_grad()
def softmax_fused(axis, x):
    ys = []
    for i in range(100):
        ys.append(torch.nn.functional.softmax(x, dim=axis))
    sync_if_needed(x)


@torch.no_grad()
def relu(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.relu(y)
    sync_if_needed(x)


@torch.no_grad()
def leaky_relu(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.leaky_relu(y)
    sync_if_needed(x)


@torch.no_grad()
def elu(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.elu(y)
    sync_if_needed(x)


@torch.no_grad()
def celu(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.celu(y)
    sync_if_needed(x)


@torch.no_grad()
def relu6(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.relu6(y)
    sync_if_needed(x)


@torch.no_grad()
def softplus(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.softplus(y)
    sync_if_needed(x)


@torch.no_grad()
def log_sigmoid(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.logsigmoid(y)
    sync_if_needed(x)


@torch.no_grad()
def prelu(x: torch.Tensor) -> torch.Tensor:
    y = x
    for _ in range(100):
        y = torch.nn.functional.prelu(y, torch.ones(1).to(y.device))
    sync_if_needed(x)


@torch.no_grad()
def mish(x: torch.Tensor) -> torch.Tensor:
    y = x
    for _ in range(100):
        return torch.nn.functional.mish(y)
    sync_if_needed(x)


@torch.no_grad()
def scalar_mult(x):
    y = x
    for i in range(100):
        y = y * (1.0 / (1 + i))
    sync_if_needed(x)


@torch.no_grad()
def cross_entropy(targets, x):
    ys = []
    for i in range(100):
        ys.append(torch.nn.functional.cross_entropy(x, targets))
    sync_if_needed(x)


@torch.no_grad()
def logsumexp(axis, x):
    ys = []
    for i in range(100):
        ys.append(torch.logsumexp(x, dim=axis))
    sync_if_needed(x)


@torch.no_grad()
def linear_fused(w, b, x):
    ys = []
    for i in range(10):
        ys.append(torch.nn.functional.linear(x, w, b))
    sync_if_needed(x)


@torch.no_grad()
def linear(w, b, x):
    ys = []
    for i in range(10):
        ys.append((x @ torch.transpose(w, -2, -1)) + b)
    sync_if_needed(x)


@torch.no_grad()
def rope(x):
    *_, N, D = x.shape
    ys = []
    for i in range(10):
        x = x.view(-1, N, D)
        positions = torch.arange(N, device=x.device)
        freqs = 10000 ** torch.linspace(0, 1, D // 2, device=x.device)
        theta = positions[:, None] * freqs[None]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        y = torch.cat([rx1[..., None], rx2[..., None]], dim=-1)
        y = y.reshape(-1, N, D)
        ys.append(y)
    sync_if_needed(x)


@torch.no_grad()
def concatenate(axis, x, y):
    ys = []
    for i in range(10):
        ys.append(torch.cat([x, y], dim=axis))
    sync_if_needed(x)


@torch.no_grad()
def cumsum(axis, x):
    ys = []
    for i in range(10):
        ys.append(x.cumsum(axis))
    sync_if_needed(x)


@torch.no_grad()
def sort(axis, x):
    ys = []
    for i in range(10):
        ys.append(torch.sort(x, dim=axis)[0])
    sync_if_needed(x)


@torch.no_grad()
def topk(axis, x):
    k = x.shape[axis] // 3
    ys = []
    for i in range(10):
        ys.append(torch.topk(x, k, dim=axis)[0])
    sync_if_needed(x)


@torch.no_grad()
def selu(x):
    y = x
    for i in range(100):
        y = torch.nn.functional.selu(y)
    sync_if_needed(x)


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

    if args.print_pid:
        print(os.getpid())
        input("Press enter to run")

    torch.set_num_threads(1)
    device = "cpu" if args.cpu else "mps"

    types = args.dtype
    if not types:
        types = [torch.float32]
    if len(types) < len(args.size):
        types = types + [types[0]] * (len(args.size) - len(types))

    xs = []
    for size, dtype in zip(args.size, types):
        xs.append(torch.randn(*size).to(device).to(dtype))
    for i, t in enumerate(args.transpose):
        if t is None:
            continue
        xs[i] = xs[i].permute(*t)
    x = xs[0]
    axis = args.axis[0]

    if args.benchmark == "matmul_square":
        print(bench(matmul_square, x))

    elif args.benchmark == "matmul":
        print(bench(matmul, *xs))

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
        print(bench(binary, "mul", *xs))

    elif args.benchmark == "softmax":
        if args.fused:
            print(bench(softmax_fused, axis, x))
        else:
            print(bench(softmax, axis, x))

    elif args.benchmark == "relu":
        print(bench(relu, x))

    elif args.benchmark == "leaky_relu":
        print(bench(leaky_relu, x))

    elif args.benchmark == "elu":
        print(bench(elu, x))

    elif args.benchmark == "relu6":
        print(bench(relu6, x))

    elif args.benchmark == "softplus":
        print(bench(softplus, x))

    elif args.benchmark == "celu":
        print(bench(celu, x))

    elif args.benchmark == "log_sigmoid":
        print(bench(log_sigmoid, x))

    elif args.benchmark == "prelu":
        print(bench(prelu, x))
    elif args.benchmark == "mish":
        print(bench(mish, x))
    elif args.benchmark == "scalar_mul":
        print(bench(scalar_mult, x))

    elif args.benchmark == "cross_entropy":
        if len(size) != 2:
            raise ValueError("Error: [cross_entropy] benchmark requires a 2 dim size")

        targets = torch.zeros(len(x), dtype=torch.long).to(x.device)
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

    else:
        raise ValueError("Unknown benchmark")
