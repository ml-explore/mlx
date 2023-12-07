import unittest
from typing import List, Callable, Tuple
import torch
import numpy
import mlx.core as mx
import time

rng = numpy.random.default_rng(0)
dtype = numpy.float32
torch_device = "mps" if torch.backends.mps.is_available() else "cpu"


def helper_test_speed(f: Callable, *args: List):
    ets = []
    ret = None
    for _ in range(8):
        del ret
        # no caching
        [x.cpu().numpy() if isinstance(x, torch.Tensor) else x.tolist() for x in args]
        st = time.perf_counter()
        ret = f(*args)
        # TODO: is there a better way to force sync?
        ret.cpu().numpy() if isinstance(ret, torch.Tensor) else ret.tolist()
        et = time.perf_counter()
        ets.append((et - st) * 1000)
    return ret.cpu().numpy() if isinstance(
        ret, torch.Tensor
    ) else ret.tolist(), numpy.min(ets)


def report_results(
    name: str,
    mx_res: numpy.ndarray,
    mx_time: float,
    torch_res: numpy.ndarray,
    torch_time: float,
):
    assert numpy.allclose(mx_res, torch_res, rtol=1e-3, atol=1e-3)
    speedup = torch_time / mx_time
    color = "\033[92m" if speedup > 1.0 else "\033[91m"
    print(
        f"name: {name:30s} torch: {torch_time:.3f} ms, mlx: {mx_time:.3f} ms, speedup: {color}{speedup:.3f}x \033[39m"
    )


def helper_test_generic(
    name: str, np_data: List[numpy.ndarray], f: Callable, f_torch: Callable
):
    with torch.no_grad():
        torch_res, torch_time = helper_test_speed(
            f_torch, *[torch.from_numpy(x).to(torch_device) for x in np_data]
        )

    mx_res, mx_time = helper_test_speed(
        f, *[mx.array(x, dtype=mx.float32) for x in np_data]
    )
    report_results(name, mx_res, mx_time, torch_res, torch_time)


def helper_test_generic_op(
    name: str, size: Tuple[int] | int, f: Callable, f_torch: Callable = None, count=2
):
    if f_torch is None:
        f_torch = f
    np_data = [rng.random(size=size, dtype=dtype) for _ in range(count)]
    helper_test_generic(name, np_data, f, f_torch)


def helper_test_matvec(name: str, size: Tuple[int] | int):
    f = lambda a, b: a @ b
    data = [
        rng.random(size=(size[0]), dtype=dtype) - 0.5,
        rng.random(size=size, dtype=dtype) - 0.5,
    ]
    helper_test_generic(name, data, f, f)


class TestSpeed(unittest.TestCase):
    def test_add(self):
        helper_test_generic_op("add", 8192, lambda a, b: a + b)

    def test_sub(self):
        helper_test_generic_op("sub", 8192, lambda a, b: a - b)

    def test_sum(self):
        helper_test_generic_op("sum", 8192, lambda a: a.sum(), count=1)

    def test_partial_sum(self):
        helper_test_generic_op("partial_sum", 8192, lambda a: a.sum(0), count=1)

    def test_exp(self):
        helper_test_generic_op("exp", 8192, lambda a: a.exp(), count=1)

    def test_mul_sum(self):
        helper_test_generic_op("mul_sum", 8192, lambda a, b: (a * b).sum())

    def test_max(self):
        helper_test_generic_op("max", 8192, lambda a: a.max(), count=1)

    def test_gemm_2048(self):
        helper_test_generic_op("gemm", 2048, lambda a, b: a @ b)

    def test_gemm_4096(self):
        helper_test_generic_op("gemm", 4096, lambda a, b: a @ b)

    def test_matvec_4096_16384(self):
        helper_test_matvec("matvec_4096_16384", (4096, 16384))

    def test_matvec_16384_4096(self):
        helper_test_matvec("matvec_16384_4096", (16384, 4096))


if __name__ == "__main__":
    unittest.main()
