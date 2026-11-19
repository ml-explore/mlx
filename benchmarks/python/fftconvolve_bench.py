import argparse

import mlx.core as mx
import numpy as np
from time_utils import time_fn


def run(device):
    mx.set_default_device(getattr(mx, device))
    rng = np.random.default_rng(811)
    print(f"device={device}, dtype=float32, mode=full")
    for n, k in ((1024, 3), (1024, 127), (16384, 1024), (16384, 4096)):
        a = mx.array(rng.normal(size=n).astype(np.float32))
        b = mx.array(rng.normal(size=k).astype(np.float32))
        mx.eval(a, b)
        direct = mx.convolve(a, b)
        fft = mx.fft.fftconvolve(a, b)
        np.testing.assert_allclose(fft, direct, atol=2e-4, rtol=2e-4)
        print(f"input={n}, kernel={k}")
        time_fn(mx.convolve, a, b)
        time_fn(mx.fft.fftconvolve, a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    run(parser.parse_args().device)
