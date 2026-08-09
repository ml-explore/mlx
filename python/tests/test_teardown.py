# Copyright © 2026 Apple Inc.

"""Regression tests for crashes at thread and interpreter teardown.

These cannot be written as ordinary assertions: the failure mode is the process dying
(std::terminate, or SIGSEGV) during `__call_tls_dtors()`, so there is nothing left to
assert on. Each case therefore runs a small program in a subprocess and checks that it
exits cleanly.

The faults are intermittent on Linux, though reliable on Windows, so a single clean run
is not evidence of a fix. Each case therefore repeats.
"""

import subprocess
import sys
import unittest

import mlx_tests

# Two worker threads, each running a compiled function whose captured state includes
# mx.random.state, on a CPU stream. This is the shape that made PyKeySequence and the
# compile cache's ThreadCleanup destructors acquire the GIL from a thread_local
# destructor -- which CPython answers, when another thread is finalizing, by calling
# PyThread_exit_thread(). The forced unwind then crosses a noexcept destructor.
GIL_IN_TLS_DTOR = """
import threading
from functools import partial
import mlx.core as mx

def make_fn():
    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def f():
        return mx.random.uniform(shape=(10, 10))

    return f


fns = [make_fn() for _ in range(2)]

def worker(f):
    with mx.stream(mx.cpu):
        a = f()
        b = f()
        mx.eval(a, b)

for f in fns:
    t = threading.Thread(target=worker, args=(f,))
    t.start()
    t.join()
"""

# Sequential worker threads doing GPU work with a compiled function returning a tuple.
# This is what exposed the throwing CUDA destructors (~CommandEncoder calling
# synchronize(), ~CublasHandles, ~CudnnHandle): the *first* thread's teardown killed the
# process as join() returned. It also covers dropping a Python reference (the compiled
# function's output structure) from a thread_local destructor without the GIL.
THREADED_COMPILE_TEARDOWN = """
import gc
import threading
import mlx.core as mx

@mx.compile
def fun(x):
    return x + 1.0, x + 1.0

def worker():
    x = mx.array([1.0])
    y, z = fun(x)
    mx.eval(y, z)
    assert y.item() == 2.0 and z.item() == 2.0

for _ in range(3):
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    gc.collect()
"""


class TestTeardown(mlx_tests.MLXTestCase):
    def _run_repeatedly(self, program, runs=8):
        failures = []
        for i in range(runs):
            p = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if p.returncode != 0:
                failures.append((i, p.returncode, p.stderr[-2000:]))
        if failures:
            i, rc, err = failures[0]
            self.fail(
                f"{len(failures)}/{runs} runs died at teardown; "
                f"first was run {i} with exit {rc}:\n{err}"
            )

    def test_gil_not_acquired_from_thread_local_destructor(self):
        self._run_repeatedly(GIL_IN_TLS_DTOR)

    def test_threaded_compile_teardown(self):
        self._run_repeatedly(THREADED_COMPILE_TEARDOWN)


if __name__ == "__main__":
    unittest.main()
