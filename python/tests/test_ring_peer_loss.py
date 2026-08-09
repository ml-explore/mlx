# Copyright © 2024 Apple Inc.

"""Regression test for a ring group losing a peer."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

# Enough iterations that the ranks are certainly mid-collective when one is killed,
# and small tensors so the loop is fast.
WORKER = textwrap.dedent("""
    import os, sys, time
    import mlx.core as mx

    world = mx.distributed.init(strict=True, backend="ring")
    x = mx.ones((64, 64), dtype=mx.float32) * (world.rank() + 1)
    print("ready", flush=True)
    while True:
        mx.eval(mx.distributed.all_sum(x, group=world))
        time.sleep(0.01)
    """)


def _free_ports(n):
    """Bind to port 0 to have the OS pick free ports, then release them.

    Racy in principle, but the alternative is a fixed port range that collides with
    anything else on the machine, which is worse on a shared CI runner.
    """
    socks = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        socks.append(s)
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


class TestRingPeerLoss(unittest.TestCase):
    N = 3
    START_TIMEOUT = 60
    EXIT_TIMEOUT = 60

    def test_survivors_do_not_hang_when_a_rank_dies(self):
        ports = _free_ports(self.N)
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hosts.json")
            with open(hostfile, "w") as f:
                json.dump([[f"127.0.0.1:{p}"] for p in ports], f)

            script = os.path.join(tmp, "worker.py")
            with open(script, "w") as f:
                f.write(WORKER)

            procs = []
            for rank in range(self.N):
                env = dict(
                    os.environ,
                    MLX_HOSTFILE=hostfile,
                    MLX_RANK=str(rank),
                    MLX_WORLD_SIZE=str(self.N),
                )
                procs.append(
                    subprocess.Popen(
                        [sys.executable, script],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                )

            try:
                # Wait for every rank to finish connecting. If the ring cannot form at
                # all here, that is an environment problem rather than a regression.
                deadline = time.time() + self.START_TIMEOUT
                for p in procs:
                    while True:
                        if time.time() > deadline:
                            self.skipTest("ring group did not start within the timeout")
                        line = p.stdout.readline()
                        if line.startswith("ready"):
                            break
                        if line == "" and p.poll() is not None:
                            self.skipTest(f"rank exited during startup: {p.returncode}")

                time.sleep(1.0)  # let a few collectives complete

                victim = procs[self.N - 1]
                victim.kill()
                victim.wait(timeout=self.EXIT_TIMEOUT)

                # The survivors must not be left waiting on a promise that nobody will
                # ever satisfy. Before the fix they stayed alive indefinitely: one
                # spinning at 100% CPU on a closed socket without logging anything, the
                # rest blocked in eval.
                deadline = time.time() + self.EXIT_TIMEOUT
                for p in procs[: self.N - 1]:
                    remaining = deadline - time.time()
                    self.assertGreater(
                        remaining, 0, "survivors were still running at the deadline"
                    )
                    try:
                        p.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        self.fail(
                            "a surviving rank did not exit after a peer was killed; "
                            "it is hung waiting on the dead peer"
                        )
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    if p.stdout is not None:
                        p.stdout.close()


if __name__ == "__main__":
    unittest.main()
