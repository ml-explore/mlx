# Copyright © 2026 Apple Inc.

import socket
import subprocess
import sys
import textwrap
import unittest

import mlx_tests


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLaunch(mlx_tests.MLXTestCase):

    def test_output_written_right_before_exit_is_kept(self):
        # Slow down the launcher's reads so that each rank writes its last line
        # and exits while the reader thread is still handling an earlier chunk.
        launcher = textwrap.dedent("""
            import os, time

            read = os.read

            def slow_read(fd, n):
                data = read(fd, n)
                time.sleep(0.05)
                return data

            os.read = slow_read

            from mlx._distributed_utils.launch import main

            raise SystemExit(main() or 0)
            """)
        rank = (
            "import sys, time; "
            "print('first', file=sys.stderr, flush=True); "
            "time.sleep(0.01); "
            "print('last', flush=True)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                launcher,
                "--backend",
                "ring",
                "--hosts",
                "127.0.0.1",
                "--repeat-hosts",
                "2",
                "--starting-port",
                str(free_port()),
                "--",
                sys.executable,
                "-c",
                rank,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split().count("last"), 2, result.stdout)
        self.assertEqual(result.stderr.split().count("first"), 2, result.stderr)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
