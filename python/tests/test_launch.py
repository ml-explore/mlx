# Copyright © 2026 Apple Inc.

import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap

import mlx_tests


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SLOW_READS = textwrap.dedent("""
    import os, time

    read = os.read

    def slow_read(fd, n):
        data = read(fd, n)
        time.sleep(0.05)
        return data

    os.read = slow_read
    """)


def kill_children(directory):
    for name in os.listdir(directory):
        with open(os.path.join(directory, name)) as f:
            try:
                os.kill(int(f.read()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def launch(rank_code, slow_reads=False, timeout=60):
    launcher = SLOW_READS if slow_reads else ""
    launcher += textwrap.dedent("""
        from mlx._distributed_utils.launch import main

        raise SystemExit(main() or 0)
        """)
    return subprocess.run(
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
            rank_code,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestLaunch(mlx_tests.MLXTestCase):
    def test_output_written_right_before_exit_is_kept(self):
        # Slow down the launcher's reads so that each rank writes its last line
        # and exits while the reader thread is still handling an earlier chunk.
        rank = (
            "import sys, time; "
            "print('first', file=sys.stderr, flush=True); "
            "time.sleep(0.01); "
            "print('last', flush=True)"
        )
        result = launch(rank, slow_reads=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split().count("last"), 2, result.stdout)
        self.assertEqual(result.stderr.split().count("first"), 2, result.stderr)

    def test_rank_closes_pipes_before_exit(self):
        # The pipes reach EOF while the rank still runs. The launcher must wait
        # for the exit and see its status.
        rank = (
            "import os, time; "
            "print('last', flush=True); "
            "os.close(1); os.close(2); "
            "time.sleep(0.5)"
        )
        result = launch(rank)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split().count("last"), 2, result.stdout)
        self.assertNotIn("exited with code", result.stderr)

    def test_stop_reaches_rank_with_closed_pipes(self):
        # Rank 1 fails while rank 0 runs with both pipes at EOF. The launcher
        # must still terminate rank 0.
        rank = textwrap.dedent("""
            import os, sys, time

            rank = os.environ["MLX_RANK"]
            print("rank", rank, flush=True)
            if rank == "0":
                os.close(1)
                os.close(2)
                time.sleep(60)
            time.sleep(0.5)
            sys.exit(1)
            """)
        result = launch(rank, timeout=30)
        self.assertEqual(sorted(result.stdout.split()), ["0", "1", "rank", "rank"])
        self.assertIn("Node with rank 0 exited with code -15", result.stderr)
        self.assertIn("Node with rank 1 exited with code 1", result.stderr)

    def test_failed_rank_stops_the_others(self):
        # Rank 0 fails at once. The launcher must terminate rank 1 and keep the
        # output of both ranks.
        rank = textwrap.dedent("""
            import os, sys, time

            rank = os.environ["MLX_RANK"]
            print("rank", rank, flush=True)
            if rank == "0":
                sys.exit(1)
            time.sleep(60)
            """)
        result = launch(rank, timeout=30)
        self.assertEqual(sorted(result.stdout.split()), ["0", "1", "rank", "rank"])
        self.assertIn("Node with rank 0 exited with code 1", result.stderr)
        self.assertIn("Node with rank 1 exited with code -15", result.stderr)

    def test_child_holding_pipes_does_not_block_exit(self):
        # A child of the rank keeps the pipes open after the rank exits, like a
        # leftover process behind ssh. The launcher must not wait for it.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(kill_children, tmp.name)
        rank = textwrap.dedent(f"""
            import os, subprocess, sys

            sleeper = "import time; time.sleep(30)"
            child = subprocess.Popen([sys.executable, "-c", sleeper])
            with open(os.path.join({tmp.name!r}, os.environ["MLX_RANK"]), "w") as f:
                f.write(str(child.pid))
            print("last", flush=True)
            """)
        result = launch(rank, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split().count("last"), 2, result.stdout)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
