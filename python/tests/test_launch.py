# Copyright © 2026 Apple Inc.

import sys
import unittest

from mlx._distributed_utils.launch import RemoteProcess


class TestMakeLaunchScript(unittest.TestCase):
    def test_explicit_python_prepends_interpreter(self):
        script = RemoteProcess.make_launch_script(
            0,
            "/opt/venv/bin/python3",
            None,
            {},
            [],
            ["train.py", "--steps", "10"],
            True,
        )
        self.assertIn("cmd=(/opt/venv/bin/python3 train.py --steps 10); ", script)
        self.assertTrue(script.endswith('exec "${cmd[@]}"'))

    def test_default_keeps_command_verbatim(self):
        script = RemoteProcess.make_launch_script(
            0, None, None, {}, [], ["train.py", "--steps", "10"], True
        )
        self.assertIn("cmd=(train.py --steps 10); ", script)
        self.assertNotIn(sys.executable, script)

    def test_rank_is_exported(self):
        script = RemoteProcess.make_launch_script(
            3, None, None, {}, [], ["a.out"], True
        )
        self.assertIn("export MLX_RANK=3; ", script)


if __name__ == "__main__":
    unittest.main()
