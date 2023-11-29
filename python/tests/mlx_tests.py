import os
import unittest

import mlx.core as mx


class MLXTestCase(unittest.TestCase):
    def setUp(self):
        self.default = mx.default_device()
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(mx, device)
            mx.set_default_device(device)

    def tearDown(self):
        mx.set_default_device(self.default)
