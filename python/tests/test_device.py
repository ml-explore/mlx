# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


# Don't inherit from MLXTestCase to avoid call to setUp
class TestDefaultDevice(unittest.TestCase):
    def test_mlx_default_device(self):
        device = mx.default_device()
        if mx.metal.is_available():
            self.assertEqual(device, mx.Device(mx.gpu))
            self.assertEqual(str(device), "Device(gpu, 0)")
            self.assertEqual(device, mx.gpu)
            self.assertEqual(mx.gpu, device)
        else:
            self.assertEqual(device.type, mx.Device(mx.cpu))
            with self.assertRaises(ValueError):
                mx.set_default_device(mx.gpu)


class TestDevice(mlx_tests.MLXTestCase):
    def test_device(self):
        device = mx.default_device()

        cpu = mx.Device(mx.cpu)
        mx.set_default_device(cpu)
        self.assertEqual(mx.default_device(), cpu)
        self.assertEqual(str(cpu), "Device(cpu, 0)")

        mx.set_default_device(mx.cpu)
        self.assertEqual(mx.default_device(), mx.cpu)
        self.assertEqual(cpu, mx.cpu)
        self.assertEqual(mx.cpu, cpu)

        # Restore device
        mx.set_default_device(device)

    def test_op_on_device(self):
        x = mx.array(1.0)
        y = mx.array(1.0)

        a = mx.add(x, y, stream=None)
        b = mx.add(x, y, stream=mx.default_device())
        self.assertEqual(a.item(), b.item())
        b = mx.add(x, y, stream=mx.cpu)
        self.assertEqual(a.item(), b.item())

        if mx.metal.is_available():
            b = mx.add(x, y, stream=mx.gpu)
            self.assertEqual(a.item(), b.item())


class TestStream(mlx_tests.MLXTestCase):
    def test_stream(self):
        s1 = mx.default_stream(mx.default_device())
        self.assertEqual(s1.device, mx.default_device())

        s2 = mx.new_stream(mx.default_device())
        self.assertEqual(s2.device, mx.default_device())
        self.assertNotEqual(s1, s2)

        if mx.metal.is_available():
            s_gpu = mx.default_stream(mx.gpu)
            self.assertEqual(s_gpu.device, mx.gpu)
        else:
            with self.assertRaises(ValueError):
                mx.default_stream(mx.gpu)

        s_cpu = mx.default_stream(mx.cpu)
        self.assertEqual(s_cpu.device, mx.cpu)

        s_cpu = mx.new_stream(mx.cpu)
        self.assertEqual(s_cpu.device, mx.cpu)

        if mx.metal.is_available():
            s_gpu = mx.new_stream(mx.gpu)
            self.assertEqual(s_gpu.device, mx.gpu)
        else:
            with self.assertRaises(ValueError):
                mx.new_stream(mx.gpu)

    def test_op_on_stream(self):
        x = mx.array(1.0)
        y = mx.array(1.0)

        a = mx.add(x, y, stream=mx.default_stream(mx.default_device()))

        if mx.metal.is_available():
            b = mx.add(x, y, stream=mx.default_stream(mx.gpu))
            self.assertEqual(a.item(), b.item())
            s_gpu = mx.new_stream(mx.gpu)
            b = mx.add(x, y, stream=s_gpu)
            self.assertEqual(a.item(), b.item())

        b = mx.add(x, y, stream=mx.default_stream(mx.cpu))
        self.assertEqual(a.item(), b.item())
        s_cpu = mx.new_stream(mx.cpu)
        b = mx.add(x, y, stream=s_cpu)
        self.assertEqual(a.item(), b.item())


if __name__ == "__main__":
    unittest.main()
