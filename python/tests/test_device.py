# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


# Don't inherit from MLXTestCase to avoid call to setUp
class TestDefaultDevice(unittest.TestCase):
    def test_mlx_default_device(self):
        device = mx.default_device()
        if mx.is_available(mx.gpu):
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

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_device_context(self):
        default = mx.default_device()
        diff = mx.cpu if default == mx.gpu else mx.gpu
        self.assertNotEqual(default, diff)
        with mx.stream(diff):
            a = mx.add(mx.zeros((2, 2)), mx.ones((2, 2)))
            mx.eval(a)
            self.assertEqual(mx.default_device(), diff)
        self.assertEqual(mx.default_device(), default)

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

        if mx.is_available(mx.gpu):
            s_gpu = mx.default_stream(mx.gpu)
            self.assertEqual(s_gpu.device, mx.gpu)
        else:
            with self.assertRaises(ValueError):
                mx.default_stream(mx.gpu)

        s_cpu = mx.default_stream(mx.cpu)
        self.assertEqual(s_cpu.device, mx.cpu)

        s_cpu = mx.new_stream(mx.cpu)
        self.assertEqual(s_cpu.device, mx.cpu)

        if mx.is_available(mx.gpu):
            s_gpu = mx.new_stream(mx.gpu)
            self.assertEqual(s_gpu.device, mx.gpu)
        else:
            with self.assertRaises(ValueError):
                mx.new_stream(mx.gpu)

    def test_stream_api(self):
        stream = mx.Stream(mx.cpu)
        self.assertEqual(stream.device, mx.cpu)
        self.assertTrue(stream.query())
        stream.synchronize()
        self.assertTrue(stream.query())

        default_stream = mx.Stream()
        self.assertEqual(default_stream.device, mx.default_device())
        with self.assertRaises(ValueError):
            stream.record_event()

        other = mx.Stream(mx.cpu)
        stream.wait_stream(other)
        stream.synchronize()

        default_device = mx.default_device()
        default_stream = mx.default_stream(default_device)
        with stream as current:
            self.assertIs(current, stream)
            self.assertEqual(mx.default_device(), mx.cpu)
            self.assertEqual(mx.default_stream(mx.cpu), stream)
        self.assertEqual(mx.default_device(), default_device)
        self.assertEqual(mx.default_stream(default_device), default_stream)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_gpu_stream_api(self):
        producer = mx.Stream(mx.gpu)
        consumer = mx.Stream(mx.gpu)
        self.assertTrue(producer.query())
        self.assertTrue(consumer.query())

        event = producer.record_event()
        self.assertIsInstance(event, mx.Event)
        same_event = mx.Event()
        self.assertIs(producer.record_event(same_event), same_event)

        consumer.wait_event(event)
        self.assertFalse(consumer.query())
        consumer.synchronize()
        self.assertTrue(consumer.query())

        consumer.wait_stream(producer)
        self.assertFalse(consumer.query())
        consumer.synchronize()
        self.assertTrue(consumer.query())

    def test_op_on_stream(self):
        x = mx.array(1.0)
        y = mx.array(1.0)

        a = mx.add(x, y, stream=mx.default_stream(mx.default_device()))

        if mx.is_available(mx.gpu):
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


class TestEvent(mlx_tests.MLXTestCase):
    def test_event_requires_gpu(self):
        with self.assertRaises(TypeError):
            mx.Event(mx.gpu, True)
        with self.assertRaises(ValueError):
            mx.Event(mx.cpu)
        with self.assertRaises(ValueError):
            mx.Event(mx.cpu, enable_timing=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_event_synchronization(self):
        record_stream = mx.new_stream(mx.gpu)
        wait_stream = mx.new_stream(mx.gpu)
        event = mx.Event()

        self.assertTrue(event.query())
        event.wait(wait_stream)
        event.synchronize()

        out = mx.exp(
            mx.arange(1 << 20, dtype=mx.float32, stream=record_stream),
            stream=record_stream,
        )
        mx.async_eval(out)
        event.record(record_stream)
        event.wait(wait_stream)
        waited_out = mx.exp(
            mx.arange(1 << 20, dtype=mx.float32, stream=wait_stream),
            stream=wait_stream,
        )
        mx.async_eval(waited_out)
        mx.synchronize(wait_stream)

        self.assertTrue(event.query())
        event.record(record_stream)
        event.synchronize()
        self.assertTrue(event.query())

        end_event = mx.Event()
        end_event.record(record_stream)
        with self.assertRaises(RuntimeError):
            event.elapsed_time(end_event)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_event(self):
        start = mx.Event(mx.gpu, enable_timing=True)
        end = mx.Event(mx.gpu, enable_timing=True)
        start.record()
        out = mx.exp(
            mx.arange(1 << 20, dtype=mx.float32, stream=mx.gpu),
            stream=mx.gpu,
        )
        mx.async_eval(out)
        end.record()
        self.assertGreater(start.elapsed_time(end), 0.0)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_unrecorded_event(self):
        with self.assertRaises(RuntimeError):
            mx.Event(mx.gpu, enable_timing=True).elapsed_time(
                mx.Event(mx.gpu, enable_timing=True)
            )


class TestDeviceInfo(mlx_tests.MLXTestCase):
    def test_device_count(self):
        cpu_count = mx.device_count(mx.cpu)
        self.assertIsInstance(cpu_count, int)
        self.assertEqual(cpu_count, 1)

        gpu_count = mx.device_count(mx.gpu)
        self.assertIsInstance(gpu_count, int)
        self.assertGreaterEqual(gpu_count, 0)

    def test_device_info_cpu(self):
        info = mx.device_info(mx.cpu)
        self.assertIsInstance(info, dict)
        self.assertIn("device_name", info)
        self.assertTrue(len(info["device_name"]) > 0)
        self.assertIn("architecture", info)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_device_info_gpu(self):
        gpu_count = mx.device_count(mx.gpu)
        for i in range(gpu_count):
            info = mx.device_info(mx.Device(mx.gpu, i))
            self.assertIsInstance(info, dict)
            self.assertIn("device_name", info)
            self.assertTrue(len(info["device_name"]) > 0)
            self.assertIn("architecture", info)

    def test_device_info_default(self):
        info = mx.device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("device_name", info)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
