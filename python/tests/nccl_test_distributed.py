# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestNCCLDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="nccl")
        cls.atol = 1e-4
        cls.rtol = 1e-4

    def test_reduce_scatter(self):

        mx.random.seed(0xF0F0F0F0)
        world = mx.distributed.init()

        dtypes = [
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
        ]
        sizes = [
            (8,),
            (64,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(world.rank())

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (mx.random.uniform(shape=sh, key=key) * 10).astype(dt)  # shape=sh

                # Reduce scatter sum
                y = mx.distributed.reduce_scatter(x)  # shape=sh/world.size()
                z = mx.distributed.all_sum(x) / world.size()  # shape=sh

                chunk = sh[0] // world.size()
                start = world.rank() * chunk
                stop = start + chunk
                z_ref = z[start:stop]

                print(f"z type: {z_ref.dtype}, y type: {y.dtype}")
                maxrelerror = (y - z_ref).abs()
                if rtol > 0:
                    maxrelerror /= z_ref.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
