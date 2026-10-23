# Copyright © 2025 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx.nn.utils import average_gradients, clip_grad_norm_sharded


class _FakeGroup:
    """A group that only reports its size and rank so that every rank's shard
    can be checked in a single process."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class MLXDistributedCommonTestCase(mlx_tests.MLXTestCase):
    def test_average_gradients(self):
        original_all_sum = mx.distributed.all_sum
        n_calls = 0
        xtype = None

        def new_all_sum(x, **kwargs):
            nonlocal n_calls
            nonlocal xtype

            n_calls += 1
            if xtype is not None:
                self.assertEqual(xtype, x.dtype)

            return original_all_sum(x, **kwargs)

        mx.distributed.all_sum = new_all_sum

        try:
            grads = [mx.ones(10) for i in range(10)]
            new_grads = average_gradients(grads)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 1)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=4 * 50)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=0)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 10)

        finally:
            mx.distributed.all_sum = original_all_sum

    def test_all_reduce(self):
        g = mx.distributed.init()
        dtypes = [
            (mx.int8, 0),
            (mx.uint8, 0),
            (mx.int32, 0),
            (mx.uint32, 0),
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(0)

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (mx.random.uniform(shape=(g.size(),) + sh, key=key) * 10).astype(dt)

                # All sum
                y = mx.distributed.all_sum(x[g.rank()], group=g)
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                y = mx.distributed.all_max(x[g.rank()], group=g)
                z = x.max(0)
                self.assertTrue(mx.all(y == z))

                # All min
                y = mx.distributed.all_min(x[g.rank()], group=g)
                z = x.min(0)
                self.assertTrue(mx.all(y == z))

    def test_donation(self):
        x = mx.random.normal((1024,))
        mx.eval(x)
        mx.synchronize()

        mx.reset_peak_memory()
        scale = mx.array(2.0)
        y = mx.distributed.all_sum(x)
        mx.eval(y)
        mx.synchronize()
        all_sum_only = mx.get_peak_memory()
        y = mx.distributed.all_sum(x) * scale
        mx.eval(y)
        mx.synchronize()
        all_sum_with_binary = mx.get_peak_memory()

        self.assertEqual(all_sum_only, all_sum_with_binary)

    def test_shard_linear(self):
        # Seed the prng to have the same inputs and weights generated everywhere
        mx.random.seed(0xF0F0F0F0)

        # Prepare inputs
        world = mx.distributed.init()
        part = (
            slice(None),
            slice(
                world.rank() * 1024 // world.size(),
                (world.rank() + 1) * 1024 // world.size(),
            ),
        )
        x = mx.random.normal((4, 1024))

        # Create and shard some linear layers
        lin = nn.Linear(1024, 1024, bias=True)
        slin1 = shard_linear(lin, "all-to-sharded")
        slin2 = shard_linear(lin, "sharded-to-all")
        y = lin(x)
        y1 = slin1(x)
        y2 = slin2(x[part])
        self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
        self.assertTrue(mx.allclose(y[part], y1, atol=self.atol, rtol=self.rtol))

        # And their quant versions (QuantizedMatmul is not supported on CUDA)
        if not mx.cuda.is_available():
            qlin = lin.to_quantized()
            slin1 = shard_linear(qlin, "all-to-sharded")
            slin2 = shard_linear(qlin, "sharded-to-all")
            y = qlin(x)
            y1 = slin1(x)
            y2 = slin2(x[part])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(mx.allclose(y[part], y1))

            # Test non-affine quantization modes (mxfp8)
            qlin_mxfp8 = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
            self.assertEqual(qlin_mxfp8.mode, "mxfp8")

            slin1_mxfp8 = shard_linear(qlin_mxfp8, "all-to-sharded")
            slin2_mxfp8 = shard_linear(qlin_mxfp8, "sharded-to-all")

            # Verify mode is propagated
            self.assertEqual(slin1_mxfp8.mode, "mxfp8")
            self.assertEqual(slin2_mxfp8.mode, "mxfp8")

            # Verify biases parameter is not set for mxfp8
            self.assertIsNone(slin1_mxfp8.get("biases"))
            self.assertIsNone(slin2_mxfp8.get("biases"))

            y = qlin_mxfp8(x)
            y1 = slin1_mxfp8(x)
            y2 = slin2_mxfp8(x[part])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(mx.allclose(y[part], y1))

        # Check the backward works as expected
        def dummy_loss(model, x, y):
            return (model(x) * y).sum()

        mod = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
        )
        smod = nn.Sequential(
            shard_linear(mod.layers[0], "all-to-sharded"),
            shard_linear(mod.layers[1], "sharded-to-all"),
            shard_linear(mod.layers[2], "all-to-sharded"),
            shard_linear(mod.layers[3], "sharded-to-all"),
        )

        grad1 = nn.value_and_grad(mod, dummy_loss)
        grad2 = nn.value_and_grad(smod, dummy_loss)

        x = mx.random.normal((4, 128))
        y = mx.random.normal((4, 128))

        l1, g1 = grad1(mod, x, y)
        l2, g2 = grad2(smod, x, y)
        mx.eval(l1, g1, l2, g2)

        part = slice(
            world.rank() * 128 // world.size(), (world.rank() + 1) * 128 // world.size()
        )
        self.assertTrue(mx.allclose(l1, l2))
        self.assertTrue(
            mx.allclose(
                g1["layers"][0]["weight"][part],
                g2["layers"][0]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][2]["weight"][part],
                g2["layers"][2]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][1]["weight"][:, part],
                g2["layers"][1]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][3]["weight"][:, part],
                g2["layers"][3]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][0]["bias"][part],
                g2["layers"][0]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][2]["bias"][part],
                g2["layers"][2]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][1]["bias"],
                g2["layers"][1]["bias"],
                atol=self.atol,
                rtol=self.rtol,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][3]["bias"],
                g2["layers"][3]["bias"],
                atol=self.atol,
                rtol=self.rtol,
            )
        )

    def test_shard_linear_uneven(self):
        mx.random.seed(0xF0F0F0F0)
        world = mx.distributed.init()
        N = world.size()

        def part(sizes):
            start = sum(sizes[: world.rank()])
            return slice(None), slice(start, start + sizes[world.rank()])

        # Explicit sizes and the default remainder split
        explicit = [16 * (i + 1) for i in range(N)]
        default = [17] + [16] * (N - 1)
        for sizes, kwargs in ((explicit, {"sizes": explicit}), (default, {})):
            dims = sum(sizes)
            x = mx.random.normal((4, dims))
            lin = nn.Linear(dims, dims, bias=True)
            slin1 = shard_linear(lin, "all-to-sharded", **kwargs)
            slin2 = shard_linear(lin, "sharded-to-all", **kwargs)
            y = lin(x)
            self.assertTrue(
                mx.allclose(y[part(sizes)], slin1(x), atol=self.atol, rtol=self.rtol)
            )
            self.assertTrue(
                mx.allclose(y, slin2(x[part(sizes)]), atol=self.atol, rtol=self.rtol)
            )

        # QuantizedMatmul is not supported on CUDA
        if not mx.cuda.is_available():
            # Explicit sizes and the default split keep quantization groups intact
            explicit = [32 * (i + 1) for i in range(N)]
            default = [96] + [64] * (N - 1)
            for sizes, kwargs in ((explicit, {"sizes": explicit}), (default, {})):
                dims = sum(sizes)
                x = mx.random.normal((4, dims))
                qlin = nn.Linear(dims, dims).to_quantized(group_size=32, bits=4)
                slin1 = shard_linear(qlin, "all-to-sharded", **kwargs)
                slin2 = shard_linear(qlin, "sharded-to-all", **kwargs)
                y = qlin(x)
                # Uneven splits change the per-rank matmul sizes, so the
                # quantized accumulation order (and thus rounding) differs
                # from the unsharded reference more than an even split does.
                self.assertTrue(
                    mx.allclose(y[part(sizes)], slin1(x), atol=1e-5, rtol=1e-3)
                )
                self.assertTrue(
                    mx.allclose(y, slin2(x[part(sizes)]), atol=1e-5, rtol=1e-3)
                )

        # Check the backward pass
        def dummy_loss(model, x, y):
            return (model(x) * y).sum()

        sizes = [16 * (i + 1) for i in range(N)]
        dims = sum(sizes)
        mod = nn.Sequential(nn.Linear(dims, dims), nn.Linear(dims, dims))
        smod = nn.Sequential(
            shard_linear(mod.layers[0], "all-to-sharded", sizes=sizes),
            shard_linear(mod.layers[1], "sharded-to-all", sizes=sizes),
        )
        x = mx.random.normal((4, dims))
        y = mx.random.normal((4, dims))
        l1, g1 = nn.value_and_grad(mod, dummy_loss)(mod, x, y)
        l2, g2 = nn.value_and_grad(smod, dummy_loss)(smod, x, y)
        mx.eval(l1, g1, l2, g2)

        # Uneven splits change the summation order of the distributed
        # gradient reduction, so the tolerance is looser than an even split
        # needs.
        rows = part(sizes)[1]
        self.assertTrue(mx.allclose(l1, l2, atol=1e-5, rtol=1e-3))
        for key in ("weight", "bias"):
            self.assertTrue(
                mx.allclose(
                    g1["layers"][0][key][rows],
                    g2["layers"][0][key],
                    atol=1e-5,
                    rtol=1e-3,
                )
            )
        self.assertTrue(
            mx.allclose(
                g1["layers"][1]["weight"][:, rows],
                g2["layers"][1]["weight"],
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_shard_linear_uneven_shards(self):
        lin = nn.Linear(11, 12)

        # Explicit sizes
        sizes = [5, 4, 3]
        shards = [
            shard_linear(lin, "all-to-sharded", sizes=sizes, group=_FakeGroup(3, r))
            for r in range(3)
        ]
        self.assertEqual([s.weight.shape for s in shards], [(5, 11), (4, 11), (3, 11)])
        self.assertTrue(
            mx.array_equal(mx.concatenate([s.weight for s in shards]), lin.weight)
        )
        self.assertTrue(
            mx.array_equal(mx.concatenate([s.bias for s in shards]), lin.bias)
        )
        self.assertIn("output_dims=12", repr(shards[0]))

        # The default split gives the remainder to the first ranks
        shards = [
            shard_linear(lin, "sharded-to-all", group=_FakeGroup(3, r))
            for r in range(3)
        ]
        self.assertEqual([s.weight.shape for s in shards], [(12, 4), (12, 4), (12, 3)])
        self.assertTrue(
            mx.array_equal(
                mx.concatenate([s.weight for s in shards], axis=1), lin.weight
            )
        )
        self.assertIn("input_dims=11", repr(shards[0]))

    def test_shard_linear_uneven_quantized_shards(self):
        # Split the packed weights, scales and biases at the same boundaries
        sizes = [96, 64, 32]
        for bits in (2, 3, 4, 5, 6, 8):
            qlin = nn.Linear(192, 16).to_quantized(group_size=32, bits=bits)
            shards = [
                shard_linear(
                    qlin, "sharded-to-all", sizes=sizes, group=_FakeGroup(3, r)
                )
                for r in range(3)
            ]
            for key in ("weight", "scales", "biases"):
                parts = [s[key] for s in shards]
                self.assertEqual(
                    [p.shape[1] for p in parts],
                    [qlin[key].shape[1] * s // 192 for s in sizes],
                )
                self.assertTrue(
                    mx.array_equal(mx.concatenate(parts, axis=1), qlin[key])
                )
            self.assertIn("input_dims=192", repr(shards[0]))

        # Output rows can be split anywhere
        qlin = nn.Linear(64, 10).to_quantized(group_size=32, bits=4)
        shards = [
            shard_linear(qlin, "all-to-sharded", sizes=[7, 3], group=_FakeGroup(2, r))
            for r in range(2)
        ]
        for key in ("weight", "scales", "biases", "bias"):
            self.assertTrue(
                mx.array_equal(mx.concatenate([s[key] for s in shards]), qlin[key])
            )
        self.assertIn("output_dims=10", repr(shards[0]))

        # Paired default splits match when the output rows allow it
        qin = nn.Linear(64, 160).to_quantized(group_size=32, bits=4)
        qout = nn.Linear(160, 64).to_quantized(group_size=32, bits=4)
        for r, rows in enumerate((64, 64, 32)):
            s1 = shard_linear(qin, "all-to-sharded", group=_FakeGroup(3, r))
            s2 = shard_linear(qout, "sharded-to-all", group=_FakeGroup(3, r))
            self.assertEqual(s1.weight.shape[0], rows)
            self.assertEqual(s2.scales.shape[1] * s2.group_size, rows)

    def test_shard_linear_uneven_errors(self):
        lin = nn.Linear(10, 12)
        group = _FakeGroup(3, 0)

        # The sizes must have one entry per rank and sum to the sharded dimension
        with self.assertRaises(ValueError):
            shard_linear(lin, "all-to-sharded", sizes=[5, 4, 2], group=group)
        with self.assertRaises(ValueError):
            shard_linear(lin, "all-to-sharded", sizes=[6, 6], group=group)

        # Every rank needs a non-empty shard
        with self.assertRaises(ValueError):
            shard_linear(lin, "all-to-sharded", sizes=[12, 0, 0], group=group)
        with self.assertRaises(ValueError):
            shard_linear(lin, "sharded-to-all", sizes=[11, -1, 0], group=group)
        with self.assertRaises(ValueError):
            shard_linear(nn.Linear(10, 2), "all-to-sharded", group=group)

        # Uneven splits require a single segment
        with self.assertRaises(ValueError):
            shard_linear(
                lin, "all-to-sharded", segments=2, sizes=[4, 4, 4], group=group
            )
        with self.assertRaises(ValueError):
            shard_linear(nn.Linear(10, 14), "all-to-sharded", segments=2, group=group)

        # Quantized input shards must be multiples of the group size
        qlin = nn.Linear(96, 10).to_quantized(group_size=32, bits=4)
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            shard_linear(qlin, "sharded-to-all", sizes=[48, 48], group=_FakeGroup(2, 0))
        with self.assertRaises(ValueError):
            shard_linear(qlin, "sharded-to-all", group=_FakeGroup(4, 0))
        with self.assertRaises(ValueError):
            shard_linear(qlin, "all-to-sharded", group=_FakeGroup(11, 0))

    def test_shard_inplace_uneven(self):
        # A dimension that does not divide by the number of devices is split
        # as evenly as possible instead of raising.
        lin = nn.Linear(12, 11)
        shards = []
        for r in range(3):
            m = nn.Linear(12, 11)
            m.update(lin.parameters())
            shard_inplace(m, "all-to-sharded", group=_FakeGroup(3, r))
            shards.append(m.weight)
        self.assertEqual([s.shape[0] for s in shards], [4, 4, 3])
        self.assertTrue(mx.array_equal(mx.concatenate(shards), lin.weight))

        # QuantizedMatmul is not supported on CUDA
        if not mx.cuda.is_available():
            # A quantized module's packed weight and its grouped scales and
            # biases must be split at the same boundaries, so the sizes are
            # multiples of the group size.
            qlin = nn.Linear(1408, 64).to_quantized(group_size=64, bits=4)
            parts = {"weight": [], "scales": [], "biases": []}
            for r in range(3):
                m = nn.Linear(1408, 64).to_quantized(group_size=64, bits=4)
                m.update(qlin.parameters())
                shard_inplace(m, "sharded-to-all", group=_FakeGroup(3, r))
                for key in parts:
                    parts[key].append(m[key])
            # 1408 is 22 groups of 64, split as 8, 7 and 7 groups.
            self.assertEqual([p.shape[-1] for p in parts["weight"]], [64, 56, 56])
            self.assertEqual([p.shape[-1] for p in parts["scales"]], [8, 7, 7])
            for key, whole in parts.items():
                self.assertTrue(
                    mx.array_equal(mx.concatenate(whole, axis=-1), qlin[key])
                )

            # A paired all-to-sharded layer splits the same dimension the same
            # way, so the two layers agree on each rank's share.
            qout = nn.Linear(64, 1408).to_quantized(group_size=64, bits=4)
            rows = []
            for r in range(3):
                m = nn.Linear(64, 1408).to_quantized(group_size=64, bits=4)
                m.update(qout.parameters())
                shard_inplace(m, "all-to-sharded", group=_FakeGroup(3, r))
                rows.append(m.weight.shape[0])
            self.assertEqual(rows, [512, 448, 448])
            self.assertEqual(rows, [p.shape[-1] * 64 for p in parts["scales"]])

    def test_shard_predicate(self):
        mx.random.seed(0xF0F0F0F0)

        class MyConv(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.aggregate = kwargs.pop("aggregate", False)
                self.conv = nn.Conv2d(*args, **kwargs)

            def __call__(self, x):
                x = self.conv(x)
                if self.aggregate:
                    x = mx.distributed.all_sum(x)
                return x

        def sharding(path, weight):
            parts = path.split(".")
            even = int(parts[1]) % 2 == 0
            if even:
                return 0
            else:
                return -1 if parts[-1] != "bias" else None

        mod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3),
        )
        smod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3, aggregate=True),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3, aggregate=True),
        )
        smod.update(mod.parameters())
        shard_inplace(smod, sharding)

        x = mx.random.normal((4, 16, 16, 3))
        y1 = mod(x)
        y2 = smod(x)
        self.assertTrue(mx.allclose(y1, y2, atol=1e-6, rtol=1e-4))

    def test_all_gather(self):
        world = mx.distributed.init()
        dtypes = [
            mx.int8,
            mx.uint8,
            mx.int32,
            mx.uint32,
            mx.float32,
            mx.float16,
            mx.bfloat16,
        ]
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

    def test_all_gather_rank_data(self):
        # Every rank contributes distinct data so that a missing, stale, or
        # misplaced peer region is detected (an all-ones gather cannot tell).
        # Sizes include one large enough that backends which slice transfers
        # across directions/wires (e.g. the jaccl ring) exercise every slice,
        # and odd sizes that exercise the tail clamping.
        world = mx.distributed.init()
        for size in [7, 1024, 1000003, 4 * 1024 * 1024]:
            base = mx.arange(size, dtype=mx.int32) % 100003
            x = base + world.rank()
            y = mx.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * size,))
            regions = y.reshape(world.size(), size)
            for r in range(world.size()):
                self.assertTrue(mx.all(regions[r] == base + r).item())

    def test_clip_grad_norm_sharded(self):
        world = mx.distributed.init()
        N = world.size()

        value = 3.0
        grads_slice = {"a": mx.ones((4, 3)) * value, "b": mx.ones((5,)) * value}
        local_numel = 4 * 3 + 5
        expected_norm = math.sqrt(N * local_numel) * value

        clipped, grad_norm = clip_grad_norm_sharded(
            grads_slice, max_norm=1e9, group=world
        )
        mx.eval(clipped, grad_norm)
        self.assertTrue(
            mx.allclose(
                grad_norm, mx.array(expected_norm), atol=self.atol, rtol=self.rtol
            )
        )
        for k in grads_slice:
            self.assertTrue(
                mx.allclose(clipped[k], grads_slice[k], atol=self.atol, rtol=self.rtol)
            )

        max_norm = 1.0
        clipped, grad_norm = clip_grad_norm_sharded(
            grads_slice, max_norm=max_norm, group=world
        )
        mx.eval(clipped, grad_norm)
        scale = max_norm / (expected_norm + 1e-6)
        for k in grads_slice:
            self.assertTrue(
                mx.allclose(
                    clipped[k], grads_slice[k] * scale, atol=self.atol, rtol=self.rtol
                )
            )

    def test_jaccl_all_gather_factory_validation(self):
        # A custom side-channel factory is only valid with the jaccl backend.
        with self.assertRaises(ValueError):
            mx.distributed.init(
                backend="ring",
                all_gather_factory=lambda rank, size: lambda src, n_bytes: b"",
            )

        # The factory must be callable.
        with self.assertRaises(TypeError):
            mx.distributed.init(backend="jaccl", all_gather_factory="not_callable")
