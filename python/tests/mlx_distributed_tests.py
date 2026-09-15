# Copyright © 2025 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
from mlx.nn.layers.distributed import (
    _rank_sizes,
    _split_uneven,
    shard_inplace,
    shard_linear,
)
from mlx.nn.utils import average_gradients, clip_grad_norm_sharded


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
        # Match shard_linear's remainder distribution.
        _sizes1024 = _rank_sizes(1024, world.size())
        _start1024 = sum(_sizes1024[: world.rank()])
        part = (
            slice(None),
            slice(_start1024, _start1024 + _sizes1024[world.rank()]),
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
            # Quantized shards use group-aligned boundaries.
            def _quant_part(group_size):
                sizes = _rank_sizes(1024, world.size(), block=group_size)
                start = sum(sizes[: world.rank()])
                return (slice(None), slice(start, start + sizes[world.rank()]))

            qpart = _quant_part(64)
            qlin = lin.to_quantized()
            slin1 = shard_linear(qlin, "all-to-sharded")
            slin2 = shard_linear(qlin, "sharded-to-all")
            y = qlin(x)
            y1 = slin1(x)
            y2 = slin2(x[qpart])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(mx.allclose(y[qpart], y1))

            # Test non-affine quantization modes (mxfp8)
            qpart_mxfp8 = _quant_part(32)
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
            y2 = slin2_mxfp8(x[qpart_mxfp8])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(mx.allclose(y[qpart_mxfp8], y1))

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

        # Match shard_linear's remainder distribution.
        _sizes128 = _rank_sizes(128, world.size())
        _start128 = sum(_sizes128[: world.rank()])
        part = slice(_start128, _start128 + _sizes128[world.rank()])
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

    def test_rank_sizes(self):
        # Evenly-divisible cases reduce to the plain even split.
        self.assertEqual(_rank_sizes(12, 3), [4, 4, 4])
        self.assertEqual(_rank_sizes(256, 4, block=64), [64, 64, 64, 64])
        # Never split a block between ranks.
        self.assertEqual(_rank_sizes(128, 4, block=64), [64, 64, 0, 0])

        # Give remainder units to the first ranks.
        self.assertEqual(_rank_sizes(10, 3), [4, 3, 3])
        self.assertEqual(_rank_sizes(11, 3), [4, 4, 3])
        self.assertEqual(_rank_sizes(1, 3), [1, 0, 0])

        # Block-aware splits preserve whole groups.
        self.assertEqual(_rank_sizes(320, 3, block=64), [128, 128, 64])
        for N in (2, 3, 4, 5, 7):
            sizes = _rank_sizes(320, N, block=64)
            self.assertEqual(sum(sizes), 320)
            self.assertTrue(all(s % 64 == 0 for s in sizes))
            self.assertLessEqual(max(sizes) - min(sizes), 64)

        # A dim that isn't a multiple of block can't be split at all.
        with self.assertRaises(ValueError):
            _rank_sizes(10, 3, block=64)

    def test_split_uneven(self):
        mx.random.seed(0)
        w = mx.random.normal((10, 8))

        # sizes=None falls back to the remainder-aware split of this
        # array's own axis length.
        parts = _split_uneven(w, 3, axis=0)
        self.assertEqual([p.shape[0] for p in parts], [4, 3, 3])
        self.assertTrue(mx.array_equal(mx.concatenate(parts, axis=0), w))

        # Explicit sizes that already match this array's axis length are
        # used directly.
        parts = _split_uneven(w, 3, axis=0, sizes=[5, 3, 2])
        self.assertEqual([p.shape[0] for p in parts], [5, 3, 2])
        self.assertTrue(mx.array_equal(mx.concatenate(parts, axis=0), w))

        # Scale logical sizes to a packed axis.
        packed = mx.random.normal((10, 4))  # axis 1 packed 2x relative to 8
        parts = _split_uneven(packed, 3, axis=1, sizes=[4, 2, 2])
        self.assertEqual([p.shape[1] for p in parts], [2, 1, 1])
        self.assertTrue(mx.array_equal(mx.concatenate(parts, axis=1), packed))

        # Incompatible sizes (neither divides the other) raise.
        with self.assertRaises(ValueError):
            _split_uneven(w, 3, axis=0, sizes=[3, 3, 3])  # sums to 9, not 10

        # Require one size per rank.
        with self.assertRaises(ValueError):
            _split_uneven(w, 3, axis=0, sizes=[5, 3, 2, 0])  # 4 sizes, N=3
        with self.assertRaises(ValueError):
            _split_uneven(w, 2, axis=0, sizes=[5, 3, 2])  # 3 sizes, N=2

    def test_shard_linear_uneven(self):
        # Inspect every shard without distributed communication.
        class _FakeGroup:
            def __init__(self, n, r):
                self._n, self._r = n, r

            def size(self):
                return self._n

            def rank(self):
                return self._r

        mx.random.seed(0xF0F0F0F0)

        lin = nn.Linear(10, 12, bias=True)
        x = mx.random.normal((4, 10))
        y = lin(x)

        # Check an explicit uneven output split.
        sizes = [5, 4, 3]
        N = len(sizes)
        shards = [
            shard_linear(lin, "all-to-sharded", sizes=sizes, group=_FakeGroup(N, r))
            for r in range(N)
        ]
        starts = [0, 5, 9]
        for r, (s, start) in enumerate(zip(shards, starts)):
            self.assertEqual(s.weight.shape, (sizes[r], 10))
            self.assertEqual(s.bias.shape, (sizes[r],))
            self.assertTrue(
                mx.allclose(s(x), y[:, start : start + sizes[r]], atol=1e-6, rtol=1e-4)
            )
            self.assertEqual(s._extra_repr().count("output_dims=12"), 1)
        # The shards reconstruct the original weight.
        self.assertTrue(
            mx.array_equal(
                mx.concatenate([s.weight for s in shards], axis=0), lin.weight
            )
        )

        # Check the same split along the input dimension.
        lin2 = nn.Linear(12, 10, bias=True)
        shards2 = [
            shard_linear(lin2, "sharded-to-all", sizes=sizes, group=_FakeGroup(N, r))
            for r in range(N)
        ]
        for r, s in enumerate(shards2):
            self.assertEqual(s.weight.shape, (10, sizes[r]))
            self.assertEqual(s._extra_repr().count("input_dims=12"), 1)
        self.assertTrue(
            mx.array_equal(
                mx.concatenate([s.weight for s in shards2], axis=1), lin2.weight
            )
        )

        # Automatic sizes also distribute a remainder instead of requiring
        # the dimension to divide evenly.
        auto = nn.Linear(10, 11, bias=True)
        auto_shards = [
            shard_linear(auto, "all-to-sharded", group=_FakeGroup(3, r))
            for r in range(3)
        ]
        self.assertEqual([s.weight.shape[0] for s in auto_shards], [4, 4, 3])
        self.assertTrue(
            mx.array_equal(
                mx.concatenate([s.weight for s in auto_shards], axis=0), auto.weight
            )
        )

        auto_in = nn.Linear(11, 10, bias=True)
        auto_in_shards = [
            shard_linear(auto_in, "sharded-to-all", group=_FakeGroup(3, r))
            for r in range(3)
        ]
        self.assertEqual([s.weight.shape[1] for s in auto_in_shards], [4, 4, 3])
        self.assertTrue(
            mx.array_equal(
                mx.concatenate([s.weight for s in auto_in_shards], axis=1),
                auto_in.weight,
            )
        )

        # Plain segmented coverage must run on CUDA too.
        segmented = nn.Linear(10, 14, bias=True)
        segmented_shards = [
            shard_linear(
                segmented, "all-to-sharded", segments=[9], group=_FakeGroup(2, r)
            )
            for r in range(2)
        ]
        self.assertEqual([s.weight.shape[0] for s in segmented_shards], [8, 6])

        # Explicit sizes must sum to the sharded dimension.
        with self.assertRaises(ValueError):
            shard_linear(
                lin, "all-to-sharded", sizes=[5, 4, 2], group=_FakeGroup(3, 0)
            )  # sums to 11, not 12

        # Explicit sizes must contain one entry per rank.
        with self.assertRaises(ValueError):
            shard_linear(lin, "all-to-sharded", sizes=sizes, group=_FakeGroup(2, 0))

        # QuantizedMatmul is not supported on CUDA.
        if not mx.cuda.is_available():
            # Output splits need not align to quantization groups.
            xq = mx.random.normal((4, 64))
            qlin = nn.Linear(64, 96, bias=True).to_quantized(group_size=32, bits=4)
            yq = qlin(xq)

            # Default output shards use the same block-aware split as a paired
            # quantized sharded-to-all layer.
            qout = nn.Linear(64, 320, bias=True).to_quantized(group_size=64, bits=4)
            qin = nn.Linear(320, 64, bias=True).to_quantized(group_size=64, bits=4)
            qout_shards = [
                shard_linear(qout, "all-to-sharded", group=_FakeGroup(3, r))
                for r in range(3)
            ]
            qin_shards = [
                shard_linear(qin, "sharded-to-all", group=_FakeGroup(3, r))
                for r in range(3)
            ]
            self.assertEqual([s.weight.shape[0] for s in qout_shards], [128, 128, 64])
            self.assertEqual([s._total_input_dims for s in qin_shards], [320, 320, 320])
            self.assertEqual(
                [s.weight.shape[1] * 32 // s.bits for s in qin_shards],
                [128, 128, 64],
            )
            q_sizes = [64, 32]  # multiples of group_size=32, sums to 96
            for r, start in zip(range(2), [0, 64]):
                s = shard_linear(
                    qlin, "all-to-sharded", sizes=q_sizes, group=_FakeGroup(2, r)
                )
                self.assertEqual(s.weight.shape[0], q_sizes[r])
                self.assertTrue(
                    mx.allclose(
                        s(xq), yq[:, start : start + q_sizes[r]], atol=1e-3, rtol=1e-2
                    )
                )
            # Not a multiple of group_size=32, but still valid: splitting output
            # rows doesn't touch a quantization group.
            for r, start in zip(range(2), [0, 48]):
                s = shard_linear(
                    qlin, "all-to-sharded", sizes=[48, 48], group=_FakeGroup(2, r)
                )
                self.assertEqual(s.weight.shape[0], 48)
                self.assertTrue(
                    mx.allclose(s(xq), yq[:, start : start + 48], atol=1e-3, rtol=1e-2)
                )

            # Input splits preserve groups for every supported bit width.
            xq2 = mx.random.normal((4, 96))
            sq_sizes = [64, 32]
            for bits in (2, 3, 4, 5, 6, 8):
                qlinb = nn.Linear(96, 64, bias=True).to_quantized(
                    group_size=32, bits=bits
                )
                shardsb = [
                    shard_linear(
                        qlinb, "sharded-to-all", sizes=sq_sizes, group=_FakeGroup(2, r)
                    )
                    for r in range(2)
                ]
                for r in range(2):
                    self.assertEqual(
                        shardsb[r].weight.shape, (64, sq_sizes[r] * bits // 32)
                    )
                    self.assertEqual(shardsb[r].scales.shape, (64, sq_sizes[r] // 32))
                self.assertEqual(shardsb[0]._extra_repr().count("input_dims=96"), 1)
                # Check each local matmul without the distributed all-sum.
                starts = [0, 64]
                for r, start in zip(range(2), starts):
                    partial = mx.quantized_matmul(
                        xq2[:, start : start + sq_sizes[r]],
                        shardsb[r].weight,
                        scales=shardsb[r].scales,
                        biases=shardsb[r].get("biases"),
                        transpose=True,
                        group_size=shardsb[r].group_size,
                        bits=shardsb[r].bits,
                        mode=shardsb[r].mode,
                    )
                    ref_x = mx.zeros((4, 96))
                    ref_x[:, start : start + sq_sizes[r]] = xq2[
                        :, start : start + sq_sizes[r]
                    ]
                    self.assertTrue(
                        mx.allclose(
                            partial,
                            qlinb(ref_x) - qlinb.bias,
                            atol=1e-2,
                            rtol=5e-2,
                        )
                    )
                if bits == 4:
                    qlin2 = qlinb  # reused by the rejection test below

            # Check grouped boundaries across three fused segments.
            qlin3 = nn.Linear(192, 8, bias=True).to_quantized(group_size=32, bits=4)
            seg_qshards = [
                shard_linear(
                    qlin3, "sharded-to-all", segments=3, group=_FakeGroup(2, r)
                )
                for r in range(2)
            ]
            seg_cols = [
                list(range(0, 32)) + list(range(64, 96)) + list(range(128, 160)),
                list(range(32, 64)) + list(range(96, 128)) + list(range(160, 192)),
            ]
            xq3 = mx.random.normal((4, 192))
            total = mx.zeros((4, 8))
            for r in range(2):
                s = seg_qshards[r]
                self.assertEqual(len(seg_cols[r]), 96)  # 32 per segment * 3 segments
                self.assertEqual(s.weight.shape, (8, 96 * 4 // 32))
                partial = mx.quantized_matmul(
                    xq3[:, seg_cols[r]],
                    s.weight,
                    scales=s.scales,
                    biases=s.get("biases"),
                    transpose=True,
                    group_size=s.group_size,
                    bits=s.bits,
                    mode=s.mode,
                )
                total = total + partial
            self.assertTrue(
                mx.allclose(total + qlin3.bias, qlin3(xq3), atol=1e-2, rtol=5e-2)
            )

            # List boundaries remain valid when every segment can be divided
            # at quantization-group boundaries.
            qlin4 = nn.Linear(128, 8, bias=True).to_quantized(group_size=32, bits=4)
            xq4 = mx.random.normal((4, 128))
            segment_cols = [
                list(range(0, 32)) + list(range(64, 96)),
                list(range(32, 64)) + list(range(96, 128)),
            ]
            for segments in ([0.5], [64]):
                list_qshards = [
                    shard_linear(
                        qlin4,
                        "sharded-to-all",
                        segments=segments,
                        group=_FakeGroup(2, r),
                    )
                    for r in range(2)
                ]
                self.assertEqual(
                    [s.weight.shape[1] * 32 // s.bits for s in list_qshards],
                    [64, 64],
                )
                self.assertEqual([s.scales.shape[1] for s in list_qshards], [2, 2])
                total = mx.zeros((4, 8))
                for shard, cols in zip(list_qshards, segment_cols):
                    total += mx.quantized_matmul(
                        xq4[:, cols],
                        shard.weight,
                        scales=shard.scales,
                        biases=shard.get("biases"),
                        transpose=True,
                        group_size=shard.group_size,
                        bits=shard.bits,
                        mode=shard.mode,
                    )
                self.assertTrue(
                    mx.allclose(total + qlin4.bias, qlin4(xq4), atol=1e-2, rtol=5e-2)
                )
            with self.assertRaises(ValueError):
                # 48 is not a multiple of group_size=32
                shard_linear(
                    qlin2, "sharded-to-all", sizes=[48, 48], group=_FakeGroup(2, 0)
                )

            # Divide each explicit size evenly among segments.
            fused = nn.Linear(10, 12, bias=True)
            with self.assertRaises(ValueError):
                shard_linear(
                    fused,
                    "all-to-sharded",
                    segments=3,
                    sizes=[5, 4, 3],
                    group=_FakeGroup(3, 0),
                )

            # Split each fused segment independently.
            seg_shards = [
                shard_linear(
                    fused, "all-to-sharded", segments=3, group=_FakeGroup(2, r)
                )
                for r in range(2)
            ]
            expected_rank0 = mx.concatenate(
                [fused.weight[0:2], fused.weight[4:6], fused.weight[8:10]], axis=0
            )
            expected_rank1 = mx.concatenate(
                [fused.weight[2:4], fused.weight[6:8], fused.weight[10:12]], axis=0
            )
            self.assertEqual(seg_shards[0].weight.shape, (6, 10))
            self.assertEqual(seg_shards[1].weight.shape, (6, 10))
            self.assertTrue(mx.array_equal(seg_shards[0].weight, expected_rank0))
            self.assertTrue(mx.array_equal(seg_shards[1].weight, expected_rank1))

            # Apply explicit sizes to each fused segment.
            seg_shards2 = [
                shard_linear(
                    fused,
                    "all-to-sharded",
                    segments=3,
                    sizes=[9, 3],
                    group=_FakeGroup(2, r),
                )
                for r in range(2)
            ]
            expected_rank0_2 = mx.concatenate(
                [fused.weight[0:3], fused.weight[4:7], fused.weight[8:11]], axis=0
            )
            expected_rank1_2 = mx.concatenate(
                [fused.weight[3:4], fused.weight[7:8], fused.weight[11:12]], axis=0
            )
            self.assertEqual(seg_shards2[0].weight.shape, (9, 10))
            self.assertEqual(seg_shards2[1].weight.shape, (3, 10))
            self.assertTrue(mx.array_equal(seg_shards2[0].weight, expected_rank0_2))
            self.assertTrue(mx.array_equal(seg_shards2[1].weight, expected_rank1_2))

            # Unequal segments do not support explicit sizes.
            with self.assertRaises(ValueError):
                shard_linear(
                    fused,
                    "all-to-sharded",
                    segments=[6],
                    sizes=[5, 4, 3],
                    group=_FakeGroup(3, 0),
                )

            # Split unequal plain segments independently.
            asym = nn.Linear(10, 14, bias=True)
            seg_list_shards = [
                shard_linear(
                    asym, "all-to-sharded", segments=[9], group=_FakeGroup(2, r)
                )
                for r in range(2)
            ]
            expected_list_rank0 = mx.concatenate(
                [asym.weight[0:5], asym.weight[9:12]], axis=0
            )
            expected_list_rank1 = mx.concatenate(
                [asym.weight[5:9], asym.weight[12:14]], axis=0
            )
            self.assertEqual(seg_list_shards[0].weight.shape, (8, 10))
            self.assertEqual(seg_list_shards[1].weight.shape, (6, 10))
            self.assertTrue(
                mx.array_equal(seg_list_shards[0].weight, expected_list_rank0)
            )
            self.assertTrue(
                mx.array_equal(seg_list_shards[1].weight, expected_list_rank1)
            )

            # Quantized output segments also split independently.
            seg_list_qshards = [
                shard_linear(
                    qlin, "all-to-sharded", segments=[48], group=_FakeGroup(2, r)
                )
                for r in range(2)
            ]
            self.assertEqual(seg_list_qshards[0].weight.shape[0], 48)
            self.assertEqual(seg_list_qshards[1].weight.shape[0], 48)
            self.assertTrue(
                mx.allclose(
                    seg_list_qshards[0](xq),
                    mx.concatenate([yq[:, 0:24], yq[:, 48:72]], axis=1),
                    atol=1e-3,
                    rtol=1e-2,
                )
            )
            self.assertTrue(
                mx.allclose(
                    seg_list_qshards[1](xq),
                    mx.concatenate([yq[:, 24:48], yq[:, 72:96]], axis=1),
                    atol=1e-3,
                    rtol=1e-2,
                )
            )

            # A list-valued split is rejected when a segment cannot be divided
            # at quantization-group boundaries.
            with self.assertRaises(ValueError):
                shard_linear(
                    qlin2, "sharded-to-all", segments=[48], group=_FakeGroup(2, 0)
                )

            # Reject zero-width quantized shards on every rank.
            tiny_qlin = nn.Linear(192, 64, bias=True).to_quantized(
                group_size=64, bits=4
            )
            for r in (0, 1):
                with self.assertRaises(ValueError):
                    shard_linear(
                        tiny_qlin, "sharded-to-all", segments=3, group=_FakeGroup(2, r)
                    )

    def test_shard_inplace_custom_parameters(self):
        class _FakeGroup:
            def __init__(self, n, r):
                self._n, self._r = n, r

            def size(self):
                return self._n

            def rank(self):
                return self._r

        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mx.zeros((10,))
                self.scales = mx.zeros((6,))

        # Names used by quantized layers do not couple unrelated arrays in a
        # generic module.
        expected = [(4, 2), (3, 2), (3, 2)]
        for r, shapes in enumerate(expected):
            module = CustomModule()
            shard_inplace(module, lambda path, value: 0, group=_FakeGroup(3, r))
            self.assertEqual((module.weight.size, module.scales.size), shapes)

        module = CustomModule()
        with self.assertRaises(ValueError):
            shard_inplace(module, lambda path, value: 1, group=_FakeGroup(2, 0))

    def test_shard_inplace_third_party_quantized_module(self):
        # Detect third-party quantized modules by their parameters.
        if not mx.cuda.is_available():

            class _FakeGroup:
                def __init__(self, n, r):
                    self._n, self._r = n, r

                def size(self):
                    return self._n

                def rank(self):
                    return self._r

            class ThirdPartyQuantized(nn.Module):
                def __init__(self, input_dims, output_dims, group_size, bits):
                    super().__init__()
                    self.group_size = group_size
                    self.bits = bits
                    lin = nn.Linear(input_dims, output_dims, bias=True).to_quantized(
                        group_size=group_size, bits=bits
                    )
                    self.weight = lin.weight
                    self.scales = lin.scales
                    if lin.get("biases") is not None:
                        self.biases = lin.biases
                    self.bias = lin.bias

            # Check an uneven grouped input split.
            ref = ThirdPartyQuantized(96, 64, group_size=32, bits=4)
            x = mx.random.normal((4, 96))
            y_full = (
                mx.quantized_matmul(
                    x,
                    ref.weight,
                    scales=ref.scales,
                    biases=ref.get("biases"),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                + ref.bias
            )
            total = None
            cols = [range(0, 64), range(64, 96)]
            for r in (0, 1):
                m = ThirdPartyQuantized(96, 64, group_size=32, bits=4)
                m.update(ref.parameters())
                shard_inplace(m, "sharded-to-all", group=_FakeGroup(2, r))
                self.assertEqual(m.weight.shape[1] * 32 // 4, [64, 32][r])
                partial = mx.quantized_matmul(
                    x[:, list(cols[r])],
                    m.weight,
                    scales=m.scales,
                    biases=m.get("biases"),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                total = partial if total is None else total + partial
            self.assertTrue(mx.allclose(total + ref.bias, y_full, atol=1e-2, rtol=5e-2))

    def test_shard_inplace_predicate_called_once(self):
        if mx.cuda.is_available():
            return

        class _FakeGroup:
            def size(self):
                return 2

            def rank(self):
                return 0

        module = nn.Linear(128, 64).to_quantized(group_size=32, bits=4)
        calls = {}

        def sharding(path, value):
            calls[path] = calls.get(path, 0) + 1
            return None if path.endswith("bias") else -1

        shard_inplace(module, sharding, group=_FakeGroup())
        self.assertTrue(calls)
        self.assertTrue(all(count == 1 for count in calls.values()))

    def test_shard_inplace_quantized_consistency(self):
        # QuantizedMatmul is not supported on CUDA.
        if not mx.cuda.is_available():
            # Keep packed weights and grouped metadata aligned.
            class _FakeGroup:
                def __init__(self, n, r):
                    self._n, self._r = n, r

                def size(self):
                    return self._n

                def rank(self):
                    return self._r

            # Check an uneven split of 96 inputs into 64 and 32.
            qlin = nn.Linear(96, 64, bias=True).to_quantized(group_size=32, bits=4)
            x = mx.random.normal((4, 96))
            y_full = qlin(x)
            total = None
            cols = [range(0, 64), range(64, 96)]
            for r in (0, 1):
                m = nn.Linear(96, 64, bias=True).to_quantized(group_size=32, bits=4)
                m.update(qlin.parameters())
                shard_inplace(m, "sharded-to-all", group=_FakeGroup(2, r))
                partial = mx.quantized_matmul(
                    x[:, list(cols[r])],
                    m.weight,
                    scales=m.scales,
                    biases=m.get("biases"),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                total = partial if total is None else total + partial
            self.assertTrue(
                mx.allclose(total + qlin.bias, y_full, atol=1e-2, rtol=5e-2)
            )

            # Handle quantized parameters in a nested module.
            class Wrap(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(96, 64, bias=True).to_quantized(
                        group_size=32, bits=4
                    )

            w = Wrap()
            for r in (0, 1):
                wm = Wrap()
                wm.update(w.parameters())
                shard_inplace(wm, "sharded-to-all", group=_FakeGroup(2, r))
                self.assertEqual(wm.linear.weight.shape[1], [8, 4][r])

            # Preserve the evenly divisible case.
            qlin2 = nn.Linear(128, 64, bias=True).to_quantized(group_size=32, bits=4)
            x2 = mx.random.normal((4, 128))
            y_full2 = qlin2(x2)
            total2 = None
            for r in (0, 1):
                m2 = nn.Linear(128, 64, bias=True).to_quantized(group_size=32, bits=4)
                m2.update(qlin2.parameters())
                shard_inplace(m2, "sharded-to-all", group=_FakeGroup(2, r))
                xr = x2[:, r * 64 : (r + 1) * 64]
                partial = mx.quantized_matmul(
                    xr,
                    m2.weight,
                    scales=m2.scales,
                    biases=m2.get("biases"),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                total2 = partial if total2 is None else total2 + partial
            self.assertTrue(
                mx.allclose(total2 + qlin2.bias, y_full2, atol=1e-2, rtol=5e-2)
            )

            # Metadata cannot be sharded without its weight.
            qlin3 = nn.Linear(96, 64).to_quantized(group_size=32, bits=4)

            def skip_weight(path, value):
                return None if path.endswith("weight") else 0

            m3 = nn.Linear(96, 64).to_quantized(group_size=32, bits=4)
            m3.update(qlin3.parameters())
            with self.assertRaises(ValueError):
                shard_inplace(m3, skip_weight, group=_FakeGroup(2, 0))

    def test_shard_inplace_quantized_list_segments(self):
        # QuantizedMatmul is not supported on CUDA.
        if not mx.cuda.is_available():

            class _FakeGroup:
                def __init__(self, n, r):
                    self._n, self._r = n, r

                def size(self):
                    return self._n

                def rank(self):
                    return self._r

            # Reject segment boundaries that differ after packing and grouping.
            qlin = nn.Linear(1024, 8, bias=True).to_quantized(group_size=64, bits=4)
            for r in (0, 1):
                m = nn.Linear(1024, 8, bias=True).to_quantized(group_size=64, bits=4)
                m.update(qlin.parameters())
                with self.assertRaises(ValueError):
                    shard_inplace(
                        m, "sharded-to-all", segments=[0.3], group=_FakeGroup(2, r)
                    )

            # Accept boundaries that align in every representation.
            qlin2 = nn.Linear(256, 8, bias=True).to_quantized(group_size=32, bits=4)
            x = mx.random.normal((4, 256))
            y_full = qlin2(x)
            # Each rank gets 64 columns from both segments.
            cols = [
                list(range(0, 64)) + list(range(128, 192)),
                list(range(64, 128)) + list(range(192, 256)),
            ]
            total = None
            for r in (0, 1):
                m2 = nn.Linear(256, 8, bias=True).to_quantized(group_size=32, bits=4)
                m2.update(qlin2.parameters())
                shard_inplace(
                    m2, "sharded-to-all", segments=[0.5], group=_FakeGroup(2, r)
                )
                partial = mx.quantized_matmul(
                    x[:, cols[r]],
                    m2.weight,
                    scales=m2.scales,
                    biases=m2.get("biases"),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                total = partial if total is None else total + partial
            self.assertTrue(
                mx.allclose(total + qlin2.bias, y_full, atol=1e-2, rtol=5e-2)
            )

    def test_shard_linear_quantized_fused_segments_output(self):
        # Fall back to plain output splits when groups cannot cover all ranks.
        output_dims, n_segments, N, group_size = 256, 2, 4, 64
        qlin = nn.Linear(64, output_dims, bias=True).to_quantized(
            group_size=group_size, bits=4
        )

        class _FakeGroup:
            def __init__(self, n, r):
                self._n, self._r = n, r

            def size(self):
                return self._n

            def rank(self):
                return self._r

        shards = [
            shard_linear(
                qlin, "all-to-sharded", segments=n_segments, group=_FakeGroup(N, r)
            )
            for r in range(N)
        ]
        self.assertEqual([s.weight.shape[0] for s in shards], [64, 64, 64, 64])

    def test_shard_inplace_quantized_zero_share_rejected(self):
        # shard_inplace must also reject zero-width quantized shards.
        class _FakeGroup:
            def __init__(self, n, r):
                self._n, self._r = n, r

            def size(self):
                return self._n

            def rank(self):
                return self._r

        # 2 output rows across 4 ranks: ranks 2 and 3 get a zero-width share.
        qlin = nn.Linear(128, 2, bias=True).to_quantized(group_size=32, bits=4)
        for r in range(4):
            m = nn.Linear(128, 2, bias=True).to_quantized(group_size=32, bits=4)
            m.update(qlin.parameters())
            with self.assertRaises(ValueError):
                shard_inplace(m, "all-to-sharded", group=_FakeGroup(4, r))

    def test_shard_inplace_matches_shard_linear_quantized(self):
        # Both APIs must resolve the same quantized shard sizes.
        if not mx.cuda.is_available():

            class _FakeGroup:
                def __init__(self, n, r):
                    self._n, self._r = n, r

                def size(self):
                    return self._n

                def rank(self):
                    return self._r

            def _shard_linear_sizes(module, sharding, N):
                try:
                    return [
                        (
                            shard_linear(
                                module, sharding, group=_FakeGroup(N, r)
                            ).weight.shape[0 if sharding == "all-to-sharded" else 1]
                        )
                        for r in range(N)
                    ], None
                except ValueError as e:
                    return None, str(e)

            def _shard_inplace_sizes(reference, module_factory, sharding, N):
                sizes = []
                for r in range(N):
                    m = module_factory()
                    m.update(reference.parameters())
                    try:
                        shard_inplace(m, sharding, group=_FakeGroup(N, r))
                    except ValueError as e:
                        return None, str(e)
                    sizes.append(
                        m.weight.shape[0 if sharding == "all-to-sharded" else 1]
                    )
                return sizes, None

            for group_size in (32, 64):
                for dim_mult in (3, 5, 12, 17):
                    dim = group_size * dim_mult
                    for N in (2, 3, 5, 8):
                        qlin_out = nn.Linear(group_size, dim, bias=True).to_quantized(
                            group_size=group_size, bits=4
                        )
                        qlin_in = nn.Linear(dim, group_size, bias=True).to_quantized(
                            group_size=group_size, bits=4
                        )

                        sl_sizes, sl_err = _shard_linear_sizes(
                            qlin_out, "all-to-sharded", N
                        )
                        si_sizes, si_err = _shard_inplace_sizes(
                            qlin_out,
                            lambda: nn.Linear(group_size, dim, bias=True).to_quantized(
                                group_size=group_size, bits=4
                            ),
                            "all-to-sharded",
                            N,
                        )
                        # The APIs must agree on acceptance and shard sizes.
                        self.assertEqual(
                            sl_err is None,
                            si_err is None,
                            f"all-to-sharded raise-disagreement: dim={dim}, "
                            f"group_size={group_size}, N={N}: "
                            f"shard_linear={sl_err!r} shard_inplace={si_err!r}",
                        )
                        if sl_err is None:
                            self.assertEqual(
                                sl_sizes,
                                si_sizes,
                                f"all-to-sharded disagreement: dim={dim}, "
                                f"group_size={group_size}, N={N}",
                            )

                        sl_in_sizes, sl_in_err = _shard_linear_sizes(
                            qlin_in, "sharded-to-all", N
                        )
                        si_in_sizes, si_in_err = _shard_inplace_sizes(
                            qlin_in,
                            lambda: nn.Linear(dim, group_size, bias=True).to_quantized(
                                group_size=group_size, bits=4
                            ),
                            "sharded-to-all",
                            N,
                        )
                        self.assertEqual(
                            sl_in_err is None,
                            si_in_err is None,
                            f"sharded-to-all raise-disagreement: dim={dim}, "
                            f"group_size={group_size}, N={N}: "
                            f"shard_linear={sl_in_err!r} shard_inplace={si_in_err!r}",
                        )
                        if sl_in_err is None:
                            self.assertEqual(
                                sl_in_sizes,
                                si_in_sizes,
                                f"sharded-to-all disagreement: dim={dim}, "
                                f"group_size={group_size}, N={N}",
                            )

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
