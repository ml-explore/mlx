# Copyright © 2023 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten


class TestNN(mlx_tests.MLXTestCase):
    def test_linear(self):
        inputs = mx.zeros((10, 4))
        layer = nn.Linear(input_dims=4, output_dims=8)
        outputs = layer(inputs)
        self.assertEqual(tuple(outputs.shape), (10, 8))

    def test_cross_entropy(self):
        logits = mx.array([[0.0, -float("inf")], [-float("inf"), 0.0]])
        targets = mx.array([0, 1])

        # Test with reduction 'none'
        losses_none = nn.losses.cross_entropy(logits, targets, reduction="none")
        expected_none = mx.array([0.0, 0.0])
        self.assertTrue(mx.array_equal(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.cross_entropy(logits, targets, reduction="mean")
        expected_mean = mx.mean(expected_none)
        self.assertEqual(losses_mean, expected_mean)

        # Test with reduction 'sum'
        losses_sum = nn.losses.cross_entropy(logits, targets, reduction="sum")
        expected_sum = mx.sum(expected_none)
        self.assertEqual(losses_sum, expected_sum)

        # Test cases with weights and no label smoothing
        logits = mx.array([[2.0, -1.0], [-1.0, 2.0]])
        targets = mx.array([0, 1])
        weights = mx.array([1.0, 2.0])

        # Reduction 'none'
        losses_none = nn.losses.cross_entropy(
            logits,
            targets,
            weights=weights,
            reduction="none",
        )
        expected_none = mx.array([0.04858735, 0.0971747])  # Calculated losses
        self.assertTrue(
            np.allclose(losses_none, expected_none, atol=1e-5),
            "Test case failed for cross_entropy loss --reduction='none' --weights=[1.0, 2.0]",
        )

        # Reduction 'mean'
        losses_mean = nn.losses.cross_entropy(
            logits,
            targets,
            weights=weights,
            reduction="mean",
        )
        expected_mean = mx.mean(expected_none)
        self.assertTrue(
            np.allclose(losses_mean, expected_mean, atol=1e-5),
            "Test case failed for cross_entropy loss --reduction='mean' --weights=[1.0, 2.0]",
        )

        # Reduction 'sum'
        losses_sum = nn.losses.cross_entropy(
            logits,
            targets,
            weights=weights,
            reduction="sum",
        )
        expected_sum = mx.sum(expected_none)
        self.assertTrue(
            np.allclose(losses_sum, expected_sum, atol=1e-5),
            "Test case failed for cross_entropy loss --reduction='sum' --weights=[1.0, 2.0]",
        )

        # Test case with equal weights and label smoothing > 0
        logits = mx.array(
            [[0, 0.2, 0.7, 0.1, 0], [0, 0.9, 0.2, 0.2, 1], [1, 0.2, 0.7, 0.9, 1]]
        )
        target = mx.array([2, 1, 0])

        losses_none = nn.losses.cross_entropy(
            logits, target, label_smoothing=0.3, reduction="none"
        )
        expected_none = mx.array([1.29693, 1.38617, 1.48176])
        self.assertTrue(
            mx.allclose(expected_none, losses_none),
            "Test case failed for cross_entropy --label_smoothing=0.3 --reduction='none'",
        )

        expected_mean = mx.mean(expected_none)
        losses_mean = nn.losses.cross_entropy(
            logits, target, label_smoothing=0.3, reduction="mean"
        )
        self.assertTrue(
            mx.allclose(losses_mean, expected_mean),
            "Test case failed for cross_entropy --label_smoothing=0.3 --reduction='mean'",
        )

        expected_sum = mx.sum(expected_none)
        losses_sum = nn.losses.cross_entropy(
            logits, target, label_smoothing=0.3, reduction="sum"
        )
        self.assertTrue(
            mx.allclose(losses_sum, expected_sum),
            "Test case failed for cross_entropy --label_smoothing=0.3 --reduction='sum'",
        )

    def test_l1_loss(self):
        predictions = mx.array([0.5, 0.2, 0.9, 0.0])
        targets = mx.array([0.5, 0.2, 0.9, 0.0])

        # Expected result
        expected_none = mx.array([0, 0, 0, 0]).astype(mx.float32)
        expected_sum = mx.sum(expected_none)
        expected_mean = mx.mean(expected_none)

        losses = nn.losses.l1_loss(predictions, targets, reduction="none")
        self.assertTrue(
            mx.array_equal(losses, expected_none),
            "Test failed for l1_loss --reduction='none'",
        )

        losses = nn.losses.l1_loss(predictions, targets, reduction="sum")
        self.assertTrue(mx.array_equal(losses, expected_sum))

        losses = nn.losses.l1_loss(predictions, targets, reduction="mean")
        self.assertTrue(mx.array_equal(losses, expected_mean))

    def test_mse_loss(self):
        predictions = mx.array([0.5, 0.2, 0.9, 0.0])
        targets = mx.array([0.7, 0.1, 0.8, 0.2])

        expected_none = mx.array([0.04, 0.01, 0.01, 0.04])
        expected_mean = mx.mean(expected_none)
        expected_sum = mx.sum(expected_none)

        # Test with reduction 'none'
        losses_none = nn.losses.mse_loss(predictions, targets, reduction="none")
        self.assertTrue(
            np.allclose(losses_none, expected_none, 1e-5),
            "Test case failed for mse_loss --reduction='none'",
        )

        # Test with reduction 'mean'
        losses_mean = nn.losses.mse_loss(predictions, targets, reduction="mean")
        self.assertEqual(
            losses_mean,
            expected_mean,
            "Test case failed for mse_loss --reduction='mean'",
        )

        # Test with reduction 'sum'
        losses_sum = nn.losses.mse_loss(predictions, targets, reduction="sum")
        self.assertEqual(
            losses_sum, expected_sum, "Test case failed for mse_loss --reduction='sum'"
        )

    def test_smooth_l1_loss(self):
        predictions = mx.array([1.5, 2.5, 0.5, 3.5])
        targets = mx.array([1.0, 2.0, 0.5, 2.5])
        beta = 1.0

        # Expected results
        expected_none = mx.array([0.125, 0.125, 0.0, 0.5])
        expected_sum = mx.sum(expected_none)
        expected_mean = mx.mean(expected_none)

        # Test with reduction 'none'
        loss_none = nn.losses.smooth_l1_loss(
            predictions, targets, beta, reduction="none"
        )
        self.assertTrue(
            mx.array_equal(loss_none, expected_none),
            "Test case failed for smooth_l1_loss --reduction='none'",
        )

        # Test with reduction 'sum'
        loss_sum = nn.losses.smooth_l1_loss(predictions, targets, beta, reduction="sum")
        self.assertEqual(
            loss_sum,
            expected_sum,
            "Test case failed for smooth_l1_loss --reduction='sum'",
        )

        # Test with reduction 'mean'
        loss_mean = nn.losses.smooth_l1_loss(
            predictions, targets, beta, reduction="mean"
        )
        self.assertEqual(
            loss_mean,
            expected_mean,
            "Test case failed for smooth_l1_loss --reduction='mean'",
        )

    def test_nll_loss(self):
        logits = mx.array([[0.0, -float("inf")], [-float("inf"), 0.0]])
        targets = mx.array([0, 1])

        # Test with reduction 'none'
        losses_none = nn.losses.nll_loss(logits, targets, reduction="none")
        expected_none = mx.array([0.0, 0.0])
        self.assertTrue(mx.array_equal(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.nll_loss(logits, targets, reduction="mean")
        expected_mean = mx.mean(expected_none)
        self.assertEqual(losses_mean, expected_mean)

        # Test with reduction 'sum'
        losses_sum = nn.losses.nll_loss(logits, targets, reduction="sum")
        expected_sum = mx.sum(expected_none)
        self.assertEqual(losses_sum, expected_sum)

    def test_kl_div_loss(self):
        p_logits = mx.log(mx.array([[0.5, 0.5], [0.8, 0.2]]))
        q_logits = mx.log(mx.array([[0.5, 0.5], [0.2, 0.8]]))

        # Test with reduction 'none'
        losses_none = nn.losses.kl_div_loss(p_logits, q_logits, reduction="none")
        expected_none = mx.array([0.0, 0.831777])
        self.assertTrue(mx.allclose(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.kl_div_loss(p_logits, q_logits, reduction="mean")
        expected_mean = mx.mean(expected_none)
        self.assertTrue(mx.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.kl_div_loss(p_logits, q_logits, reduction="sum")
        expected_sum = mx.sum(expected_none)
        self.assertTrue(mx.allclose(losses_sum, expected_sum))

    def test_triplet_loss(self):
        anchors = mx.array([[1, 2, 3], [1, 2, 3]])
        positives = mx.array([[4, 5, 6], [0, -1, 2]])
        negatives = mx.array([[7, 8, 9], [3, 2, 3]])

        # Test with reduction 'none'
        losses_none = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="none"
        )
        expected_none = mx.array([0, 2.31662])
        self.assertTrue(mx.allclose(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="mean"
        )
        expected_mean = mx.mean(expected_none)
        self.assertTrue(mx.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="sum"
        )
        expected_sum = mx.sum(expected_none)
        self.assertTrue(mx.allclose(losses_sum, expected_sum))

    def test_gelu(self):
        inputs = [1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414]

        # From: jax.nn.gelu(np.array(inputs), approximate=False)
        expected = np.array(
            [1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383]
        )

        out = nn.GELU()(mx.array(inputs))
        self.assertTrue(np.allclose(out, expected))

        # Crudely check the approximations
        x = mx.arange(-6.0, 6.0, 12 / 100)
        y = nn.gelu(x)
        y_hat1 = nn.gelu_approx(x)
        y_hat2 = nn.gelu_fast_approx(x)
        self.assertLess(mx.abs(y - y_hat1).max(), 0.0003)
        self.assertLess(mx.abs(y - y_hat2).max(), 0.02)

    def test_group_norm(self):
        x = mx.arange(100, dtype=mx.float32)
        x = x.reshape(1, 10, 10, 1)
        x = mx.broadcast_to(x, (2, 10, 10, 4))
        x = mx.concatenate([x, 0.5 * x], axis=-1)

        # Group norm in groups last mode
        g = nn.GroupNorm(2, 8)
        y = g(x)
        means = y.reshape(2, -1, 2).mean(axis=1)
        var = y.reshape(2, -1, 2).var(axis=1)
        self.assertTrue(np.allclose(means, np.zeros_like(means), atol=1e-6))
        self.assertTrue(np.allclose(var, np.ones_like(var), atol=1e-6))
        g.weight = g.weight * 2
        g.bias = g.bias + 3
        y = g(x)
        means = y.reshape(2, -1, 2).mean(axis=1)
        var = y.reshape(2, -1, 2).var(axis=1)
        self.assertTrue(np.allclose(means, 3 * np.ones_like(means), atol=1e-6))
        self.assertTrue(np.allclose(var, 4 * np.ones_like(var), atol=1e-6))

        # Group norm in groups first mode
        g = nn.GroupNorm(2, 8, pytorch_compatible=True)
        y = g(x)
        means = y.reshape(2, -1, 2, 4).mean(axis=(1, -1))
        var = y.reshape(2, -1, 2, 4).var(axis=(1, -1))
        self.assertTrue(np.allclose(means, np.zeros_like(means), atol=1e-6))
        self.assertTrue(np.allclose(var, np.ones_like(var), atol=1e-6))
        g.weight = g.weight * 2
        g.bias = g.bias + 3
        y = g(x)
        means = y.reshape(2, -1, 2, 4).mean(axis=(1, -1))
        var = y.reshape(2, -1, 2, 4).var(axis=(1, -1))
        self.assertTrue(np.allclose(means, 3 * np.ones_like(means), atol=1e-6))
        self.assertTrue(np.allclose(var, 4 * np.ones_like(var), atol=1e-6))

    def test_conv1d(self):
        N = 5
        L = 12
        ks = 3
        C_in = 2
        C_out = 4
        x = mx.ones((N, L, C_in))
        c = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=ks)
        c.weight = mx.ones_like(c.weight)
        y = c(x)
        self.assertEqual(y.shape, [N, L - ks + 1, C_out])
        self.assertTrue(mx.allclose(y, mx.full(y.shape, ks * C_in, mx.float32)))

        c = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=ks, stride=2)
        y = c(x)
        self.assertEqual(y.shape, [N, (L - ks + 1) // 2, C_out])
        self.assertTrue("bias" in c.parameters())

        c = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=ks, bias=False)
        self.assertTrue("bias" not in c.parameters())

    def test_conv2d(self):
        x = mx.ones((4, 8, 8, 3))
        c = nn.Conv2d(3, 1, 8)
        y = c(x)
        self.assertEqual(y.shape, [4, 1, 1, 1])
        c.weight = mx.ones_like(c.weight) / 8 / 8 / 3
        y = c(x)
        self.assertTrue(np.allclose(y[:, 0, 0, 0], x.mean(axis=(1, 2, 3))))

        # 3x3 conv no padding stride 1
        c = nn.Conv2d(3, 8, 3)
        y = c(x)
        self.assertEqual(y.shape, [4, 6, 6, 8])
        self.assertLess(mx.abs(y - c.weight.sum((1, 2, 3))).max(), 1e-4)

        # 3x3 conv padding 1 stride 1
        c = nn.Conv2d(3, 8, 3, padding=1)
        y = c(x)
        self.assertEqual(y.shape, [4, 8, 8, 8])
        self.assertLess(mx.abs(y[:, 1:7, 1:7] - c.weight.sum((1, 2, 3))).max(), 1e-4)
        self.assertLess(
            mx.abs(y[:, 0, 0] - c.weight[:, 1:, 1:].sum(axis=(1, 2, 3))).max(),
            1e-4,
        )
        self.assertLess(
            mx.abs(y[:, 7, 7] - c.weight[:, :-1, :-1].sum(axis=(1, 2, 3))).max(),
            1e-4,
        )
        self.assertLess(
            mx.abs(y[:, 1:7, 7] - c.weight[:, :, :-1].sum(axis=(1, 2, 3))).max(),
            1e-4,
        )
        self.assertLess(
            mx.abs(y[:, 7, 1:7] - c.weight[:, :-1, :].sum(axis=(1, 2, 3))).max(),
            1e-4,
        )

        # 3x3 conv no padding stride 2
        c = nn.Conv2d(3, 8, 3, padding=0, stride=2)
        y = c(x)
        self.assertEqual(y.shape, [4, 3, 3, 8])
        self.assertLess(mx.abs(y - c.weight.sum((1, 2, 3))).max(), 1e-4)

    def test_sequential(self):
        x = mx.ones((10, 2))
        m = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
        y = m(x)
        self.assertEqual(y.shape, [10, 1])
        params = m.parameters()
        self.assertTrue("layers" in params)
        self.assertEqual(len(params["layers"]), 3)
        self.assertTrue("weight" in params["layers"][0])
        self.assertEqual(len(params["layers"][1]), 0)
        self.assertTrue("weight" in params["layers"][2])

        m.layers[1] = nn.relu
        y2 = m(x)
        self.assertTrue(mx.array_equal(y, y2))

    def test_module_utilities(self):
        m = nn.Sequential(
            nn.Sequential(nn.Linear(2, 10), nn.relu),
            nn.Sequential(nn.Linear(10, 10), nn.ReLU()),
            nn.Linear(10, 1),
            mx.sigmoid,
        )

        children = m.children()
        self.assertTrue(isinstance(children, dict))
        self.assertEqual(len(children), 1)
        self.assertTrue(isinstance(children["layers"], list))
        self.assertEqual(len(children["layers"]), 4)
        self.assertEqual(children["layers"][3], {})
        flat_children = tree_flatten(children, is_leaf=nn.Module.is_module)
        self.assertEqual(len(flat_children), 3)

        leaves = tree_flatten(m.leaf_modules(), is_leaf=nn.Module.is_module)
        self.assertEqual(len(leaves), 4)
        self.assertEqual(leaves[0][0], "layers.0.layers.0")
        self.assertEqual(leaves[1][0], "layers.1.layers.0")
        self.assertEqual(leaves[2][0], "layers.1.layers.1")
        self.assertEqual(leaves[3][0], "layers.2")
        self.assertTrue(leaves[0][1] is m.layers[0].layers[0])
        self.assertTrue(leaves[1][1] is m.layers[1].layers[0])
        self.assertTrue(leaves[2][1] is m.layers[1].layers[1])
        self.assertTrue(leaves[3][1] is m.layers[2])

        m.eval()

        def assert_not_training(k, m):
            self.assertFalse(m.training)

        m.apply_to_modules(assert_not_training)

        m.train()

        def assert_training(k, m):
            self.assertTrue(m.training)

        m.apply_to_modules(assert_training)

    def test_sin_pe(self):
        m = nn.SinusoidalPositionalEncoding(16, min_freq=0.01)
        x = mx.arange(10)
        y = m(x)

        self.assertEqual(y.shape, [10, 16])
        similarities = y @ y.T
        self.assertLess(
            mx.abs(similarities[mx.arange(10), mx.arange(10)] - 1).max(), 1e-5
        )

    def test_io(self):
        def make_model():
            return nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))

        m = make_model()
        tdir = tempfile.TemporaryDirectory()
        file = os.path.join(tdir.name, "model.npz")
        m.save_weights(file)
        m_load = make_model()
        m_load.load_weights(file)
        tdir.cleanup()

        eq_tree = tree_map(mx.array_equal, m.parameters(), m_load.parameters())
        self.assertTrue(all(tree_flatten(eq_tree)))

    def test_relu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.relu(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0])))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_leaky_relu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.leaky_relu(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, -0.01, 0.0])))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

        y = nn.LeakyReLU(negative_slope=0.1)(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, -0.1, 0.0])))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_elu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.elu(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6321, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

        y = nn.ELU(alpha=1.1)(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6953, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_relu6(self):
        x = mx.array([1.0, -1.0, 0.0, 7.0, -7.0])
        y = nn.relu6(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0, 6.0, 0.0])))
        self.assertEqual(y.shape, [5])
        self.assertEqual(y.dtype, mx.float32)

    def test_softplus(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softplus(x)
        epsilon = 1e-4
        expected_y = mx.array([1.3133, 0.3133, 0.6931])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_celu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.celu(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6321, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

        y = nn.CELU(alpha=1.1)(x)
        expected_y = mx.array([1.0, -0.6568, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_log_sigmoid(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.log_sigmoid(x)
        epsilon = 1e-4
        expected_y = mx.array([-0.3133, -1.3133, -0.6931])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_prelu(self):
        self.assertEqualArray(
            nn.PReLU()(mx.array([1.0, -1.0, 0.0, 0.5])),
            mx.array([1.0, -0.25, 0.0, 0.5]),
        )

    def test_mish(self):
        self.assertEqualArray(
            nn.Mish()(mx.array([1.0, -1.0, 0.0, 0.5])),
            mx.array([0.8651, -0.3034, 0.0000, 0.3752]),
        )

    def test_rope(self):
        for kwargs in [{}, {"traditional": False}, {"base": 10000}]:
            rope = nn.RoPE(4, **kwargs)
            shape = (1, 3, 4)
            x = mx.random.uniform(shape=shape)
            y = rope(x)
            self.assertTrue(y.shape, shape)
            self.assertTrue(y.dtype, mx.float32)

            y = rope(x, offset=3)
            self.assertTrue(y.shape, shape)

            y = rope(x.astype(mx.float16))
            self.assertTrue(y.dtype, mx.float16)

    def test_alibi(self):
        alibi = nn.ALiBi()
        shape = [1, 8, 20, 20]
        x = mx.random.uniform(shape=shape)
        y = alibi(x)
        self.assertTrue(y.shape, shape)
        self.assertTrue(y.dtype, mx.float32)

        y = alibi(x.astype(mx.float16))
        self.assertTrue(y.dtype, mx.float16)


if __name__ == "__main__":
    unittest.main()
