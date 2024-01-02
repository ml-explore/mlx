# Copyright © 2023 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten


class TestBase(mlx_tests.MLXTestCase):
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

    def test_load_from_weights(self):
        m = nn.Linear(2, 2)

        # Too few weights
        weights = [("weight", mx.ones((2, 2)))]
        with self.assertRaises(ValueError):
            m.load_weights(weights)

        m.load_weights(weights, strict=False)
        self.assertTrue(mx.array_equal(m.weight, weights[0][1]))

        # Wrong name
        with self.assertRaises(ValueError):
            m.load_weights([("weihgt", mx.ones((2, 2)))])

        # Ok
        m.load_weights([("weihgt", mx.ones((2, 2)))], strict=False)

        # Too many weights
        with self.assertRaises(ValueError):
            m.load_weights(
                [
                    ("weight", mx.ones((2, 2))),
                    ("bias", mx.ones((2,))),
                    ("bias2", mx.ones((2,))),
                ]
            )

        # Wrong shape
        with self.assertRaises(ValueError):
            m.load_weights(
                [
                    ("weight", mx.ones((2, 2))),
                    ("bias", mx.ones((2, 1))),
                ]
            )

        # Wrong type
        with self.assertRaises(ValueError):
            m.load_weights(
                [
                    ("weight", mx.ones((2, 2))),
                    ("bias", 3),
                ]
            )


class TestLayers(mlx_tests.MLXTestCase):

    def test_identity(self):
        inputs = mx.zeros((10, 4))
        layer = nn.Identity()
        outputs = layer(inputs)
        self.assertEqual(tuple(inputs.shape), tuple(outputs.shape))

    def test_linear(self):
        inputs = mx.zeros((10, 4))
        layer = nn.Linear(input_dims=4, output_dims=8)
        outputs = layer(inputs)
        self.assertEqual(tuple(outputs.shape), (10, 8))

    def test_bilinear(self):
        inputs1 = mx.zeros((10, 2))
        inputs2 = mx.zeros((10, 4))
        layer = nn.Bilinear(input1_dims=2, input2_dims=4, output_dims=6)
        outputs = layer(inputs1, inputs2)
        self.assertEqual(tuple(outputs.shape), (10, 6))

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

    def test_batch_norm(self):
        mx.random.seed(42)
        x = mx.random.normal((5, 4), dtype=mx.float32)

        # Batch norm
        bn = nn.BatchNorm(num_features=4, affine=True)
        self.assertTrue(mx.allclose(bn.running_mean, mx.zeros_like(bn.running_mean)))
        self.assertTrue(mx.allclose(bn.running_var, mx.ones_like(bn.running_var)))
        y = bn(x)
        expected_y = mx.array(
            [
                [-0.439520, 1.647328, -0.955515, 1.966031],
                [-1.726690, -1.449826, -0.234026, -0.723364],
                [0.938414, -0.349603, -0.354470, -0.175369],
                [0.305006, 0.234914, -0.393017, -0.459385],
                [0.922789, -0.082813, 1.937028, -0.607913],
            ],
        )
        expected_mean = mx.array([0.008929, 0.005680, -0.016092, 0.027778])
        expected_var = mx.array([0.928435, 1.00455, 1.04117, 0.94258])
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(mx.allclose(y, expected_y, atol=1e-5))
        self.assertTrue(mx.allclose(bn.running_mean, expected_mean, atol=1e-5))
        self.assertTrue(mx.allclose(bn.running_var, expected_var, atol=1e-5))

        # test eval mode
        bn.eval()
        y = bn(x)
        expected_y = mx.array(
            [
                [-0.15984, 1.73159, -1.25456, 1.57891],
                [-0.872193, -1.4281, -0.414439, -0.228678],
                [0.602743, -0.30566, -0.554687, 0.139639],
                [0.252199, 0.29066, -0.599572, -0.0512532],
                [0.594096, -0.0334829, 2.11359, -0.151081],
            ]
        )

        self.assertTrue(x.shape == y.shape)
        self.assertTrue(mx.allclose(y, expected_y, atol=1e-5))

        # test_no_affine
        bn = nn.BatchNorm(num_features=4, affine=False)
        y = bn(x)
        expected_y = mx.array(
            [
                [-0.439520, 1.647328, -0.955515, 1.966031],
                [-1.726690, -1.449826, -0.234026, -0.723364],
                [0.938414, -0.349603, -0.354470, -0.175369],
                [0.305006, 0.234914, -0.393017, -0.459385],
                [0.922789, -0.082813, 1.937028, -0.607913],
            ]
        )
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(mx.allclose(y, expected_y, atol=1e-5))

        # test with 3D input
        mx.random.seed(42)
        N = 2
        L = 4
        C = 5
        x = mx.random.normal((N, L, C), dtype=mx.float32)

        # Batch norm
        bn = nn.BatchNorm(num_features=C, affine=True)
        self.assertTrue(mx.allclose(bn.running_mean, mx.zeros_like(bn.running_mean)))
        self.assertTrue(mx.allclose(bn.running_var, mx.ones_like(bn.running_var)))
        y = bn(x)
        self.assertTrue(x.shape == y.shape)
        expected_y = mx.array(
            [
                [
                    [-0.335754, 0.342054, 1.02653, 0.628588, -1.63899],
                    [1.92092, 0.432319, 0.343043, 1.95489, 1.0696],
                    [-0.853748, 1.3661, 0.868569, 0.0199196, -0.887284],
                    [0.459206, -0.684822, -0.706354, -0.271531, 0.566341],
                ],
                [
                    [-0.921179, 0.684951, -0.77466, -0.490372, -0.247032],
                    [1.10839, -2.13179, 0.628924, -1.62639, -0.539708],
                    [-0.348943, 0.412194, -2.03818, 0.524972, 1.64568],
                    [-1.02889, -0.421, 0.652127, -0.740079, 0.0313996],
                ],
            ]
        )
        self.assertTrue(mx.allclose(y, expected_y, atol=1e-5))
        expected_mean = mx.array(
            [[[0.00207845, -5.3259e-05, 0.04755, -0.0697296, 0.0236228]]]
        )
        expected_var = mx.array([[[0.968415, 1.05322, 0.96913, 0.932305, 0.967224]]])
        self.assertTrue(mx.allclose(bn.running_mean, expected_mean, atol=1e-5))
        self.assertTrue(mx.allclose(bn.running_var, expected_var, atol=1e-5))

        x = mx.random.normal((N, L, C, L, C), dtype=mx.float32)
        with self.assertRaises(ValueError):
            y = bn(x)

        # Check that the running stats are in the param dictionary
        bn_parameters = bn.parameters()
        self.assertIn("running_mean", bn_parameters)
        self.assertIn("running_var", bn_parameters)
        self.assertIn("weight", bn_parameters)
        self.assertIn("bias", bn_parameters)

        bn_trainable = bn.trainable_parameters()
        self.assertNotIn("running_mean", bn_trainable)
        self.assertNotIn("running_var", bn_trainable)
        self.assertIn("weight", bn_trainable)
        self.assertIn("bias", bn_trainable)

        bn.unfreeze()
        bn_trainable = bn.trainable_parameters()
        self.assertNotIn("running_mean", bn_trainable)
        self.assertNotIn("running_var", bn_trainable)
        self.assertIn("weight", bn_trainable)
        self.assertIn("bias", bn_trainable)

    def test_batch_norm_stats(self):
        batch_size = 2
        num_features = 4
        h = 3
        w = 3
        momentum = 0.1

        batch_norm = nn.BatchNorm(num_features)

        batch_norm.train()
        running_mean = np.array(batch_norm.running_mean)
        running_var = np.array(batch_norm.running_var)

        data = mx.random.normal((batch_size, num_features))

        normalized_data = batch_norm(data)
        np_data = np.array(data)
        means = np.mean(np_data, axis=0)
        variances = np.var(np_data, axis=0)
        running_mean = (1 - momentum) * running_mean + momentum * means
        running_var = (1 - momentum) * running_var + momentum * variances
        self.assertTrue(np.allclose(batch_norm.running_mean, running_mean, atol=1e-5))
        self.assertTrue(np.allclose(batch_norm.running_var, running_var, atol=1e-5))

        batch_norm = nn.BatchNorm(num_features)

        batch_norm.train()
        running_mean = np.array(batch_norm.running_mean)
        running_var = np.array(batch_norm.running_var)
        data = mx.random.normal((batch_size, h, w, num_features))

        normalized_data = batch_norm(data)
        np_data = np.array(data)
        means = np.mean(np_data, axis=(0, 1, 2))
        variances = np.var(np_data, axis=(0, 1, 2))
        running_mean = (1 - momentum) * running_mean + momentum * means
        running_var = (1 - momentum) * running_var + momentum * variances
        self.assertTrue(np.allclose(batch_norm.running_mean, running_mean, atol=1e-5))
        self.assertTrue(np.allclose(batch_norm.running_var, running_var, atol=1e-5))

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

    def test_sin_pe(self):
        m = nn.SinusoidalPositionalEncoding(16, min_freq=0.01)
        x = mx.arange(10)
        y = m(x)

        self.assertEqual(y.shape, [10, 16])
        similarities = y @ y.T
        self.assertLess(
            mx.abs(similarities[mx.arange(10), mx.arange(10)] - 1).max(), 1e-5
        )

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

    def test_softmax(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softmax(x)
        epsilon = 1e-4
        expected_y = mx.array([0.6652, 0.0900, 0.2447])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_softplus(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softplus(x)
        epsilon = 1e-4
        expected_y = mx.array([1.3133, 0.3133, 0.6931])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.dtype, mx.float32)

    def test_softsign(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softsign(x)
        epsilon = 1e-4
        expected_y = mx.array([0.5, -0.5, 0.0])
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

    def test_log_softmax(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = nn.log_softmax(x)
        epsilon = 1e-4
        expected_y = mx.array([-2.4076, -1.4076, -0.4076])
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

    def test_hardswish(self):
        x = mx.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        y = nn.hardswish(x)
        epsilon = 1e-4
        expected_y = mx.array([0.0, -0.375, 0.0, 1.125, 3.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, [5])
        self.assertEqual(y.dtype, mx.float32)

    def test_rope(self):
        for kwargs in [{}, {"traditional": False}, {"base": 10000}, {"scale": 0.25}]:
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

    def test_dropout(self):
        x = mx.ones((2, 4))
        y = nn.Dropout(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float32)

        x = mx.ones((2, 4), dtype=mx.bfloat16)
        y = nn.Dropout(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4), dtype=mx.float16)
        y = nn.Dropout(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float16)

    def test_dropout2d(self):
        x = mx.ones((2, 4, 4, 4))
        y = nn.Dropout2d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float32)

        x = mx.ones((2, 4, 4, 4), dtype=mx.bfloat16)
        y = nn.Dropout2d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4, 4, 4), dtype=mx.float16)
        y = nn.Dropout2d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float16)

    def test_dropout3d(self):
        x = mx.ones((2, 4, 4, 4, 4))
        y = nn.Dropout3d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float32)

        x = mx.ones((2, 4, 4, 4, 4), dtype=mx.bfloat16)
        y = nn.Dropout3d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4, 4, 4, 4), dtype=mx.float16)
        y = nn.Dropout3d(0.5)(x)
        self.assertTrue(y.shape, x.shape)
        self.assertTrue(y.dtype, mx.float16)


if __name__ == "__main__":
    unittest.main()
