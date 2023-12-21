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

    def test_l1_loss(self):
        predictions = mx.array([0.5, 0.2, 0.9, 0.0])
        targets = mx.array([0.5, 0.2, 0.9, 0.0])
        losses = nn.losses.l1_loss(predictions, targets, reduction="none")
        self.assertEqual(losses, 0.0)

    def test_mse_loss(self):
        predictions = mx.array([0.5, 0.2, 0.9, 0.0])
        targets = mx.array([0.7, 0.1, 0.8, 0.2])

        # Test with reduction 'none'
        losses_none = nn.losses.mse_loss(predictions, targets, reduction="none")
        expected_none = mx.array([0.04, 0.01, 0.01, 0.04])
        self.assertTrue(mx.allclose(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.mse_loss(predictions, targets, reduction="mean")
        expected_mean = mx.mean(expected_none)
        self.assertEqual(losses_mean, expected_mean)

        # Test with reduction 'sum'
        losses_sum = nn.losses.mse_loss(predictions, targets, reduction="sum")
        expected_sum = mx.sum(expected_none)
        self.assertEqual(losses_sum, expected_sum)

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

    def test_binary_cross_entropy(self):
        inputs = mx.array([[0.5, 0.5, 0.2, 0.9], [0.1, 0.3, 0.5, 0.5]])
        targets = mx.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

        # Test with reduction 'none'
        losses_none = nn.losses.binary_cross_entropy(inputs, targets, reduction="none")
        expected_none = mx.array(
            [
                [
                    0.6931471824645996,
                    0.6931471824645996,
                    0.2231435477733612,
                    0.10536054521799088,
                ],
                [
                    2.3025851249694824,
                    0.3566749691963196,
                    0.6931471824645996,
                    0.6931471824645996,
                ],
            ]
        )
        self.assertTrue(mx.allclose(losses_none, expected_none, rtol=1e-5, atol=1e-8))

        # Test with reduction 'mean'
        losses_mean = nn.losses.binary_cross_entropy(inputs, targets, reduction="mean")
        expected_mean = mx.mean(expected_none)
        self.assertTrue(mx.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.binary_cross_entropy(inputs, targets, reduction="sum")
        expected_sum = mx.sum(expected_none)
        self.assertTrue(mx.allclose(losses_sum, expected_sum))

    def test_bce_loss_module(self):
        inputs = mx.array([[0.5, 0.5, 0.2, 0.9], [0.1, 0.3, 0.5, 0.5]])
        targets = mx.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

        # Test with reduction 'none'
        loss_module_none = nn.losses.BCELoss(reduction="none")
        losses_none = loss_module_none(inputs, targets)
        expected_none = mx.array(
            [
                [
                    0.6931471824645996,
                    0.6931471824645996,
                    0.2231435477733612,
                    0.10536054521799088,
                ],
                [
                    2.3025851249694824,
                    0.3566749691963196,
                    0.6931471824645996,
                    0.6931471824645996,
                ],
            ]
        )
        self.assertTrue(mx.allclose(losses_none, expected_none, rtol=1e-5, atol=1e-8))

        # Test with reduction 'mean'
        loss_module_mean = nn.losses.BCELoss(reduction="mean")
        losses_mean = loss_module_mean(inputs, targets)
        expected_mean = mx.mean(expected_none)
        self.assertTrue(mx.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        loss_module_sum = nn.losses.BCELoss(reduction="sum")
        losses_sum = loss_module_sum(inputs, targets)
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

    def test_instance_norm(self):
        # Test InstanceNorm1d
        x = mx.array(
            [
                [
                    [-0.0119524, -0.500331, 1.12958, 1.39955],
                    [1.1263, 0.517899, -0.21413, 0.891329],
                    [2.02223, -1.21143, -2.48738, 1.63289],
                ],
                [
                    [0.241417, -1.42512, 2.739, -1.23175],
                    [-0.619157, 0.970817, -1.2506, 0.32756],
                    [-0.77484, -1.31352, 1.56844, 1.13969],
                ],
            ]
        )
        inorm = nn.InstanceNorm(num_features=3)
        y = inorm(x)
        expected_y = [
            [
                [-0.657082, -1.27879, 0.796097, 1.13978],
                [1.07593, -0.123075, -1.56572, 0.61286],
                [1.0712, -0.632503, -1.30476, 0.866066],
            ],
            [
                [0.0964433, -0.904773, 1.59693, -0.788599],
                [-0.557908, 1.30444, -1.29751, 0.550987],
                [-0.759886, -1.20013, 1.15521, 0.804804],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))
        # Test InstanceNorm2d
        x = mx.array(
            [
                [
                    [
                        [-0.458824, -0.447996, 0.0486988],
                        [1.13049, 0.301795, -2.23876],
                        [0.0986325, -1.25257, -0.329399],
                    ],
                    [
                        [0.483254, -0.176577, -0.0611224],
                        [0.345315, 0.99207, -0.758631],
                        [-1.82973, 0.154442, -0.319107],
                    ],
                    [
                        [-0.58611, -0.622545, 1.8845],
                        [-0.926389, -0.184927, -1.12639],
                        [-0.241765, -0.556204, 0.830584],
                    ],
                ],
                [
                    [
                        [1.04407, 0.0800776, 0.782321],
                        [0.671423, -0.110299, 0.159905],
                        [0.810252, 0.182597, -0.0621687],
                    ],
                    [
                        [0.073752, 1.2513, -0.444367],
                        [-1.21689, -1.42248, 0.516452],
                        [1.50456, 0.0576239, 0.184253],
                    ],
                    [
                        [0.407081, 1.20627, 0.563132],
                        [-1.88979, 1.17838, -0.539121],
                        [1.08659, 0.973883, 0.784216],
                    ],
                ],
            ]
        )
        inorm = nn.InstanceNorm(num_features=3)
        y = inorm(x)
        expected_y = [
            [
                [
                    [-0.120422, -0.108465, 0.440008],
                    [1.63457, 0.719488, -2.08591],
                    [0.495147, -0.996913, 0.0224944],
                ],
                [
                    [0.801504, -0.0608616, 0.0900314],
                    [0.621224, 1.4665, -0.821576],
                    [-2.22144, 0.371763, -0.247141],
                ],
                [
                    [-0.463984, -0.504602, 2.29032],
                    [-0.843336, -0.0167355, -1.0663],
                    [-0.0800997, -0.430644, 1.11538],
                ],
            ],
            [
                [
                    [1.59749, -0.776381, 0.95293],
                    [0.679838, -1.24519, -0.579803],
                    [1.02171, -0.523923, -1.12667],
                ],
                [
                    [0.0190289, 1.28291, -0.537076],
                    [-1.36624, -1.5869, 0.494185],
                    [1.55474, 0.00171834, 0.137631],
                ],
                [
                    [-0.012331, 0.817234, 0.149652],
                    [-2.39651, 0.78829, -0.994498],
                    [0.693007, 0.576016, 0.37914],
                ],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))
        # Test InstanceNorm3d
        x = mx.array(
            [
                [
                    [
                        [[0.777621, -2.1722], [-1.41317, 0.284446]],
                        [[0.11, -0.837743], [-2.40205, 0.336682]],
                        [[0.789185, -1.42998], [-0.459489, 0.0298199]],
                    ],
                    [
                        [[0.528145, 0.128192], [0.476288, -0.649858]],
                        [[-0.12431, 1.93502], [-1.25873, -0.261986]],
                        [[-1.63747, -1.73247], [-2.15559, 0.10275]],
                    ],
                    [
                        [[-1.56133, 0.153862], [-1.20411, 0.152112]],
                        [[1.18768, 0.00236324], [-2.04243, 1.54289]],
                        [[0.67917, -0.402572], [-0.249959, -0.821897]],
                    ],
                ],
                [
                    [
                        [[-2.12354, 0.317797], [-0.146628, 0.0329215]],
                        [[-1.55784, 2.41031], [0.226341, 0.265387]],
                        [[0.990317, 0.475161], [-1.37804, -0.501041]],
                    ],
                    [
                        [[0.643973, -0.682916], [-0.987925, 1.54086]],
                        [[0.71179, -0.290786], [0.057712, -0.742304]],
                        [[-0.399875, -1.10479], [1.40097, 0.0723374]],
                    ],
                    [
                        [[0.72391, 0.016364], [0.573199, 0.213092]],
                        [[-0.0678402, 0.00449439], [-1.58342, 1.28133]],
                        [[-0.357647, -1.07389], [0.141618, -0.386141]],
                    ],
                ],
            ]
        )
        inorm = nn.InstanceNorm(num_features=3)
        y = inorm(x)
        expected_y = [
            [
                [
                    [[1.23593, -1.54739], [-0.831204, 0.770588]],
                    [[0.605988, -0.288258], [-1.76427, 0.819875]],
                    [[1.24684, -0.847068], [0.0686449, 0.530334]],
                ],
                [
                    [[0.821849, 0.462867], [0.775304, -0.23548]],
                    [[0.236231, 2.0846], [-0.78198, 0.112659]],
                    [[-1.12192, -1.20719], [-1.58697, 0.440032]],
                ],
                [
                    [[-1.30944, 0.357126], [-0.962338, 0.355425]],
                    [[1.36163, 0.209922], [-1.77689, 1.70677]],
                    [[0.867539, -0.183531], [-0.0352458, -0.590967]],
                ],
            ],
            [
                [
                    [[-1.75315, 0.343736], [-0.0551618, 0.0990544]],
                    [[-1.26726, 2.14101], [0.265184, 0.298721]],
                    [[0.921369, 0.478897], [-1.11283, -0.35957]],
                ],
                [
                    [[0.733967, -0.822472], [-1.18025, 1.78602]],
                    [[0.813517, -0.362504], [0.0462839, -0.892134]],
                    [[-0.490465, -1.31732], [1.62192, 0.0634394]],
                ],
                [
                    [[1.04349, 0.080661], [0.838402, 0.348368]],
                    [[-0.033924, 0.0645089], [-2.09632, 1.80203]],
                    [[-0.428293, -1.40296], [0.251107, -0.467067]],
                ],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))

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
            [mx.array([1.0, -1.0, 0.0, 0.5])],
            nn.PReLU(),
            mx.array([1.0, -0.25, 0.0, 0.5]),
        )

    def test_mish(self):
        self.assertEqualArray(
            [mx.array([1.0, -1.0, 0.0, 0.5])],
            nn.Mish(),
            mx.array([0.8651, -0.3034, 0.0000, 0.3752]),
        )


if __name__ == "__main__":
    unittest.main()
