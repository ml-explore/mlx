# Copyright © 2023-2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_reduce


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

    def test_module_attributes(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.val = None
                self.initialize()

            def initialize(self):
                self.val = mx.array(1.0)

        model = Model()
        self.assertTrue(mx.array_equal(model.val, mx.array(1.0)))

        model.val = None
        self.assertEqual(model.val, None)

        model.val = mx.array([3])
        self.assertEqual(model.val.item(), 3)

    def test_model_with_dict(self):
        class DictModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = {"w1": mx.zeros((2, 2)), "w2": mx.ones((2, 2))}

        model = DictModule()
        params = tree_flatten(model.parameters(), destination={})
        self.assertEqual(len(params), 2)
        self.assertTrue(mx.array_equal(params["weights.w1"], mx.zeros((2, 2))))
        self.assertTrue(mx.array_equal(params["weights.w2"], mx.ones((2, 2))))

    def test_save_npz_weights(self):
        def make_model():
            return nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))

        m = make_model()
        tdir = tempfile.TemporaryDirectory()
        npz_file = os.path.join(tdir.name, "model.npz")
        m.save_weights(npz_file)
        m_load = make_model()
        m_load.load_weights(npz_file)

        # Eval before cleanup so model file is unlocked.
        mx.eval(m_load.state)
        tdir.cleanup()

        eq_tree = tree_map(mx.array_equal, m.parameters(), m_load.parameters())
        self.assertTrue(all(tree_flatten(eq_tree)))

    def test_save_safetensors_weights(self):
        def make_model():
            return nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU())

        m = make_model()
        tdir = tempfile.TemporaryDirectory()
        safetensors_file = os.path.join(tdir.name, "model.safetensors")
        m.save_weights(safetensors_file)
        m_load = make_model()
        m_load.load_weights(safetensors_file)

        # Eval before cleanup so model file is unlocked.
        mx.eval(m_load.state)
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

        # Empty weights is ok if strict is false
        m.load_weights([], strict=False)

    def test_module_state(self):
        m = nn.Linear(10, 1)
        m.state["hello"] = "world"
        self.assertEqual(m.state["hello"], "world")

    def test_chaining(self):
        m = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        pre_freeze_num_params = len(m.parameters())
        m.freeze().unfreeze()
        self.assertEqual(len(m.parameters()), pre_freeze_num_params)
        params_dict = m.parameters()

        self.assertFalse(m.update(params_dict).eval()._training)
        self.assertTrue(m.train()._training)

    def test_quantize(self):
        m = nn.Sequential(nn.Embedding(5, 256), nn.ReLU(), nn.Linear(256, 256))
        nn.quantize(m)
        self.assertTrue(isinstance(m.layers[0], nn.QuantizedEmbedding))
        self.assertTrue(isinstance(m.layers[1], nn.ReLU))
        self.assertTrue(isinstance(m.layers[2], nn.QuantizedLinear))

        m = nn.Sequential(nn.Embedding(5, 256), nn.ReLU(), nn.Linear(256, 256))
        nn.quantize(m, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        self.assertTrue(isinstance(m.layers[0], nn.Embedding))
        self.assertTrue(isinstance(m.layers[1], nn.ReLU))
        self.assertTrue(isinstance(m.layers[2], nn.QuantizedLinear))

        nn.quantize(m, group_size=32, mode="mxfp4")
        self.assertTrue(isinstance(m.layers[0], nn.QuantizedEmbedding))
        self.assertTrue(isinstance(m.layers[1], nn.ReLU))
        self.assertTrue(isinstance(m.layers[2], nn.QuantizedLinear))
        self.assertTrue(isinstance(m.layers[2].scales, mx.array))

    def test_quantize_freeze(self):
        lin = nn.Linear(512, 512)
        qlin = lin.to_quantized()
        qlin.unfreeze(keys=["scales"])
        size = tree_reduce(lambda acc, p: acc + p.size, qlin.trainable_parameters(), 0)
        self.assertTrue(size > 0)

    def test_grad_of_module(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = nn.Linear(3, 3)

        model = Model()

        def loss_fn(model):
            return model.m1(x).sum()

        x = mx.zeros((3,))
        mx.grad(loss_fn)(model)

    def test_update(self):
        m = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))

        # Updating non-existent parameters
        with self.assertRaises(ValueError):
            updates = {"layers": [{"value": 0}]}
            m.update(updates)

        with self.assertRaises(ValueError):
            updates = {"layers": ["hello"]}
            m.update(updates)

        # Wronge type
        with self.assertRaises(ValueError):
            updates = {"layers": [{"weight": "hi"}]}
            m.update(updates)

    def test_update_modules(self):
        m = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))

        # Updating non-existent modules should not be allowed by default
        with self.assertRaises(ValueError):
            m = m.update_modules({"values": [0, 1]})

        # Update wrong types
        with self.assertRaises(ValueError):
            m = m.update_modules({"layers": [0, 1]})

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.test = mx.array(1.0)
                self.list = [mx.array(1.0), mx.array(2.0)]

        m = MyModule()
        with self.assertRaises(ValueError):
            m = m.update_modules({"test": "hi"})
        with self.assertRaises(ValueError):
            m = m.update_modules({"list": ["hi"]})

        # Allow updating a strict subset
        m = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        m.update_modules({"layers": [{}, nn.Linear(3, 4)]})
        self.assertEqual(m.layers[1].weight.shape, (4, 3))

        # Using leaf_modules in the update should always work
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stuff = [nn.Linear(2, 2), 0, nn.Linear(2, 2)]
                self.more_stuff = {"hi": nn.Linear(2, 2), "bye": 0}

        m = MyModel()
        m.update_modules(m.leaf_modules())

    def test_parameter_deletion(self):
        m = nn.Linear(32, 32)
        del m.weight
        self.assertFalse(hasattr(m, "weight"))

    def test_circular_leaks(self):
        y = mx.random.uniform(1)
        mx.eval(y)

        def make_and_update():
            model = nn.Linear(1024, 512)
            mx.eval(model.parameters())
            leaves = {}
            model.update_modules(leaves)

        mx.synchronize()
        pre = mx.get_active_memory()
        make_and_update()
        mx.synchronize()
        post = mx.get_active_memory()
        self.assertEqual(pre, post)


class TestLayers(mlx_tests.MLXTestCase):
    def test_identity(self):
        inputs = mx.zeros((10, 4))
        layer = nn.Identity()
        outputs = layer(inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_linear(self):
        inputs = mx.zeros((10, 4))
        layer = nn.Linear(input_dims=4, output_dims=8)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (10, 8))

    def test_bilinear(self):
        inputs1 = mx.zeros((10, 2))
        inputs2 = mx.zeros((10, 4))
        layer = nn.Bilinear(input1_dims=2, input2_dims=4, output_dims=6)
        outputs = layer(inputs1, inputs2)
        self.assertEqual(outputs.shape, (10, 6))

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
                    [-0.0119524, 1.1263, 2.02223],
                    [-0.500331, 0.517899, -1.21143],
                    [1.12958, -0.21413, -2.48738],
                    [1.39955, 0.891329, 1.63289],
                ],
                [
                    [0.241417, -0.619157, -0.77484],
                    [-1.42512, 0.970817, -1.31352],
                    [2.739, -1.2506, 1.56844],
                    [-1.23175, 0.32756, 1.13969],
                ],
            ]
        )
        inorm = nn.InstanceNorm(dims=3)
        y = inorm(x)
        expected_y = [
            [
                [-0.657082, 1.07593, 1.0712],
                [-1.27879, -0.123074, -0.632505],
                [0.796101, -1.56572, -1.30476],
                [1.13978, 0.612862, 0.866067],
            ],
            [
                [0.0964426, -0.557906, -0.759885],
                [-0.904772, 1.30444, -1.20013],
                [1.59693, -1.29752, 1.15521],
                [-0.7886, 0.550987, 0.804807],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))
        # Test InstanceNorm2d
        x = mx.array(
            [
                [
                    [
                        [-0.458824, 0.483254, -0.58611],
                        [-0.447996, -0.176577, -0.622545],
                        [0.0486988, -0.0611224, 1.8845],
                    ],
                    [
                        [1.13049, 0.345315, -0.926389],
                        [0.301795, 0.99207, -0.184927],
                        [-2.23876, -0.758631, -1.12639],
                    ],
                    [
                        [0.0986325, -1.82973, -0.241765],
                        [-1.25257, 0.154442, -0.556204],
                        [-0.329399, -0.319107, 0.830584],
                    ],
                ],
                [
                    [
                        [1.04407, 0.073752, 0.407081],
                        [0.0800776, 1.2513, 1.20627],
                        [0.782321, -0.444367, 0.563132],
                    ],
                    [
                        [0.671423, -1.21689, -1.88979],
                        [-0.110299, -1.42248, 1.17838],
                        [0.159905, 0.516452, -0.539121],
                    ],
                    [
                        [0.810252, 1.50456, 1.08659],
                        [0.182597, 0.0576239, 0.973883],
                        [-0.0621687, 0.184253, 0.784216],
                    ],
                ],
            ]
        )
        inorm = nn.InstanceNorm(dims=3)
        y = inorm(x)
        expected_y = [
            [
                [
                    [-0.120422, 0.801503, -0.463983],
                    [-0.108465, -0.0608611, -0.504602],
                    [0.440008, 0.090032, 2.29032],
                ],
                [
                    [1.63457, 0.621224, -0.843335],
                    [0.719488, 1.4665, -0.0167344],
                    [-2.08591, -0.821575, -1.0663],
                ],
                [
                    [0.495147, -2.22145, -0.0800989],
                    [-0.996913, 0.371763, -0.430643],
                    [0.022495, -0.24714, 1.11538],
                ],
            ],
            [
                [
                    [1.5975, 0.0190292, -0.0123306],
                    [-0.776381, 1.28291, 0.817237],
                    [0.952927, -0.537076, 0.149652],
                ],
                [
                    [0.679836, -1.36624, -2.39651],
                    [-1.24519, -1.5869, 0.788287],
                    [-0.579802, 0.494186, -0.994499],
                ],
                [
                    [1.02171, 1.55474, 0.693008],
                    [-0.523922, 0.00171862, 0.576016],
                    [-1.12667, 0.137632, 0.37914],
                ],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))
        # # Test InstanceNorm3d
        x = mx.array(
            [
                [
                    [
                        [[0.777621, 0.528145, -1.56133], [-2.1722, 0.128192, 0.153862]],
                        [
                            [-1.41317, 0.476288, -1.20411],
                            [0.284446, -0.649858, 0.152112],
                        ],
                    ],
                    [
                        [[0.11, -0.12431, 1.18768], [-0.837743, 1.93502, 0.00236324]],
                        [
                            [-2.40205, -1.25873, -2.04243],
                            [0.336682, -0.261986, 1.54289],
                        ],
                    ],
                    [
                        [
                            [0.789185, -1.63747, 0.67917],
                            [-1.42998, -1.73247, -0.402572],
                        ],
                        [
                            [-0.459489, -2.15559, -0.249959],
                            [0.0298199, 0.10275, -0.821897],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [-2.12354, 0.643973, 0.72391],
                            [0.317797, -0.682916, 0.016364],
                        ],
                        [
                            [-0.146628, -0.987925, 0.573199],
                            [0.0329215, 1.54086, 0.213092],
                        ],
                    ],
                    [
                        [
                            [-1.55784, 0.71179, -0.0678402],
                            [2.41031, -0.290786, 0.00449439],
                        ],
                        [
                            [0.226341, 0.057712, -1.58342],
                            [0.265387, -0.742304, 1.28133],
                        ],
                    ],
                    [
                        [
                            [0.990317, -0.399875, -0.357647],
                            [0.475161, -1.10479, -1.07389],
                        ],
                        [
                            [-1.37804, 1.40097, 0.141618],
                            [-0.501041, 0.0723374, -0.386141],
                        ],
                    ],
                ],
            ]
        )
        inorm = nn.InstanceNorm(dims=3)
        y = inorm(x)
        expected_y = [
            [
                [
                    [[1.23593, 0.821849, -1.30944], [-1.54739, 0.462867, 0.357126]],
                    [[-0.831204, 0.775304, -0.962338], [0.770588, -0.23548, 0.355425]],
                ],
                [
                    [[0.605988, 0.236231, 1.36163], [-0.288258, 2.0846, 0.209922]],
                    [[-1.76427, -0.78198, -1.77689], [0.819875, 0.112659, 1.70677]],
                ],
                [
                    [[1.24684, -1.12192, 0.867539], [-0.847068, -1.20719, -0.183531]],
                    [
                        [0.0686449, -1.58697, -0.0352458],
                        [0.530334, 0.440032, -0.590967],
                    ],
                ],
            ],
            [
                [
                    [[-1.75315, 0.733967, 1.04349], [0.343736, -0.822472, 0.080661]],
                    [[-0.0551618, -1.18025, 0.838402], [0.0990544, 1.78602, 0.348368]],
                ],
                [
                    [[-1.26726, 0.813517, -0.033924], [2.14101, -0.362504, 0.0645089]],
                    [[0.265184, 0.0462839, -2.09632], [0.298721, -0.892134, 1.80203]],
                ],
                [
                    [[0.921369, -0.490465, -0.428293], [0.478897, -1.31732, -1.40296]],
                    [[-1.11283, 1.62192, 0.251107], [-0.35957, 0.0634394, -0.467067]],
                ],
            ],
        ]
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(np.allclose(y, expected_y, atol=1e-5))
        # Test repr
        self.assertTrue(str(inorm) == "InstanceNorm(3, eps=1e-05, affine=False)")

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
        running_mean = batch_norm.running_mean
        running_var = batch_norm.running_var

        data = mx.random.normal((batch_size, num_features))

        normalized_data = batch_norm(data)
        means = mx.mean(data, axis=0)
        variances = mx.var(data, axis=0)
        running_mean = (1 - momentum) * running_mean + momentum * means
        running_var = (1 - momentum) * running_var + momentum * variances
        self.assertTrue(mx.allclose(batch_norm.running_mean, running_mean, atol=1e-5))
        self.assertTrue(mx.allclose(batch_norm.running_var, running_var, atol=1e-5))

        batch_norm = nn.BatchNorm(num_features)

        batch_norm.train()
        running_mean = batch_norm.running_mean
        running_var = batch_norm.running_var
        data = mx.random.normal((batch_size, h, w, num_features))

        normalized_data = batch_norm(data)
        means = mx.mean(data, axis=(0, 1, 2))
        variances = mx.var(data, axis=(0, 1, 2))
        running_mean = (1 - momentum) * running_mean + momentum * means
        running_var = (1 - momentum) * running_var + momentum * variances
        self.assertTrue(mx.allclose(batch_norm.running_mean, running_mean, atol=1e-5))
        self.assertTrue(mx.allclose(batch_norm.running_var, running_var, atol=1e-5))

        self.assertEqual(batch_norm.running_mean.shape, running_mean.shape)
        self.assertEqual(batch_norm.running_var.shape, running_var.shape)

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
        self.assertEqual(y.shape, (N, L - ks + 1, C_out))
        self.assertTrue(mx.allclose(y, mx.full(y.shape, ks * C_in, mx.float32)))

        c = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=ks, stride=2)
        y = c(x)
        self.assertEqual(y.shape, (N, (L - ks + 1) // 2, C_out))
        self.assertTrue("bias" in c.parameters())

        dil = 2
        c = nn.Conv1d(
            in_channels=C_in, out_channels=C_out, kernel_size=ks, dilation=dil
        )
        y = c(x)
        self.assertEqual(y.shape, (N, L - (ks - 1) * dil, C_out))

        c = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=ks, bias=False)
        self.assertTrue("bias" not in c.parameters())

        groups = C_in
        c = nn.Conv1d(
            in_channels=C_in, out_channels=C_out, kernel_size=ks, groups=groups
        )
        y = c(x)
        self.assertEqual(c.weight.shape, (C_out, ks, C_in // groups))
        self.assertEqual(y.shape, (N, L - ks + 1, C_out))

    def test_conv2d(self):
        x = mx.ones((4, 8, 8, 3))
        c = nn.Conv2d(3, 1, 8)
        y = c(x)
        self.assertEqual(y.shape, (4, 1, 1, 1))
        c.weight = mx.ones_like(c.weight) / 8 / 8 / 3
        y = c(x)
        self.assertTrue(np.allclose(y[:, 0, 0, 0], x.mean(axis=(1, 2, 3))))

        # 3x3 conv no padding stride 1
        c = nn.Conv2d(3, 8, 3)
        y = c(x)
        self.assertEqual(y.shape, (4, 6, 6, 8))
        self.assertLess(mx.abs(y - c.weight.sum((1, 2, 3))).max(), 1e-4)

        # 3x3 conv padding 1 stride 1
        c = nn.Conv2d(3, 8, 3, padding=1)
        y = c(x)
        self.assertEqual(y.shape, (4, 8, 8, 8))
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
        self.assertEqual(y.shape, (4, 3, 3, 8))
        self.assertLess(mx.abs(y - c.weight.sum((1, 2, 3))).max(), 1e-4)

        c = nn.Conv2d(3, 8, 3, dilation=2)
        y = c(x)
        self.assertEqual(y.shape, (4, 4, 4, 8))
        self.assertLess(mx.abs(y - c.weight.sum((1, 2, 3))).max(), 1e-4)

        # 3x3 conv groups > 1
        x = mx.ones((4, 7, 7, 4))
        c = nn.Conv2d(4, 8, 3, padding=1, stride=1, groups=2)
        y = c(x)
        self.assertEqual(y.shape, (4, 7, 7, 8))

    def test_sequential(self):
        x = mx.ones((10, 2))
        m = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
        y = m(x)
        self.assertEqual(y.shape, (10, 1))
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
        # From: jax.nn.gelu(np.array(inputs), approximate=True)
        expected_approx = np.array(
            [1.0091482, -0.1693441, 0.22918446, 0.60491, 0.4945476]
        )

        out = nn.GELU()(mx.array(inputs))
        self.assertTrue(np.allclose(out, expected))

        # Test the precise/tanh approximation
        out_approx = nn.GELU(approx="precise")(mx.array(inputs))
        out_approx_tanh = nn.GELU(approx="tanh")(mx.array(inputs))
        self.assertTrue(np.allclose(out_approx, expected_approx))
        self.assertTrue(np.allclose(out_approx_tanh, expected_approx))
        self.assertTrue(np.allclose(out_approx, out_approx_tanh))

        # Crudely check the approximations
        x = mx.arange(-6.0, 6.0, 12 / 100)
        y = nn.gelu(x)
        y_hat1 = nn.gelu_approx(x)
        y_hat2 = nn.gelu_fast_approx(x)
        self.assertLess(mx.abs(y - y_hat1).max(), 0.0005)
        self.assertLess(mx.abs(y - y_hat2).max(), 0.025)

    def test_sin_pe(self):
        m = nn.SinusoidalPositionalEncoding(16, min_freq=0.01)
        x = mx.arange(10)
        y = m(x)

        self.assertEqual(y.shape, (10, 16))
        similarities = y @ y.T
        self.assertLess(
            mx.abs(similarities[mx.arange(10), mx.arange(10)] - 1).max(), 1e-5
        )

    def test_sigmoid(self):
        x = mx.array([1.0, 0.0, -1.0])
        y1 = mx.sigmoid(x)
        y2 = nn.activations.sigmoid(x)
        y3 = nn.Sigmoid()(x)

        self.assertEqualArray(y1, y2, atol=0, rtol=0)
        self.assertEqualArray(y1, y3, atol=0, rtol=0)

    def test_relu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.relu(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0])))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_leaky_relu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.leaky_relu(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, -0.01, 0.0])))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

        y = nn.LeakyReLU(negative_slope=0.1)(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, -0.1, 0.0])))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_elu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.elu(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6321, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

        y = nn.ELU(alpha=1.1)(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6953, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_relu6(self):
        x = mx.array([1.0, -1.0, 0.0, 7.0, -7.0])
        y = nn.relu6(x)
        self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0, 6.0, 0.0])))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(y.dtype, mx.float32)

    def test_softmax(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softmax(x)
        epsilon = 1e-4
        expected_y = mx.array([0.6652, 0.0900, 0.2447])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_softmin(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = nn.softmin(x)
        epsilon = 1e-4
        expected_y = mx.array([0.6652, 0.2447, 0.0900])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_softplus(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softplus(x)
        epsilon = 1e-4
        expected_y = mx.array([1.3133, 0.3133, 0.6931])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_softsign(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softsign(x)
        epsilon = 1e-4
        expected_y = mx.array([0.5, -0.5, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_softshrink(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.softshrink(x)
        epsilon = 1e-4
        expected_y = mx.array([0.5, -0.5, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

        y = nn.Softshrink(lambd=0.7)(x)
        expected_y = mx.array([0.3, -0.3, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_celu(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.celu(x)
        epsilon = 1e-4
        expected_y = mx.array([1.0, -0.6321, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

        y = nn.CELU(alpha=1.1)(x)
        expected_y = mx.array([1.0, -0.6568, 0.0])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_log_softmax(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = nn.log_softmax(x)
        epsilon = 1e-4
        expected_y = mx.array([-2.4076, -1.4076, -0.4076])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, mx.float32)

    def test_log_sigmoid(self):
        x = mx.array([1.0, -1.0, 0.0])
        y = nn.log_sigmoid(x)
        epsilon = 1e-4
        expected_y = mx.array([-0.3133, -1.3133, -0.6931])
        self.assertTrue(mx.all(mx.abs(y - expected_y) < epsilon))
        self.assertEqual(y.shape, (3,))
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
        self.assertEqual(y.shape, (5,))
        self.assertEqual(y.dtype, mx.float32)

    def test_glu(self):
        x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
        y = mx.array([[[0.952574, 1.96403]]], dtype=mx.float32)
        out = nn.glu(x)
        self.assertEqualArray(out, y)

    def test_hard_tanh(self):
        x = mx.array([1.0, -2.0, 0.0, 0.5, 2.0])
        y = nn.hard_tanh(x)
        expected_y = mx.array([1.0, -1.0, 0.0, 0.5, 1.0])
        self.assertTrue(mx.array_equal(y, expected_y))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(y.dtype, mx.float32)

    def test_hard_shrink(self):
        x = mx.array([1.0, -0.5, 0.0, 0.5, -1.5])
        y = nn.hard_shrink(x)
        expected_y = mx.array([1.0, 0.0, 0.0, 0.0, -1.5])
        self.assertTrue(mx.array_equal(y, expected_y))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(y.dtype, mx.float32)

        y = nn.hard_shrink(x, lambd=0.1)
        expected_y = mx.array([1.0, -0.5, 0.0, 0.5, -1.5])
        self.assertTrue(mx.array_equal(y, expected_y))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(y.dtype, mx.float32)

    def test_rope(self):
        for kwargs in [{}, {"traditional": False}, {"base": 10000}, {"scale": 0.25}]:
            rope = nn.RoPE(4, **kwargs)
            shape = (1, 3, 4)
            x = mx.random.uniform(shape=shape)
            y = rope(x)
            self.assertEqual(y.shape, shape)
            self.assertEqual(y.dtype, mx.float32)

            y = rope(x, offset=3)
            self.assertEqual(y.shape, shape)

            y = rope(x.astype(mx.float16))
            self.assertEqual(y.dtype, mx.float16)

    def test_alibi(self):
        alibi = nn.ALiBi()
        shape = (1, 8, 20, 20)
        x = mx.random.uniform(shape=shape)
        y = alibi(x)
        self.assertEqual(y.shape, shape)
        self.assertEqual(y.dtype, mx.float32)

        y = alibi(x.astype(mx.float16))
        self.assertEqual(y.dtype, mx.float16)

    def test_dropout(self):
        x = mx.ones((2, 4))
        y = nn.Dropout(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float32)

        x = mx.ones((2, 4), dtype=mx.bfloat16)
        y = nn.Dropout(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4), dtype=mx.float16)
        y = nn.Dropout(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float16)

    def test_dropout2d(self):
        x = mx.ones((2, 4, 4, 4))
        y = nn.Dropout2d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float32)

        x = mx.ones((2, 4, 4, 4), dtype=mx.bfloat16)
        y = nn.Dropout2d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4, 4, 4), dtype=mx.float16)
        y = nn.Dropout2d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float16)

    def test_dropout3d(self):
        x = mx.ones((2, 4, 4, 4, 4))
        y = nn.Dropout3d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float32)

        x = mx.ones((2, 4, 4, 4, 4), dtype=mx.bfloat16)
        y = nn.Dropout3d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.bfloat16)

        x = mx.ones((2, 4, 4, 4, 4), dtype=mx.float16)
        y = nn.Dropout3d(0.5)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, mx.float16)

    def test_upsample(self):
        b, h, w, c = 1, 2, 2, 1
        scale_factor = 2
        upsample_nearest = nn.Upsample(
            scale_factor=scale_factor, mode="nearest", align_corners=True
        )
        upsample_bilinear = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=True
        )
        upsample_nearest = nn.Upsample(
            scale_factor=scale_factor, mode="nearest", align_corners=True
        )
        upsample_bilinear_no_align_corners = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=False
        )
        upsample_nearest_no_align_corners = nn.Upsample(
            scale_factor=scale_factor, mode="nearest", align_corners=False
        )
        # Test single feature map, align corners
        x = mx.arange(b * h * w * c).reshape((b, c, h, w)).transpose((0, 2, 3, 1))
        expected_nearest = mx.array(
            [[[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]]
        ).transpose((0, 2, 3, 1))
        expected_bilinear = mx.array(
            [
                [
                    [
                        [0, 0.333333, 0.666667, 1],
                        [0.666667, 1, 1.33333, 1.66667],
                        [1.33333, 1.66667, 2, 2.33333],
                        [2, 2.33333, 2.66667, 3],
                    ]
                ]
            ]
        ).transpose((0, 2, 3, 1))
        # Test single feature map, no align corners
        x = (
            mx.arange(1, b * h * w * c + 1)
            .reshape((b, c, h, w))
            .transpose((0, 2, 3, 1))
        )
        expected_bilinear_no_align_corners = mx.array(
            [
                [
                    [
                        [1.0000, 1.2500, 1.7500, 2.0000],
                        [1.5000, 1.7500, 2.2500, 2.5000],
                        [2.5000, 2.7500, 3.2500, 3.5000],
                        [3.0000, 3.2500, 3.7500, 4.0000],
                    ]
                ]
            ]
        ).transpose((0, 2, 3, 1))
        expected_nearest_no_align_corners = mx.array(
            [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]]
        ).transpose((0, 2, 3, 1))
        self.assertTrue(
            np.allclose(
                upsample_nearest_no_align_corners(x), expected_nearest_no_align_corners
            )
        )
        self.assertTrue(
            np.allclose(
                upsample_bilinear_no_align_corners(x),
                expected_bilinear_no_align_corners,
            )
        )

        # Test a more complex batch
        b, h, w, c = 2, 3, 3, 2
        scale_factor = 2
        x = mx.arange((b * h * w * c)).reshape((b, c, h, w)).transpose((0, 2, 3, 1))

        upsample_nearest = nn.Upsample(
            scale_factor=scale_factor, mode="nearest", align_corners=True
        )
        upsample_bilinear = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=True
        )

        expected_nearest = mx.array(
            [
                [
                    [
                        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
                        [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
                        [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                        [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                    ],
                    [
                        [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                        [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                        [12.0, 12.0, 13.0, 13.0, 14.0, 14.0],
                        [12.0, 12.0, 13.0, 13.0, 14.0, 14.0],
                        [15.0, 15.0, 16.0, 16.0, 17.0, 17.0],
                        [15.0, 15.0, 16.0, 16.0, 17.0, 17.0],
                    ],
                ],
                [
                    [
                        [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
                        [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
                        [21.0, 21.0, 22.0, 22.0, 23.0, 23.0],
                        [21.0, 21.0, 22.0, 22.0, 23.0, 23.0],
                        [24.0, 24.0, 25.0, 25.0, 26.0, 26.0],
                        [24.0, 24.0, 25.0, 25.0, 26.0, 26.0],
                    ],
                    [
                        [27.0, 27.0, 28.0, 28.0, 29.0, 29.0],
                        [27.0, 27.0, 28.0, 28.0, 29.0, 29.0],
                        [30.0, 30.0, 31.0, 31.0, 32.0, 32.0],
                        [30.0, 30.0, 31.0, 31.0, 32.0, 32.0],
                        [33.0, 33.0, 34.0, 34.0, 35.0, 35.0],
                        [33.0, 33.0, 34.0, 34.0, 35.0, 35.0],
                    ],
                ],
            ]
        ).transpose((0, 2, 3, 1))
        expected_bilinear = mx.array(
            [
                [
                    [
                        [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
                        [1.2, 1.6, 2.0, 2.4, 2.8, 3.2],
                        [2.4, 2.8, 3.2, 3.6, 4.0, 4.4],
                        [3.6, 4.0, 4.4, 4.8, 5.2, 5.6],
                        [4.8, 5.2, 5.6, 6.0, 6.4, 6.8],
                        [6.0, 6.4, 6.8, 7.2, 7.6, 8.0],
                    ],
                    [
                        [9.0, 9.4, 9.8, 10.2, 10.6, 11.0],
                        [10.2, 10.6, 11.0, 11.4, 11.8, 12.2],
                        [11.4, 11.8, 12.2, 12.6, 13.0, 13.4],
                        [12.6, 13.0, 13.4, 13.8, 14.2, 14.6],
                        [13.8, 14.2, 14.6, 15.0, 15.4, 15.8],
                        [15.0, 15.4, 15.8, 16.2, 16.6, 17.0],
                    ],
                ],
                [
                    [
                        [18.0, 18.4, 18.8, 19.2, 19.6, 20.0],
                        [19.2, 19.6, 20.0, 20.4, 20.8, 21.2],
                        [20.4, 20.8, 21.2, 21.6, 22.0, 22.4],
                        [21.6, 22.0, 22.4, 22.8, 23.2, 23.6],
                        [22.8, 23.2, 23.6, 24.0, 24.4, 24.8],
                        [24.0, 24.4, 24.8, 25.2, 25.6, 26.0],
                    ],
                    [
                        [27.0, 27.4, 27.8, 28.2, 28.6, 29.0],
                        [28.2, 28.6, 29.0, 29.4, 29.8, 30.2],
                        [29.4, 29.8, 30.2, 30.6, 31.0, 31.4],
                        [30.6, 31.0, 31.4, 31.8, 32.2, 32.6],
                        [31.8, 32.2, 32.6, 33.0, 33.4, 33.8],
                        [33.0, 33.4, 33.8, 34.2, 34.6, 35.0],
                    ],
                ],
            ]
        ).transpose((0, 2, 3, 1))
        self.assertTrue(np.allclose(upsample_nearest(x), expected_nearest))
        self.assertTrue(np.allclose(upsample_bilinear(x), expected_bilinear))

        # Test different height and width scale_factor
        b, h, w, c = 1, 2, 2, 2
        x = mx.arange(b * h * w * c).reshape((b, c, h, w)).transpose((0, 2, 3, 1))
        upsample_nearest = nn.Upsample(
            scale_factor=(2, 3), mode="nearest", align_corners=True
        )
        upsample_bilinear = nn.Upsample(
            scale_factor=(2, 3), mode="linear", align_corners=True
        )

        expected_nearest = mx.array(
            [
                [
                    [
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [2, 2, 2, 3, 3, 3],
                        [2, 2, 2, 3, 3, 3],
                    ],
                    [
                        [4, 4, 4, 5, 5, 5],
                        [4, 4, 4, 5, 5, 5],
                        [6, 6, 6, 7, 7, 7],
                        [6, 6, 6, 7, 7, 7],
                    ],
                ]
            ]
        ).transpose((0, 2, 3, 1))
        expected_bilinear = mx.array(
            [
                [
                    [
                        [0, 0.2, 0.4, 0.6, 0.8, 1],
                        [0.666667, 0.866667, 1.06667, 1.26667, 1.46667, 1.66667],
                        [1.33333, 1.53333, 1.73333, 1.93333, 2.13333, 2.33333],
                        [2, 2.2, 2.4, 2.6, 2.8, 3],
                    ],
                    [
                        [4, 4.2, 4.4, 4.6, 4.8, 5],
                        [4.66667, 4.86667, 5.06667, 5.26667, 5.46667, 5.66667],
                        [5.33333, 5.53333, 5.73333, 5.93333, 6.13333, 6.33333],
                        [6, 6.2, 6.4, 6.6, 6.8, 7],
                    ],
                ]
            ]
        ).transpose((0, 2, 3, 1))
        self.assertTrue(np.allclose(upsample_nearest(x), expected_nearest))
        self.assertTrue(np.allclose(upsample_bilinear(x), expected_bilinear))

        # Test repr
        self.assertEqual(
            str(nn.Upsample(scale_factor=2)),
            "Upsample(scale_factor=2.0, mode='nearest', align_corners=False)",
        )
        self.assertEqual(
            str(nn.Upsample(scale_factor=(2, 3))),
            "Upsample(scale_factor=(2.0, 3.0), mode='nearest', align_corners=False)",
        )

    def test_pooling(self):
        # Test 1d pooling
        x = mx.array(
            [
                [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
            ]
        )
        expected_max_pool_output_no_padding_stride_1 = [
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[15.0, 16.0, 17.0], [18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
        ]
        expected_max_pool_output_no_padding_stride_2 = [
            [[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]],
            [[15.0, 16.0, 17.0], [21.0, 22.0, 23.0]],
        ]
        expected_max_pool_output_padding_1_stride_2 = [
            [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[12.0, 13.0, 14.0], [18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
        ]
        expected_max_pool_output_padding_1_stride_2_kernel_3 = [
            [[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]],
            [[15.0, 16.0, 17.0], [21.0, 22.0, 23.0]],
        ]
        expected_avg_pool_output_no_padding_stride_1 = [
            [
                [1.5000, 2.5000, 3.5000],
                [4.5000, 5.5000, 6.5000],
                [7.5000, 8.5000, 9.5000],
            ],
            [
                [13.5000, 14.5000, 15.5000],
                [16.5000, 17.5000, 18.5000],
                [19.5000, 20.5000, 21.5000],
            ],
        ]
        expected_avg_pool_output_no_padding_stride_2 = [
            [[1.5000, 2.5000, 3.5000], [7.5000, 8.5000, 9.5000]],
            [[13.5000, 14.5000, 15.5000], [19.5000, 20.5000, 21.5000]],
        ]
        expected_avg_pool_output_padding_1_stride_2 = [
            [
                [0.0000, 0.5000, 1.0000],
                [4.5000, 5.5000, 6.5000],
                [4.5000, 5.0000, 5.5000],
            ],
            [
                [6.0000, 6.5000, 7.0000],
                [16.5000, 17.5000, 18.5000],
                [10.5000, 11.0000, 11.5000],
            ],
        ]
        expected_avg_pool_output_padding_1_kernel_3 = [
            [[1, 1.66667, 2.33333], [6, 7, 8]],
            [[9, 9.66667, 10.3333], [18, 19, 20]],
        ]
        self.assertTrue(
            np.array_equal(
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0)(x),
                expected_max_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)(x),
                expected_max_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)(x),
                expected_max_pool_output_padding_1_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)(x),
                expected_max_pool_output_padding_1_stride_2_kernel_3,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool1d(kernel_size=2, stride=1, padding=0)(x),
                expected_avg_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool1d(kernel_size=2, stride=2, padding=0)(x),
                expected_avg_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool1d(kernel_size=2, stride=2, padding=1)(x),
                expected_avg_pool_output_padding_1_stride_2,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1)(x),
                expected_avg_pool_output_padding_1_kernel_3,
            )
        )
        # Test 2d pooling
        x = mx.array(
            [
                [
                    [[0, 16], [1, 17], [2, 18], [3, 19]],
                    [[4, 20], [5, 21], [6, 22], [7, 23]],
                    [[8, 24], [9, 25], [10, 26], [11, 27]],
                    [[12, 28], [13, 29], [14, 30], [15, 31]],
                ]
            ]
        )
        expected_max_pool_output_no_padding_stride_1 = [
            [
                [[5, 21], [6, 22], [7, 23]],
                [[9, 25], [10, 26], [11, 27]],
                [[13, 29], [14, 30], [15, 31]],
            ]
        ]
        expected_max_pool_output_no_padding_stride_2 = [
            [[[5, 21], [7, 23]], [[13, 29], [15, 31]]]
        ]
        expected_max_pool_output_padding_1 = [
            [
                [[0, 16], [2, 18], [3, 19]],
                [[8, 24], [10, 26], [11, 27]],
                [[12, 28], [14, 30], [15, 31]],
            ]
        ]
        expected_mean_pool_output_no_padding_stride_1 = [
            [
                [[2.5000, 18.5000], [3.5000, 19.5000], [4.5000, 20.5000]],
                [[6.5000, 22.5000], [7.5000, 23.5000], [8.5000, 24.5000]],
                [[10.5000, 26.5000], [11.5000, 27.5000], [12.5000, 28.5000]],
            ]
        ]
        expected_mean_pool_output_no_padding_stride_2 = [
            [
                [[2.5000, 18.5000], [4.5000, 20.5000]],
                [[10.5000, 26.5000], [12.5000, 28.5000]],
            ]
        ]
        expected_mean_pool_output_padding_1 = [
            [
                [[0.0000, 4.0000], [0.7500, 8.7500], [0.7500, 4.7500]],
                [[3.0000, 11.0000], [7.5000, 23.5000], [4.5000, 12.5000]],
                [[3.0000, 7.0000], [6.7500, 14.7500], [3.7500, 7.7500]],
            ]
        ]
        self.assertTrue(
            np.array_equal(
                nn.MaxPool2d(kernel_size=2, stride=1, padding=0)(x),
                expected_max_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(x),
                expected_max_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(x),
                expected_max_pool_output_padding_1,
            )
        )
        # Average pooling
        self.assertTrue(
            np.allclose(
                nn.AvgPool2d(kernel_size=2, stride=1, padding=0)(x),
                expected_mean_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x),
                expected_mean_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=1)(x),
                expected_mean_pool_output_padding_1,
            )
        )
        # Test multiple batches
        x = mx.array(
            [
                [
                    [[0, 1], [2, 3], [4, 5], [6, 7]],
                    [[8, 9], [10, 11], [12, 13], [14, 15]],
                    [[16, 17], [18, 19], [20, 21], [22, 23]],
                    [[24, 25], [26, 27], [28, 29], [30, 31]],
                ],
                [
                    [[32, 33], [34, 35], [36, 37], [38, 39]],
                    [[40, 41], [42, 43], [44, 45], [46, 47]],
                    [[48, 49], [50, 51], [52, 53], [54, 55]],
                    [[56, 57], [58, 59], [60, 61], [62, 63]],
                ],
            ]
        )
        expected_max_pool_output = [
            [[[10.0, 11.0], [14.0, 15.0]], [[26.0, 27.0], [30.0, 31.0]]],
            [[[42.0, 43.0], [46.0, 47.0]], [[58.0, 59.0], [62.0, 63.0]]],
        ]
        expected_avg_pool_output = [
            [[[2.22222, 2.66667], [5.33333, 6]], [[11.3333, 12], [20, 21]]],
            [[[16.4444, 16.8889], [26.6667, 27.3333]], [[32.6667, 33.3333], [52, 53]]],
        ]
        self.assertTrue(
            np.array_equal(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x),
                expected_max_pool_output,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x),
                expected_avg_pool_output,
            )
        )
        # Test irregular kernel (2, 4), stride (3, 1) and padding (1, 2)
        x = mx.array(
            [
                [
                    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                    [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
                    [[24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35]],
                    [[36, 37, 38], [39, 40, 41], [42, 43, 44], [45, 46, 47]],
                ],
                [
                    [[48, 49, 50], [51, 52, 53], [54, 55, 56], [57, 58, 59]],
                    [[60, 61, 62], [63, 64, 65], [66, 67, 68], [69, 70, 71]],
                    [[72, 73, 74], [75, 76, 77], [78, 79, 80], [81, 82, 83]],
                    [[84, 85, 86], [87, 88, 89], [90, 91, 92], [93, 94, 95]],
                ],
            ]
        )
        expected_irregular_max_pool_output = [
            [
                [
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [9.0, 10.0, 11.0],
                    [9.0, 10.0, 11.0],
                ],
                [
                    [39.0, 40.0, 41.0],
                    [42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0],
                    [45.0, 46.0, 47.0],
                    [45.0, 46.0, 47.0],
                ],
            ],
            [
                [
                    [51.0, 52.0, 53.0],
                    [54.0, 55.0, 56.0],
                    [57.0, 58.0, 59.0],
                    [57.0, 58.0, 59.0],
                    [57.0, 58.0, 59.0],
                ],
                [
                    [87.0, 88.0, 89.0],
                    [90.0, 91.0, 92.0],
                    [93.0, 94.0, 95.0],
                    [93.0, 94.0, 95.0],
                    [93.0, 94.0, 95.0],
                ],
            ],
        ]
        expected_irregular_average_pool_output = [
            [
                [
                    [0.3750, 0.6250, 0.8750],
                    [1.1250, 1.5000, 1.8750],
                    [2.2500, 2.7500, 3.2500],
                    [2.2500, 2.6250, 3.0000],
                    [1.8750, 2.1250, 2.3750],
                ],
                [
                    [15.7500, 16.2500, 16.7500],
                    [24.7500, 25.5000, 26.2500],
                    [34.5000, 35.5000, 36.5000],
                    [27.0000, 27.7500, 28.5000],
                    [18.7500, 19.2500, 19.7500],
                ],
            ],
            [
                [
                    [12.3750, 12.6250, 12.8750],
                    [19.1250, 19.5000, 19.8750],
                    [26.2500, 26.7500, 27.2500],
                    [20.2500, 20.6250, 21.0000],
                    [13.8750, 14.1250, 14.3750],
                ],
                [
                    [39.7500, 40.2500, 40.7500],
                    [60.7500, 61.5000, 62.2500],
                    [82.5000, 83.5000, 84.5000],
                    [63.0000, 63.7500, 64.5000],
                    [42.7500, 43.2500, 43.7500],
                ],
            ],
        ]
        self.assertTrue(
            np.array_equal(
                nn.MaxPool2d(kernel_size=(2, 4), stride=(3, 1), padding=(1, 2))(x),
                expected_irregular_max_pool_output,
            )
        )
        self.assertTrue(
            np.allclose(
                nn.AvgPool2d(kernel_size=(2, 4), stride=(3, 1), padding=(1, 2))(x),
                expected_irregular_average_pool_output,
            )
        )
        # Test repr
        self.assertEqual(
            str(nn.MaxPool1d(kernel_size=3, padding=2)),
            "MaxPool1d(kernel_size=(3,), stride=(3,), padding=(2,))",
        )
        self.assertEqual(
            str(nn.AvgPool1d(kernel_size=2, stride=3)),
            "AvgPool1d(kernel_size=(2,), stride=(3,), padding=(0,))",
        )
        self.assertEqual(
            str(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            "MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))",
        )
        self.assertEqual(
            str(nn.AvgPool2d(kernel_size=(1, 2), stride=2, padding=(1, 2))),
            "AvgPool2d(kernel_size=(1, 2), stride=(2, 2), padding=(1, 2))",
        )
        # Test 3d pooling
        x = mx.array(
            [
                [
                    [
                        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                        [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                        [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                    ],
                    [
                        [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
                        [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
                        [[45, 46, 47], [48, 49, 50], [51, 52, 53]],
                    ],
                ]
            ]
        )
        expected_max_pool_output_no_padding_stride_1 = [
            [[[[39, 40, 41], [42, 43, 44]], [[48, 49, 50], [51, 52, 53]]]]
        ]

        expected_max_pool_output_no_padding_stride_2 = [[[[[39, 40, 41]]]]]
        expected_max_pool_output_padding_1 = [
            [
                [[[0, 1, 2], [6, 7, 8]], [[18, 19, 20], [24, 25, 26]]],
                [[[27, 28, 29], [33, 34, 35]], [[45, 46, 47], [51, 52, 53]]],
            ]
        ]
        expected_irregular_max_pool_output = [
            [
                [[[9, 10, 11], [12, 13, 14], [15, 16, 17]]],
                [[[36, 37, 38], [39, 40, 41], [42, 43, 44]]],
            ]
        ]

        self.assertTrue(
            np.array_equal(
                nn.MaxPool3d(kernel_size=2, stride=1, padding=0)(x),
                expected_max_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x),
                expected_max_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool3d(kernel_size=2, stride=2, padding=1)(x),
                expected_max_pool_output_padding_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.MaxPool3d(kernel_size=(1, 2, 1), stride=(1, 2, 1))(x),
                expected_irregular_max_pool_output,
            )
        )
        self.assertEqual(
            str(nn.MaxPool3d(kernel_size=3, stride=3, padding=2)),
            "MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=(2, 2, 2))",
        )

        expected_avg_pool_output_no_padding_stride_1 = [
            [
                [
                    [[19.5, 20.5, 21.5], [22.5, 23.5, 24.5]],
                    [[28.5, 29.5, 30.5], [31.5, 32.5, 33.5]],
                ]
            ]
        ]

        expected_avg_pool_output_no_padding_stride_2 = [[[[[19.5, 20.5, 21.5]]]]]
        expected_avg_pool_output_padding_1 = [
            [
                [
                    [[0, 0.125, 0.25], [1.125, 1.375, 1.625]],
                    [[3.375, 3.625, 3.875], [9, 9.5, 10]],
                ],
                [
                    [[3.375, 3.5, 3.625], [7.875, 8.125, 8.375]],
                    [[10.125, 10.375, 10.625], [22.5, 23, 23.5]],
                ],
            ]
        ]
        expected_irregular_avg_pool_output = [
            [
                [[[4.5, 5.5, 6.5], [7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]],
                [[[31.5, 32.5, 33.5], [34.5, 35.5, 36.5], [37.5, 38.5, 39.5]]],
            ]
        ]

        self.assertTrue(
            np.array_equal(
                nn.AvgPool3d(kernel_size=2, stride=1, padding=0)(x),
                expected_avg_pool_output_no_padding_stride_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.AvgPool3d(kernel_size=2, stride=2, padding=0)(x),
                expected_avg_pool_output_no_padding_stride_2,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.AvgPool3d(kernel_size=2, stride=2, padding=1)(x),
                expected_avg_pool_output_padding_1,
            )
        )
        self.assertTrue(
            np.array_equal(
                nn.AvgPool3d(kernel_size=(1, 2, 1), stride=(1, 2, 1))(x),
                expected_irregular_avg_pool_output,
            )
        )
        self.assertEqual(
            str(nn.AvgPool3d(kernel_size=3, stride=3, padding=2)),
            "AvgPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=(2, 2, 2))",
        )

    def test_set_dtype(self):
        def assert_dtype(layer, dtype):
            for k, v in tree_flatten(layer.parameters()):
                self.assertEqual(v.dtype, dtype, f"dtype mismatch for {k}")

        layer = nn.Linear(input_dims=4, output_dims=8, bias=True)
        assert_dtype(layer, mx.float32)

        layer.set_dtype(mx.bfloat16)
        assert_dtype(layer, mx.bfloat16)

        layer.set_dtype(mx.float32, lambda x: False)
        assert_dtype(layer, mx.bfloat16)

        layer.set_dtype(mx.int32, lambda x: True)
        assert_dtype(layer, mx.int32)

        layer.set_dtype(mx.int64, predicate=None)
        assert_dtype(layer, mx.int64)

        layer.set_dtype(mx.int16, lambda x: mx.issubdtype(x, mx.integer))
        assert_dtype(layer, mx.int16)

    def test_rnn(self):
        layer = nn.RNN(input_size=5, hidden_size=12, bias=True)
        inp = mx.random.normal((2, 25, 5))

        h_out = layer(inp)
        self.assertEqual(h_out.shape, (2, 25, 12))

        layer = nn.RNN(
            5,
            12,
            bias=False,
            nonlinearity=lambda x: mx.maximum(0, x),
        )

        h_out = layer(inp)
        self.assertEqual(h_out.shape, (2, 25, 12))

        with self.assertRaises(ValueError):
            nn.RNN(5, 12, nonlinearity="tanh")

        inp = mx.random.normal((44, 5))
        h_out = layer(inp)
        self.assertEqual(h_out.shape, (44, 12))

        h_out = layer(inp, hidden=h_out[-1, :])
        self.assertEqual(h_out.shape, (44, 12))

    def test_gru(self):
        layer = nn.GRU(5, 12, bias=True)
        inp = mx.random.normal((2, 25, 5))

        h_out = layer(inp)
        self.assertEqual(h_out.shape, (2, 25, 12))

        h_out = layer(inp, hidden=h_out[:, -1, :])
        self.assertEqual(h_out.shape, (2, 25, 12))

        inp = mx.random.normal((44, 5))
        h_out = layer(inp)
        self.assertEqual(h_out.shape, (44, 12))

        h_out = layer(inp, h_out[-1, :])
        self.assertEqual(h_out.shape, (44, 12))

    def test_lstm(self):
        layer = nn.LSTM(5, 12)
        inp = mx.random.normal((2, 25, 5))

        h_out, c_out = layer(inp)
        self.assertEqual(h_out.shape, (2, 25, 12))
        self.assertEqual(c_out.shape, (2, 25, 12))

        h_out, c_out = layer(inp, hidden=h_out[:, -1, :], cell=c_out[:, -1, :])
        self.assertEqual(h_out.shape, (2, 25, 12))
        self.assertEqual(c_out.shape, (2, 25, 12))

        inp = mx.random.normal((44, 5))
        h_out, c_out = layer(inp)
        self.assertEqual(h_out.shape, (44, 12))
        self.assertEqual(c_out.shape, (44, 12))

        inp = mx.random.normal((44, 5))
        h_out, c_out = layer(inp, hidden=h_out[-1, :], cell=c_out[-1, :])
        self.assertEqual(h_out.shape, (44, 12))
        self.assertEqual(c_out.shape, (44, 12))

    def test_quantized_embedding(self):
        emb = nn.Embedding(32, 256)
        qemb = nn.QuantizedEmbedding.from_embedding(emb, bits=8)
        x = mx.array([2, 6, 9, 3, 0, 3])
        y = emb(x)
        yq = qemb(x)
        self.assertLess((y - yq).abs().max(), qemb.scales.max())

        x = mx.random.uniform(shape=(2, 256))
        y = emb.as_linear(x)
        yq = qemb.as_linear(x)

        def cosine(a, b):
            ab = (a * b).sum(-1)
            aa = mx.linalg.norm(a, axis=-1)
            bb = mx.linalg.norm(b, axis=-1)
            return ab / aa / bb

        self.assertGreater(cosine(y, yq).min(), 0.99)

    def test_causal_mask(self):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(4, mx.float16)
        self.assertFalse(mx.any(mx.isnan(mask)))
        self.assertTrue(mask[0, -1].item() < 0)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(4, mx.bfloat16)
        self.assertFalse(mx.any(mx.isnan(mask)))
        self.assertTrue(mask[0, -1].item() < 0)

    def test_attention(self):
        attn = nn.MultiHeadAttention(32, 4)
        x = mx.random.normal(shape=(2, 5, 32))
        out = attn(x, x, x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
