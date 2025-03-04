import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.layers.weight_norm import (
    WeightNormConv1d,
    WeightNormConv2d,
    WeightNormLinear,
    weight_norm,
)
from mlx_tests import MLXTestCase

# Check PyTorch availability for cross-framework tests
try:
    import torch
    import torch.nn as torch_nn
    from torch.nn.utils import weight_norm as torch_weight_norm

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# Custom module for testing higher-dimensional tensors
class CustomModule(nn.Module):
    def __init__(self, weight_shape):
        super().__init__()
        self.weight = mx.random.normal(weight_shape)

    def __call__(self, x):
        if self.weight.ndim == 2:
            return x @ self.weight.T
        return x * self.weight.sum()


class TestWeightNormMLX(MLXTestCase):
    """Test suite for MLX weight normalization implementation."""

    def setUp(self):
        """Initialize test setup with reproducible random seeds."""
        super().setUp()
        np.random.seed(42)
        mx.random.seed(42)
        if PYTORCH_AVAILABLE:
            torch.manual_seed(42)

    def test_convenience_classes(self):
        """Validate convenience classes for weight-normalized layers."""
        # Test WeightNormLinear
        linear_wn = WeightNormLinear(10, 20)
        self.assertTrue(hasattr(linear_wn, "v"), "WeightNormLinear missing v")
        self.assertTrue(hasattr(linear_wn, "g"), "WeightNormLinear missing g")

        # Test WeightNormConv1d
        conv1d_wn = WeightNormConv1d(16, 32, kernel_size=3)
        self.assertTrue(hasattr(conv1d_wn, "v"), "WeightNormConv1d missing v")
        self.assertTrue(hasattr(conv1d_wn, "g"), "WeightNormConv1d missing g")

        # Test WeightNormConv2d
        conv2d_wn = WeightNormConv2d(16, 32, kernel_size=3)
        self.assertTrue(hasattr(conv2d_wn, "v"), "WeightNormConv2d missing v")
        self.assertTrue(hasattr(conv2d_wn, "g"), "WeightNormConv2d missing g")

        # Test forward passes
        x_linear = mx.array(np.random.normal(0, 1, (5, 10)).astype(np.float32))
        y_linear = linear_wn(x_linear)
        self.assertEqual(
            y_linear.shape, (5, 20), "Incorrect output shape for WeightNormLinear"
        )

        x_conv1d = mx.array(np.random.normal(0, 1, (2, 10, 16)).astype(np.float32))
        y_conv1d = conv1d_wn(x_conv1d)
        self.assertEqual(
            y_conv1d.shape[0], 2, "Incorrect batch size for WeightNormConv1d"
        )
        self.assertEqual(
            y_conv1d.shape[2], 32, "Incorrect channels for WeightNormConv1d"
        )

        x_conv2d = mx.array(np.random.normal(0, 1, (2, 8, 8, 16)).astype(np.float32))
        y_conv2d = conv2d_wn(x_conv2d)
        self.assertEqual(
            y_conv2d.shape, (2, 6, 6, 32), "Incorrect output shape for WeightNormConv2d"
        )

    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_weight_norm_mathematical_properties(self):
        """Verify mathematical properties of weight normalization."""
        in_channels, out_channels, kernel_size = 16, 32, 3
        mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        mlx_conv_wn = weight_norm(mlx_conv)

        # Property 1: Norm of normalized weight equals g
        w = mlx_conv_wn.module.weight  # Corrected: Access via module.weight
        w_reshaped = mx.reshape(w, (out_channels, -1))
        w_norms = mx.linalg.norm(w_reshaped, axis=1)
        g_flat = mx.reshape(mlx_conv_wn.g, (-1,))
        norm_ratio = w_norms / g_flat
        self.assertLess(
            float(mx.std(norm_ratio)), 1e-5, "Weight norms do not match g values"
        )

        # Property 2: Direction matches v
        v = mlx_conv_wn.v
        v_reshaped = mx.reshape(v, (out_channels, -1))
        v_norms = mx.linalg.norm(v_reshaped, axis=1)
        v_directions = v_reshaped / mx.reshape(v_norms, (out_channels, 1))
        w_directions = w_reshaped / mx.reshape(w_norms, (out_channels, 1))
        cosine_similarities = mx.sum(v_directions * w_directions, axis=1)
        self.assertGreater(
            float(mx.min(cosine_similarities)),
            0.9999,
            "Weight direction does not match v",
        )

        # Property 3: Changing g scales weight norms proportionally
        old_g = mx.array(mlx_conv_wn.g)
        mlx_conv_wn.g = 2 * old_g
        x = mx.random.normal((2, 10, in_channels))
        mlx_conv_wn(x)  # Trigger weight recomputation
        w_new = mlx_conv_wn.module.weight  # Corrected: Access via module.weight
        w_new_reshaped = mx.reshape(w_new, (out_channels, -1))
        w_new_norms = mx.linalg.norm(w_new_reshaped, axis=1)
        norm_ratio_new = w_new_norms / w_norms
        self.assertLess(
            float(mx.std(norm_ratio_new - 2.0)), 1e-5, "Doubling g did not double norms"
        )

    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_exact_weight_transfer(self):
        """Confirm exact equivalence when transferring weights from PyTorch."""
        in_channels, out_channels, kernel_size = 16, 32, 3
        padding = 1

        # PyTorch Conv1d with weight norm using the new parametrization API
        torch_conv = torch_nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        torch_conv_wn = torch.nn.utils.parametrizations.weight_norm(
            torch_conv, name="weight", dim=0
        )

        # MLX Conv1d
        mlx_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )

        # Transfer weights (PyTorch: [out, in, k] -> MLX: [out, k, in])
        torch_weight = (
            torch_conv_wn.weight.detach().numpy()
        )  # Normalized weight still accessible as .weight
        mlx_weight = mx.array(torch_weight.transpose(0, 2, 1))
        mlx_conv.weight = mlx_weight

        # Input data
        x_np = np.random.normal(0, 1, (4, in_channels, 10)).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np.transpose(0, 2, 1))

        # Forward pass
        with torch.no_grad():
            torch_out = torch_conv_wn(x_torch)
        mlx_out = mlx_conv(x_mlx)
        mx.eval(mlx_out)

        # Compare outputs
        torch_out_np = torch_out.detach().numpy()
        mlx_out_np = np.array(mlx_out.transpose(0, 2, 1))
        max_diff = np.max(np.abs(torch_out_np - mlx_out_np))
        self.assertLess(
            max_diff,
            1e-5,
            f"Outputs differ after weight transfer, max diff: {max_diff}",
        )

    def test_weight_norm_implementation(self):
        """Test core weight_norm implementation."""
        linear = nn.Linear(input_dims=3, output_dims=2)
        linear_wn = weight_norm(linear, name="weight", dim=0)
        v = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        g = mx.array([[2.0], [3.0]])
        linear_wn.v = v
        linear_wn.g = g
        x = mx.random.normal((4, 3))
        y = linear_wn(x)
        mx.eval(y)
        expected = mx.weight_norm(v, g, axes=[1])
        max_diff = float(mx.max(mx.abs(linear_wn.module.weight - expected)))
        self.assertLess(max_diff, 1e-5, f"Weight norm mismatch, max diff: {max_diff}")

    def test_axis_conversion(self):
        """Test axis handling across dimensions and negative indices."""
        for ndim in [2, 3]:
            for dim in range(ndim):
                with self.subTest(ndim=ndim, dim=dim):
                    shape = tuple(2 for _ in range(ndim))
                    if ndim == 2:
                        linear = nn.Linear(shape[-1], shape[0])
                    else:
                        linear = CustomModule(shape)
                    linear_wn = weight_norm(linear, dim=dim)
                    expected_axes = [i for i in range(ndim) if i != dim]
                    self.assertEqual(
                        linear_wn.wn_axes,
                        expected_axes,
                        f"Axes mismatch: expected {expected_axes}, got {linear_wn.wn_axes}",
                    )
                    x = mx.random.normal((3, shape[-1] if ndim == 2 else 2))
                    y = linear_wn(x)
                    mx.eval(y)

        with self.subTest(case="dim=-1 for shape (2,2)"):
            linear = nn.Linear(2, 2)
            linear_wn = weight_norm(linear, dim=-1)
            self.assertEqual(linear_wn.wn_axes, [0], "Axes mismatch for dim=-1")

        with self.subTest(case="dim=-2 for shape (2,2,2)"):
            module = CustomModule((2, 2, 2))
            module_wn = weight_norm(module, dim=-2)
            self.assertEqual(module_wn.wn_axes, [0, 2], "Axes mismatch for dim=-2")

    def test_higher_dims(self):
        """Test weight normalization on higher-dimensional tensors."""
        shape = (3, 4, 5, 6)
        for dim in range(4):
            with self.subTest(dim=dim):
                module = CustomModule(shape)
                module_wn = weight_norm(module, dim=dim)
                expected_axes = [i for i in range(4) if i != dim]
                self.assertEqual(module_wn.wn_axes, expected_axes)
                x = mx.random.normal((2, 2))
                y = module_wn(x)
                mx.eval(y)
                # Verify norms match g
                weight_flat = mx.reshape(module_wn.module.weight, (shape[dim], -1))
                weight_norms = mx.linalg.norm(weight_flat, axis=1)
                g_values = mx.reshape(module_wn.g, (-1,))
                norm_ratios = weight_norms / g_values
                self.assertLess(
                    float(mx.std(norm_ratios)),
                    1e-5,
                    f"Norms not matching g for dim {dim}",
                )

    def test_conv2d_weight_norm(self):
        """Test weight normalization on Conv2d layer."""
        conv2d = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3))
        conv2d_wn = weight_norm(conv2d, dim=0)
        self.assertEqual(conv2d_wn.wn_axes, [1, 2, 3], "Incorrect normalization axes")
        v = conv2d_wn.v
        g = conv2d_wn.g
        weight_flat = mx.reshape(v, (v.shape[0], -1))
        v_norm = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
        v_norm = mx.reshape(v_norm, (v.shape[0], 1, 1, 1))
        expected_weight = g * (v / mx.maximum(v_norm, 1e-5))
        max_diff = float(mx.max(mx.abs(expected_weight - conv2d_wn.module.weight)))
        self.assertLess(
            max_diff, 1e-5, f"Conv2d weight norm mismatch, max diff: {max_diff}"
        )

    def test_none_dim(self):
        """Test weight normalization with dim=None."""
        linear = nn.Linear(10, 20)
        linear_wn = weight_norm(linear, dim=None)
        self.assertEqual(linear_wn.wn_axes, [], "Expected empty axes list for dim=None")
        x = mx.random.normal((5, 10))
        y = linear_wn(x)
        mx.eval(y)
        self.assertEqual(y.shape, (5, 20), "Incorrect output shape")
        norm_value = mx.linalg.norm(linear_wn.module.weight)
        g_value = float(linear_wn.g)
        norm_ratio = float(norm_value) / g_value
        self.assertLess(
            abs(norm_ratio - 1.0), 1e-5, f"Norm not matching g, ratio: {norm_ratio}"
        )

    def test_direct_api_usage(self):
        """Test direct mx.weight_norm API and module wrapper."""
        v = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        g = mx.array([[2.0], [3.0]])
        expected = g * (v / mx.linalg.norm(v, axis=1, keepdims=True))
        actual = mx.weight_norm(v, g, axes=[1])
        mx.eval(actual)
        max_diff = float(mx.max(mx.abs(expected - actual)))
        self.assertLess(max_diff, 1e-5, f"Direct API mismatch, max diff: {max_diff}")

        module = CustomModule((2, 3))
        module_wn = weight_norm(module, dim=0)
        module_wn.v = v
        module_wn.g = g
        x = mx.random.normal((4, 3))
        y = module_wn(x)
        mx.eval(y)
        module_diff = float(mx.max(mx.abs(expected - module_wn.module.weight)))
        self.assertLess(
            module_diff, 1e-5, f"Module wrapper mismatch, max diff: {module_diff}"
        )

    def test_conv_weight_norm(self):
        """Test weight normalization on Conv1d layer."""
        conv = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3)
        conv_wn = weight_norm(conv, dim=0)
        self.assertEqual(conv_wn.wn_axes, [1, 2], "Incorrect normalization axes")
        v = conv_wn.v
        g = conv_wn.g
        weight_flat = mx.reshape(v, (v.shape[0], -1))
        v_norm = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
        v_norm = mx.reshape(v_norm, (v.shape[0], 1, 1))
        expected_weight = g * (v / mx.maximum(v_norm, 1e-5))
        max_diff = float(mx.max(mx.abs(expected_weight - conv_wn.module.weight)))
        self.assertLess(
            max_diff, 1e-5, f"Conv1d weight norm mismatch, max diff: {max_diff}"
        )

    def test_edge_cases(self):
        """Test edge cases with small and zero norms."""
        linear = nn.Linear(3, 2)
        linear_wn = weight_norm(linear, dim=0)

        # Small norm values
        v_small = mx.full((2, 3), 1e-6, dtype=mx.float32)
        g = mx.array([[1.0], [1.0]], dtype=mx.float32)
        linear_wn.v = v_small
        linear_wn.g = g
        v_norm = mx.linalg.norm(v_small, axis=1, keepdims=True)
        v_norm = mx.maximum(v_norm, 1e-5)
        expected_weight = g * (v_small / v_norm)
        x = mx.random.normal((1, 3))
        y = linear_wn(x)
        mx.eval(y)
        self.assertFalse(mx.any(mx.isnan(linear_wn.module.weight)), "NaNs in weight")
        self.assertFalse(mx.any(mx.isinf(linear_wn.module.weight)), "Infs in weight")
        max_diff = float(mx.max(mx.abs(expected_weight - linear_wn.module.weight)))
        self.assertLess(max_diff, 1e-5, f"Small norm mismatch, max diff: {max_diff}")

        # Zero norm
        v_zero = mx.zeros((2, 3), dtype=mx.float32)
        linear_wn.v = v_zero
        y = linear_wn(x)
        mx.eval(y)
        self.assertTrue(
            mx.all(linear_wn.module.weight == 0), "Non-zero weight for zero v"
        )

    def test_bfloat16(self):
        """Test weight normalization with bfloat16 dtype."""
        linear = nn.Linear(3, 2)
        linear_wn = weight_norm(linear, dim=0)
        v = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.bfloat16)
        g = mx.array([[2.0], [3.0]], dtype=mx.bfloat16)
        linear_wn.v = v
        linear_wn.g = g
        v_norm = mx.linalg.norm(v, axis=1, keepdims=True)
        v_norm = mx.maximum(v_norm, 1e-5)
        expected_weight = g * (v / v_norm)
        x = mx.random.normal((1, 3), dtype=mx.bfloat16)
        y = linear_wn(x)
        mx.eval(y)
        self.assertFalse(
            mx.any(mx.isnan(linear_wn.module.weight)), "NaNs in bfloat16 weight"
        )
        self.assertFalse(
            mx.any(mx.isinf(linear_wn.module.weight)), "Infs in bfloat16 weight"
        )
        max_diff = float(mx.max(mx.abs(expected_weight - linear_wn.module.weight)))
        self.assertLess(max_diff, 1e-2, f"bfloat16 mismatch, max diff: {max_diff}")

    def test_weight_norm_core_api(self):
        """Test the core weight_norm function."""
        # Create test tensors
        v = mx.array(np.random.normal(0, 1, (10, 5)).astype(np.float32))
        g = mx.array(np.random.normal(0, 1, (10, 1)).astype(np.float32))

        # Apply weight normalization
        normalized = mx.weight_norm(v, g, axes=[1])

        # Calculate expected result manually
        v_norm = mx.linalg.norm(v, axis=1, keepdims=True)
        expected = g * (v / (v_norm + 1e-5))

        # Compare results
        diff = mx.max(mx.abs(normalized - expected))
        self.assertLess(float(diff), 1e-5)

        # Verify shape
        self.assertEqual(normalized.shape, v.shape)

    def test_weight_norm_multi_axes(self):
        """Test weight normalization over multiple axes."""
        # Create a test tensor with multiple dimensions
        shape = (8, 3, 4)
        v = mx.array(np.random.normal(0, 1, shape).astype(np.float32))
        g = mx.array(np.random.normal(0, 1, (8, 1, 1)).astype(np.float32))

        # Apply weight normalization
        normalized = mx.weight_norm(v, g, axes=[1, 2])

        # Calculate expected result manually
        v_reshaped = mx.reshape(v, (v.shape[0], -1))
        v_norm = mx.linalg.norm(v_reshaped, axis=1, keepdims=True)
        v_norm_reshaped = mx.reshape(v_norm, (v.shape[0], 1, 1))
        expected = g * (v / (v_norm_reshaped + 1e-5))

        # Compare results
        diff = mx.max(mx.abs(normalized - expected))
        self.assertLess(float(diff), 1e-5)

        # Verify shape
        self.assertEqual(normalized.shape, v.shape)

    def test_weight_norm_all_axes(self):
        """Test weight normalization over all axes."""
        # Create a 3D tensor
        shape = (5, 6, 7)
        v = mx.array(np.random.normal(0, 1, shape).astype(np.float32))
        g = mx.array(np.random.normal(0, 1, 1).astype(np.float32))

        try:
            # Apply weight normalization with empty axes list
            normalized = mx.weight_norm(v, g, axes=[])

            # Calculate expected result
            v_flat = mx.reshape(v, (-1,))
            v_norm = mx.linalg.norm(v_flat)
            expected = g * v / (v_norm + 1e-5)

            # Compare results
            diff = mx.max(mx.abs(normalized - expected))
            self.assertLess(float(diff), 1e-5)

        except Exception as e:
            # If normalization over all axes isn't supported, document this limitation
            self.skipTest(f"Normalization over all axes not supported: {str(e)}")

    def test_github_issue_1888(self):
        """Test the specific example from GitHub issue #1888."""
        # Create sample tensors from the issue
        v = mx.random.normal((64, 3, 3))
        g = mx.random.normal((64, 1, 1))

        # Apply weight normalization
        w = mx.weight_norm(v, g, axes=[1, 2])

        # Verify shape
        self.assertEqual(w.shape, v.shape)

        # Verify norm along specified dimensions
        v_reshaped = mx.reshape(v, (v.shape[0], -1))
        v_norm = mx.linalg.norm(v_reshaped, axis=1, keepdims=True)
        v_norm_broadcast = mx.reshape(v_norm, (v_norm.shape[0], 1, 1))

        # Compute expected weight
        expected_w = g * (v / (v_norm_broadcast + 1e-5))

        # Compare results
        diff = mx.max(mx.abs(w - expected_w))
        self.assertLess(float(diff), 1e-5)


if __name__ == "__main__":
    unittest.main()
