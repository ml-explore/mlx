import mlx.core as mx
import mlx.distributions.normal as normal
import mlx_tests
from mlx.distributions.independent import Independent
from mlx.distributions.transformed_distribution import TransformedDistribution
from mlx.distributions.transforms import ExpTransform

GOLDEN_VALUES = {
    "normal_log_prob_scalar": -1.7370857156692723,
    "normal_log_prob_batched": [-1.4189385332046727, -1.7370857156692723],
    "normal_entropy_scalar": 2.1120858192443848,
    "exp_transform_forward": [1.0, 2.7182817459106445, 0.1353352814912796],
    "log_normal_log_prob": -1.4180873656830202,
    "independent_normal_joint_log_prob": -3.8725460335371453,
}


class TestDistributions(mlx_tests.MLXTestCase):
    def assert_all_close(self, a, b, atol=1e-6, rtol=1e-5):
        self.assertTrue(
            mx.allclose(a, b, atol=atol, rtol=rtol),
            f"Arrays are not close: \na = {a}\n, b = {b}",
        )

    def test_sample_reproducibility(self):
        key = mx.random.key(42)
        loc, scale = mx.array(10.0), mx.array(0.1)
        sample1 = normal.sample(loc, scale, key=key)
        sample2 = normal.sample(loc, scale, key=key)
        self.assertTrue(mx.all(sample1 == sample2))

        key2 = mx.random.key(43)
        sample3 = normal.sample(loc, scale, key=key2)
        self.assertFalse(mx.all(sample1 == sample3))

    def test_log_prob_golden_scalar(self):
        expected = mx.array(GOLDEN_VALUES["normal_log_prob_scalar"])
        actual = normal.log_prob(
            value=mx.array(1.5), loc=mx.array(0.5), scale=mx.array(2.0)
        )
        self.assert_all_close(actual, expected)

    def test_log_prob_golden_batched(self):
        expected = mx.array(GOLDEN_VALUES["normal_log_prob_batched"])
        actual = normal.log_prob(
            value=mx.array([-1.0, 6.0]),
            loc=mx.array([0.0, 5.0]),
            scale=mx.array([1.0, 2.0]),
        )
        self.assert_all_close(actual, expected)

    def test_entropy_golden_scalar(self):
        expected = mx.array(GOLDEN_VALUES["normal_entropy_scalar"])
        actual = normal.entropy(scale=mx.array(2.0))
        self.assert_all_close(actual, expected)

    def test_mean_and_variance_golden(self):
        self.assert_all_close(normal.mean(mx.array(1.23)), mx.array(1.23))
        self.assert_all_close(normal.variance(mx.array(2.0)), mx.array(4.0))

    def test_gradient_wrt_loc_golden(self):
        loc, scale, value = mx.array(0.5), mx.array(2.0), mx.array(1.5)
        loss_fn = lambda p: normal.log_prob(value, loc=p, scale=scale)
        grad_mlx = mx.grad(loss_fn)(loc)

        # Analytical gradient: (value - loc) / scale**2
        grad_analytical = (value - loc) / scale**2
        self.assert_all_close(grad_mlx, grad_analytical)

    def test_gradient_wrt_scale_golden(self):
        loc, scale, value = mx.array(0.5), mx.array(2.0), mx.array(1.5)
        loss_fn = lambda p: normal.log_prob(value, loc=loc, scale=p)
        grad_mlx = mx.grad(loss_fn)(scale)

        # Analytical gradient: ((value - loc)**2 / scale**3) - (1 / scale)
        grad_analytical = ((value - loc) ** 2 / scale**3) - (1 / scale)
        self.assert_all_close(grad_mlx, grad_analytical)

    def test_normal_symmetry_property_manual(self):
        scale, value = mx.array(1.5), mx.array(0.7)
        log_prob1 = normal.log_prob(value, loc=mx.array(0.0), scale=scale)
        log_prob2 = normal.log_prob(-value, loc=mx.array(0.0), scale=scale)
        self.assert_all_close(log_prob1, log_prob2)

    def test_log_prob_peak_at_mean_property_manual(self):
        loc, scale, delta = mx.array(0.5), mx.array(2.0), 1e-3
        log_prob_at_mean = normal.log_prob(loc, loc=loc, scale=scale)
        log_prob_off_mean = normal.log_prob(loc + delta, loc=loc, scale=scale)
        self.assertGreaterEqual(log_prob_at_mean.item(), log_prob_off_mean.item())

    def test_transformed_distribution_log_prob_golden(self):
        log_normal_factory = TransformedDistribution(normal, ExpTransform())
        expected = mx.array(GOLDEN_VALUES["log_normal_log_prob"])
        actual = log_normal_factory.log_prob(
            value=mx.array(2.0), loc=mx.array(0.5), scale=mx.array(0.8)
        )
        self.assert_all_close(actual, expected)

    def test_independent_log_prob_golden(self):
        independent_normal = Independent(normal, reinterpreted_batch_ndims=1)
        expected = mx.array(GOLDEN_VALUES["independent_normal_joint_log_prob"])
        actual = independent_normal.log_prob(
            value=mx.array([1.0, 2.0]),
            loc=mx.array([0.0, 0.5]),
            scale=mx.array([1.0, 0.8]),
        )
        self.assert_all_close(actual, expected)

    def test_exp_transform_golden(self):
        transform = ExpTransform()
        x = mx.array([0.0, 1.0, -2.0])
        y_expected = mx.array(GOLDEN_VALUES["exp_transform_forward"])
        y_actual = transform.forward(x)
        self.assert_all_close(y_actual, y_expected)

        x_reconstructed = transform.inverse(y_actual)
        self.assert_all_close(x_reconstructed, x)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
