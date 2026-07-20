# Copyright © 2023 Apple Inc.

import io
import math
import unittest

import mlx.core as mx
import mlx_tests


class TestRandom(mlx_tests.MLXTestCase):
    def test_global_rng(self):
        mx.random.seed(3)
        a = mx.random.uniform()
        b = mx.random.uniform()

        mx.random.seed(3)
        x = mx.random.uniform()
        y = mx.random.uniform()

        self.assertEqual(a.item(), x.item())
        self.assertEqual(y.item(), b.item())

    def test_key(self):
        k1 = mx.random.key(0)
        k2 = mx.random.key(0)
        self.assertTrue(mx.array_equal(k1, k2))

        k2 = mx.random.key(1)
        self.assertFalse(mx.array_equal(k1, k2))

    def test_key_split(self):
        key = mx.random.key(0)

        k1, k2 = mx.random.split(key)
        self.assertFalse(mx.array_equal(k1, k2))

        r1, r2 = mx.random.split(key)
        self.assertTrue(mx.array_equal(k1, r1))
        self.assertTrue(mx.array_equal(k2, r2))

        keys = mx.random.split(key, 10)
        self.assertEqual(keys.shape, (10, 2))

    def test_uniform(self):
        key = mx.random.key(0)
        a = mx.random.uniform(key=key)
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, mx.float32)

        b = mx.random.uniform(key=key)
        self.assertEqual(a.item(), b.item())

        a = mx.random.uniform(shape=(2, 3))
        self.assertEqual(a.shape, (2, 3))

        a = mx.random.uniform(shape=(1000,), low=-1, high=5)
        self.assertTrue(mx.all((a > -1) < 5).item())

        a = mx.random.uniform(shape=(1000,), low=mx.array(-1), high=5)
        self.assertTrue(mx.all((a > -1) < 5).item())

        a = mx.random.uniform(low=-0.1, high=0.1, shape=(1,), dtype=mx.bfloat16)
        self.assertEqual(a.dtype, mx.bfloat16)

        self.assertEqual(mx.random.uniform().dtype, mx.random.uniform(dtype=None).dtype)

    def test_normal_and_laplace(self):
        # Same tests for normal and laplace.
        for distribution_sampler in [mx.random.normal, mx.random.laplace]:
            key = mx.random.key(0)
            a = distribution_sampler(key=key)
            self.assertEqual(a.shape, ())
            self.assertEqual(a.dtype, mx.float32)

            b = distribution_sampler(key=key)
            self.assertEqual(a.item(), b.item())

            a = distribution_sampler(shape=(2, 3))
            self.assertEqual(a.shape, (2, 3))

            ## Generate in float16 or bfloat16
            for t in [mx.float16, mx.bfloat16]:
                a = distribution_sampler(dtype=t)
                self.assertEqual(a.dtype, t)

            # Generate with a given mean and standard deviation
            loc = 1.0
            scale = 2.0

            a = distribution_sampler(shape=(3, 2), loc=loc, scale=scale, key=key)
            b = scale * distribution_sampler(shape=(3, 2), key=key) + loc
            self.assertTrue(mx.allclose(a, b))

            a = distribution_sampler(
                shape=(3, 2), loc=loc, scale=scale, dtype=mx.float16, key=key
            )
            b = (
                scale * distribution_sampler(shape=(3, 2), dtype=mx.float16, key=key)
                + loc
            )
            self.assertTrue(mx.allclose(a, b))

            self.assertEqual(
                distribution_sampler().dtype, distribution_sampler(dtype=None).dtype
            )

            # Test not getting -inf or inf with half precison
            for hp in [mx.float16, mx.bfloat16]:
                a = abs(distribution_sampler(shape=(10000,), loc=0, scale=1, dtype=hp))
                self.assertTrue(mx.all(a < mx.inf))

    def test_multivariate_normal(self):
        key = mx.random.key(0)
        mean = mx.array([0, 0])
        cov = mx.array([[1, 0], [0, 1]])

        a = mx.random.multivariate_normal(mean, cov, key=key, stream=mx.cpu)
        self.assertEqual(a.shape, (2,))

        ## Check dtypes
        for t in [mx.float32]:
            a = mx.random.multivariate_normal(
                mean, cov, dtype=t, key=key, stream=mx.cpu
            )
            self.assertEqual(a.dtype, t)
        for t in [
            mx.int8,
            mx.int32,
            mx.int64,
            mx.uint8,
            mx.uint32,
            mx.uint64,
            mx.float16,
            mx.bfloat16,
        ]:
            with self.assertRaises(ValueError):
                mx.random.multivariate_normal(
                    mean, cov, dtype=t, key=key, stream=mx.cpu
                )

        ## Check incompatible shapes
        with self.assertRaises(ValueError):
            mean = mx.zeros((2, 2))
            cov = mx.zeros((2, 2))
            mx.random.multivariate_normal(mean, cov, shape=(3,), key=key, stream=mx.cpu)

        with self.assertRaises(ValueError):
            mean = mx.zeros((2))
            cov = mx.zeros((2, 2, 2))
            mx.random.multivariate_normal(mean, cov, shape=(3,), key=key, stream=mx.cpu)

        with self.assertRaises(ValueError):
            mean = mx.zeros((3,))
            cov = mx.zeros((2, 2))
            mx.random.multivariate_normal(mean, cov, key=key, stream=mx.cpu)

        with self.assertRaises(ValueError):
            mean = mx.zeros((2,))
            cov = mx.zeros((2, 3))
            mx.random.multivariate_normal(mean, cov, key=key, stream=mx.cpu)

        ## Different shape of mean and cov
        mean = mx.array([[0, 7], [1, 2], [3, 4]])
        cov = mx.array([[1, 0.5], [0.5, 1]])
        a = mx.random.multivariate_normal(mean, cov, shape=(4, 3), stream=mx.cpu)
        self.assertEqual(a.shape, (4, 3, 2))

        ## Check correcteness of the mean and covariance
        n_test = int(1e5)

        def check_jointly_gaussian(data, mean, cov):
            empirical_mean = mx.mean(data, axis=0)
            empirical_cov = (
                (data - empirical_mean).T @ (data - empirical_mean) / data.shape[0]
            )
            N = data.shape[1]
            self.assertTrue(
                mx.allclose(
                    empirical_mean, mean, rtol=0.0, atol=10 * N**2 / math.sqrt(n_test)
                )
            )
            self.assertTrue(
                mx.allclose(
                    empirical_cov, cov, rtol=0.0, atol=10 * N**2 / math.sqrt(n_test)
                )
            )

        mean = mx.array([4.0, 7.0])
        cov = mx.array([[2, 0.5], [0.5, 1]])
        data = mx.random.multivariate_normal(
            mean, cov, shape=(n_test,), key=key, stream=mx.cpu
        )
        check_jointly_gaussian(data, mean, cov)

        mean = mx.arange(3)
        cov = mx.array([[1, -1, 0.5], [-1, 1, -0.5], [0.5, -0.5, 1]])
        data = mx.random.multivariate_normal(
            mean, cov, shape=(n_test,), key=key, stream=mx.cpu
        )
        check_jointly_gaussian(data, mean, cov)

    def test_randint(self):
        a = mx.random.randint(0, 1, [])
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, mx.int32)

        shape = (88,)
        low = mx.array(3)
        high = mx.array(15)

        key = mx.random.key(0)
        a = mx.random.randint(low, high, shape, key=key)
        self.assertEqual(a.shape, shape)
        self.assertEqual(a.dtype, mx.int32)

        # Check using the same key yields the same value
        b = mx.random.randint(low, high, shape, key=key)
        self.assertListEqual(a.tolist(), b.tolist())

        shape = (3, 4)
        low = mx.reshape(mx.array([0] * 3), [3, 1])
        high = mx.reshape(mx.array([12, 13, 14, 15]), [1, 4])

        a = mx.random.randint(low, high, shape)
        self.assertEqual(a.shape, shape)

        a = mx.random.randint(-10, 10, [1000, 1000])
        self.assertTrue(mx.all(-10 <= a).item() and mx.all(a < 10).item())

        a = mx.random.randint(10, -10, [1000, 1000])
        self.assertTrue(mx.all(a == 10).item())

        self.assertEqual(
            mx.random.randint(0, 1).dtype, mx.random.randint(0, 1, dtype=None).dtype
        )

    def test_bernoulli(self):
        a = mx.random.bernoulli()
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, mx.bool_)

        a = mx.random.bernoulli(mx.array(0.5), [5])
        self.assertEqual(a.shape, (5,))

        a = mx.random.bernoulli(mx.array([2.0, -2.0]))
        self.assertEqual(a.tolist(), [True, False])
        self.assertEqual(a.shape, (2,))

        p = mx.array([0.1, 0.2, 0.3])
        mx.reshape(p, [1, 3])
        x = mx.random.bernoulli(p, [4, 3])
        self.assertEqual(x.shape, (4, 3))

        with self.assertRaises(ValueError):
            mx.random.bernoulli(p, [2])  # Bad shape

        with self.assertRaises(ValueError):
            mx.random.bernoulli(0, [2])  # Bad type

    def test_truncated_normal(self):
        a = mx.random.truncated_normal(-2.0, 2.0)
        self.assertEqual(a.size, 1)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.random.truncated_normal(mx.array([]), mx.array([]))
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.size, 0)

        lower = mx.reshape(mx.array([-2.0, 0.0]), [1, 2])
        upper = mx.reshape(mx.array([0.0, 1.0, 2.0]), [3, 1])
        a = mx.random.truncated_normal(lower, upper)

        self.assertEqual(a.shape, (3, 2))
        self.assertTrue(mx.all(lower <= a).item() and mx.all(a <= upper).item())

        a = mx.random.truncated_normal(2.0, -2.0)
        self.assertTrue(mx.all(a == 2.0).item())

        a = mx.random.truncated_normal(-3.0, 3.0, [542, 399])
        self.assertEqual(a.shape, (542, 399))

        lower = mx.array([-2.0, -1.0])
        higher = mx.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            mx.random.truncated_normal(lower, higher)  # Bad shape

        self.assertEqual(
            mx.random.truncated_normal(0, 1).dtype,
            mx.random.truncated_normal(0, 1, dtype=None).dtype,
        )

    def test_gumbel(self):
        samples = mx.random.gumbel(shape=(100, 100))
        self.assertEqual(samples.shape, (100, 100))
        self.assertEqual(samples.dtype, mx.float32)
        mean = 0.5772
        # Std deviation of the sample mean is small (<0.02),
        # so this test is pretty conservative
        self.assertTrue(mx.abs(mx.mean(samples) - mean) < 0.2)

        self.assertEqual(
            mx.random.gumbel((1, 1)).dtype, mx.random.gumbel((1, 1), dtype=None).dtype
        )

    def test_categorical(self):
        logits = mx.zeros((10, 20))
        self.assertEqual(mx.random.categorical(logits, -1).shape, (10,))
        self.assertEqual(mx.random.categorical(logits, 0).shape, (20,))
        self.assertEqual(mx.random.categorical(logits, 1).shape, (10,))

        out = mx.random.categorical(logits)
        self.assertEqual(out.shape, (10,))
        self.assertEqual(out.dtype, mx.uint32)
        self.assertTrue(mx.max(out).item() < 20)

        out = mx.random.categorical(logits, 0, [5, 20])
        self.assertEqual(out.shape, (5, 20))
        self.assertTrue(mx.max(out).item() < 10)

        out = mx.random.categorical(logits, 1, num_samples=7)
        self.assertEqual(out.shape, (10, 7))
        out = mx.random.categorical(logits, 0, num_samples=7)
        self.assertEqual(out.shape, (20, 7))

        with self.assertRaises(ValueError):
            mx.random.categorical(logits, shape=[10, 5], num_samples=5)

    def test_categorical_search_cpu_and_transforms(self):
        cdf_values = [
            [[10, 30, 100], [20, 20, 80]],
            [[5, 25, 50], [1, 49, 100]],
        ]
        bits_values = [
            [[0, 1 << 31, (1 << 32) - 1, 7], [0, 1 << 30, 1 << 31, 99]],
            [[0, 1 << 30, 1 << 31, (1 << 32) - 1], [3, 17, 31, 127]],
        ]
        cdf = mx.array(cdf_values, dtype=mx.uint64)
        random_bits = mx.array(bits_values, dtype=mx.uint32)

        def reference_row(row_cdf, row_bits):
            total = row_cdf[-1]
            total_high = total >> 32
            total_low = total & 0xFFFFFFFF
            output = []
            for word in row_bits:
                target = word * total_high
                target += (word * total_low) >> 32
                index = 0
                while index < len(row_cdf) and target >= row_cdf[index]:
                    index += 1
                output.append(min(index, len(row_cdf) - 1))
            return output

        expected = [
            [
                reference_row(row_cdf, row_bits)
                for row_cdf, row_bits in zip(cdf_batch, bits_batch)
            ]
            for cdf_batch, bits_batch in zip(cdf_values, bits_values)
        ]

        def search(one_cdf, one_bits):
            return mx.random._categorical_search(one_cdf, one_bits, stream=mx.cpu)

        direct = search(cdf, random_bits)
        mx.eval(direct)
        self.assertEqual(direct.tolist(), expected)
        self.assertEqual(direct.dtype, mx.uint32)

        both_mapped = mx.vmap(search)(cdf, random_bits)
        cdf_only = mx.vmap(search, in_axes=(0, None))(cdf, random_bits[0])
        bits_only = mx.vmap(search, in_axes=(None, 0))(cdf[0], random_bits)
        mx.eval(both_mapped, cdf_only, bits_only)
        self.assertTrue(mx.array_equal(both_mapped, direct))
        self.assertEqual(
            cdf_only.tolist(),
            [search(cdf[i], random_bits[0]).tolist() for i in range(2)],
        )
        self.assertEqual(
            bits_only.tolist(),
            [search(cdf[0], random_bits[i]).tolist() for i in range(2)],
        )

        compiled = mx.compile(search)(cdf, random_bits)
        compiled_vmap = mx.compile(mx.vmap(search))(cdf, random_bits)
        mx.eval(compiled, compiled_vmap)
        self.assertTrue(mx.array_equal(compiled, direct))
        self.assertTrue(mx.array_equal(compiled_vmap, direct))

        if mx.metal.is_available():
            metal = mx.random._categorical_search(cdf, random_bits, stream=mx.gpu)
            metal_vmap = mx.vmap(
                lambda one_cdf, one_bits: mx.random._categorical_search(
                    one_cdf, one_bits, stream=mx.gpu
                )
            )(cdf, random_bits)
            mx.eval(metal, metal_vmap)
            self.assertTrue(mx.array_equal(metal, direct))
            self.assertTrue(mx.array_equal(metal_vmap, direct))

        one_dimensional = search(cdf[0, 0], random_bits[0, 0])
        mx.eval(one_dimensional)
        self.assertEqual(one_dimensional.tolist(), expected[0][0])

        with self.assertRaises(ValueError):
            search(cdf.astype(mx.uint32), random_bits)
        with self.assertRaises(ValueError):
            search(cdf, random_bits.astype(mx.uint64))
        with self.assertRaises(ValueError):
            search(cdf, random_bits[:, 0])

    def test_categorical_fixed_source_candidate(self):
        key = mx.random.key(17)
        logits = mx.array([[-3.0, -1.0, 0.0, 2.0], [1.0, -2.0, 0.5, -0.5]])

        def candidate(one_logits, one_key, stream=mx.cpu):
            return mx.random._categorical_fixed(
                one_logits,
                num_samples=257,
                axis=-1,
                key=one_key,
                stream=stream,
            )

        cpu = candidate(logits, key)
        repeated = candidate(logits, key)
        shifted = candidate(logits + mx.array([[7.0], [-9.0]]), key)
        mx.eval(cpu, repeated, shifted)
        self.assertEqual(cpu.shape, (2, 257))
        self.assertEqual(cpu.dtype, mx.uint32)
        self.assertTrue(mx.array_equal(cpu, repeated))
        self.assertTrue(mx.array_equal(cpu, shifted))
        self.assertTrue(mx.all(cpu < 4).item())

        if mx.metal.is_available():
            gpu = candidate(logits, key, mx.gpu)
            mx.eval(gpu)
            self.assertTrue(mx.array_equal(gpu, cpu))

        for dtype in [mx.float16, mx.bfloat16, mx.int32, mx.bool_]:
            typed = logits.astype(dtype)
            actual = candidate(typed, key)
            expected = candidate(typed.astype(mx.float32), key)
            mx.eval(actual, expected)
            self.assertTrue(mx.array_equal(actual, expected))

        axis_logits = mx.arange(24).reshape(2, 3, 4) / 8
        axis_zero = mx.random._categorical_fixed(
            axis_logits,
            num_samples=5,
            axis=0,
            key=key,
            stream=mx.cpu,
        )
        axis_one = mx.random._categorical_fixed(
            axis_logits,
            num_samples=5,
            axis=1,
            key=key,
            stream=mx.cpu,
        )
        axis_last = mx.random._categorical_fixed(
            axis_logits,
            num_samples=5,
            axis=-1,
            key=key,
            stream=mx.cpu,
        )
        self.assertEqual(axis_zero.shape, (3, 4, 5))
        self.assertEqual(axis_one.shape, (2, 4, 5))
        self.assertEqual(axis_last.shape, (2, 3, 5))

        masked = mx.random._categorical_fixed(
            mx.array([[-math.inf, 0.0, -math.inf]]),
            num_samples=100,
            key=key,
            stream=mx.cpu,
        )
        mx.eval(masked)
        self.assertTrue(mx.all(masked == 1).item())

        nonfinite_cases = [
            ([0.0, math.inf, -1.0], 1),
            ([math.inf, 0.0, math.inf], 0),
            ([-math.inf, -math.inf, -math.inf], 0),
            ([math.nan, math.nan, math.nan], 0),
            ([-math.inf, math.nan, -math.inf], 0),
            ([math.nan, math.inf, 0.0], 1),
        ]
        for values, expected_index in nonfinite_cases:
            special_logits = mx.array([values])
            special_cpu = mx.random._categorical_fixed(
                special_logits, num_samples=100, key=key, stream=mx.cpu
            )
            mx.eval(special_cpu)
            self.assertTrue(mx.all(special_cpu == expected_index).item())
            if mx.metal.is_available():
                special_gpu = mx.random._categorical_fixed(
                    special_logits, num_samples=100, key=key, stream=mx.gpu
                )
                mx.eval(special_gpu)
                self.assertTrue(mx.array_equal(special_gpu, special_cpu))

        nan_masked = mx.random._categorical_fixed(
            mx.array([[math.nan, 0.0, 1.0]]),
            num_samples=1_000,
            key=key,
            stream=mx.cpu,
        )
        mx.eval(nan_masked)
        self.assertTrue(mx.all(nan_masked != 0).item())

        mapped_logits = mx.array([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.5, 1.5, -0.5]])
        mapped_keys = mx.random.split(mx.random.key(3), num=3)
        both = mx.vmap(candidate)(mapped_logits, mapped_keys)
        logits_only = mx.vmap(candidate, in_axes=(0, None))(
            mapped_logits, mapped_keys[0]
        )
        keys_only = mx.vmap(candidate, in_axes=(None, 0))(mapped_logits[0], mapped_keys)
        both_expected = mx.stack(
            [candidate(mapped_logits[i], mapped_keys[i]) for i in range(3)]
        )
        logits_expected = mx.stack(
            [candidate(mapped_logits[i], mapped_keys[0]) for i in range(3)]
        )
        keys_expected = mx.stack(
            [candidate(mapped_logits[0], mapped_keys[i]) for i in range(3)]
        )
        compiled = mx.compile(candidate)(mapped_logits[0], mapped_keys[0])
        compiled_vmap = mx.compile(mx.vmap(candidate))(mapped_logits, mapped_keys)
        mx.eval(
            both,
            logits_only,
            keys_only,
            both_expected,
            logits_expected,
            keys_expected,
            compiled,
            compiled_vmap,
        )
        self.assertTrue(mx.array_equal(both, both_expected))
        self.assertTrue(mx.array_equal(logits_only, logits_expected))
        self.assertTrue(mx.array_equal(keys_only, keys_expected))
        self.assertTrue(mx.array_equal(compiled, both_expected[0]))
        self.assertTrue(mx.array_equal(compiled_vmap, both_expected))

    def test_categorical_source_dispatch_and_transforms(self):
        def graph(array):
            output = io.StringIO()
            mx.export_to_dot(output, array)
            return output.getvalue()

        backend_stream = mx.gpu if mx.is_available(mx.gpu) else mx.cpu
        expect_fixed = mx.metal.is_available()

        outside_logits = mx.zeros((256,))
        outside = mx.random.categorical(
            outside_logits,
            num_samples=256,
            key=mx.random.key(0),
            stream=backend_stream,
        )
        self.assertNotIn("CategoricalSearch", graph(outside))
        self.assertIn("ArgReduce", graph(outside))

        inside_logits = mx.zeros((8, 256))
        inside = mx.random.categorical(
            inside_logits,
            num_samples=256,
            key=mx.random.key(0),
            stream=backend_stream,
        )
        if expect_fixed:
            self.assertIn("CategoricalSearch", graph(inside))
            self.assertNotIn("ArgReduce", graph(inside))
        else:
            self.assertNotIn("CategoricalSearch", graph(inside))
            self.assertIn("ArgReduce", graph(inside))

        complex_logits = mx.zeros((1024,), dtype=mx.complex64)
        complex_sample = mx.random.categorical(
            complex_logits,
            num_samples=512,
            key=mx.random.key(0),
            stream=backend_stream,
        )
        self.assertNotIn("CategoricalSearch", graph(complex_sample))
        self.assertIn("ArgReduce", graph(complex_sample))

        cpu_fallback = mx.random.categorical(
            inside_logits,
            num_samples=256,
            key=mx.random.key(0),
            stream=mx.cpu,
        )
        self.assertNotIn("CategoricalSearch", graph(cpu_fallback))
        self.assertIn("ArgReduce", graph(cpu_fallback))

        mapped_logits = mx.zeros((3, 1024))
        mapped_keys = mx.random.split(mx.random.key(1), num=3)

        def sample(one_logits, one_key):
            return mx.random.categorical(
                one_logits,
                num_samples=1024,
                key=one_key,
                stream=backend_stream,
            )

        mapped = mx.vmap(sample)(mapped_logits, mapped_keys)
        compiled_mapped = mx.compile(mx.vmap(sample))(mapped_logits, mapped_keys)
        nested_logits = mx.zeros((3, 2, 1024))
        nested_mapped = mx.vmap(sample)(nested_logits, mapped_keys)
        nested_compiled = mx.compile(mx.vmap(sample))(nested_logits, mapped_keys)
        if expect_fixed:
            self.assertIn("CategoricalSearch", graph(mapped))
        else:
            self.assertNotIn("CategoricalSearch", graph(mapped))
        mx.eval(
            inside,
            cpu_fallback,
            mapped,
            compiled_mapped,
            nested_mapped,
            nested_compiled,
        )
        self.assertEqual(inside.shape, (8, 256))
        self.assertTrue(mx.array_equal(mapped, compiled_mapped))
        self.assertEqual(nested_mapped.shape, (3, 2, 1024))
        self.assertTrue(mx.array_equal(nested_mapped, nested_compiled))

    def test_permutation(self):
        x = sorted(mx.random.permutation(4).tolist())
        self.assertEqual([0, 1, 2, 3], x)

        x = mx.array([0, 1, 2, 3])
        x = sorted(mx.random.permutation(x).tolist())
        self.assertEqual([0, 1, 2, 3], x)

        x = mx.array([0, 1, 2, 3])
        x = sorted(mx.random.permutation(x).tolist())

        # 2-D
        x = mx.arange(16).reshape(4, 4)
        out = mx.sort(mx.random.permutation(x, axis=0), axis=0)
        self.assertTrue(mx.array_equal(x, out))
        out = mx.sort(mx.random.permutation(x, axis=1), axis=1)
        self.assertTrue(mx.array_equal(x, out))

        # Basically 0 probability this should fail.
        sorted_x = mx.arange(16384)
        x = mx.random.permutation(16384)
        self.assertFalse(mx.array_equal(sorted_x, x))

        # Preserves shape / doesn't cast input to int
        x = mx.random.permutation(mx.array([[1]]))
        self.assertEqual(x.shape, (1, 1))

    def test_complex_normal(self):
        sample = mx.random.normal(tuple(), dtype=mx.complex64)
        self.assertEqual(sample.shape, tuple())
        self.assertEqual(sample.dtype, mx.complex64)

        sample = mx.random.normal((1, 2, 3, 4), dtype=mx.complex64)
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, mx.complex64)

        sample = mx.random.normal((1, 2, 3, 4), dtype=mx.complex64, scale=2.0, loc=3.0)
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, mx.complex64)

        sample = mx.random.normal(
            (1, 2, 3, 4), dtype=mx.complex64, scale=2.0, loc=3.0 + 1j
        )
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, mx.complex64)

    def test_broadcastable_scale_loc(self):
        b = mx.random.normal((10, 2))
        sample = mx.random.normal((2, 10, 2), loc=b, scale=b)
        mx.eval(sample)
        self.assertEqual(sample.shape, (2, 10, 2))

        with self.assertRaises(ValueError):
            b = mx.random.normal((10,))
            sample = mx.random.normal((2, 10, 2), loc=b, scale=b)

        b = mx.random.normal((3, 1, 2))
        sample = mx.random.normal((3, 4, 2), dtype=mx.float16, loc=b, scale=b)
        mx.eval(sample)
        self.assertEqual(sample.shape, (3, 4, 2))
        self.assertEqual(sample.dtype, mx.float16)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
