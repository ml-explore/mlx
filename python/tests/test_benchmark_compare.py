# Copyright © 2026 Apple Inc.

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_PATH = REPO_ROOT / "benchmarks/python/comparative/compare.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def load_compare_module():
    spec = importlib.util.spec_from_file_location("benchmark_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkCompareUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = load_compare_module()

    def test_run_repeated_reports_samples_and_summary(self):
        with mock.patch.object(
            self.compare, "run_or_raise", side_effect=[0.3, 0.1, 0.2]
        ):
            result = self.compare.run_repeated(["benchmark"], 3)

        self.assertEqual(result["samples_seconds"], [0.3, 0.1, 0.2])
        self.assertEqual(result["median_seconds"], 0.2)
        self.assertEqual(result["min_seconds"], 0.1)
        self.assertEqual(result["max_seconds"], 0.3)

    def test_run_or_raise_includes_child_process_error(self):
        command = [
            sys.executable,
            "-c",
            "import sys; print('details', file=sys.stderr); sys.exit(7)",
        ]

        with self.assertRaisesRegex(RuntimeError, "details"):
            self.compare.run_or_raise(command)

    def test_run_or_raise_rejects_invalid_timing_output(self):
        command = [sys.executable, "-c", "print('not-a-timing')"]

        with self.assertRaisesRegex(ValueError, "invalid timing data"):
            self.compare.run_or_raise(command)

    def test_dtype_comparison_records_both_summaries(self):
        first = {
            "median_seconds": 1.0,
            "min_seconds": 0.9,
            "max_seconds": 1.1,
            "samples_seconds": [1.0],
        }
        second = {
            "median_seconds": 2.0,
            "min_seconds": 1.9,
            "max_seconds": 2.1,
            "samples_seconds": [2.0],
        }
        results = []

        with mock.patch.object(
            self.compare, "run_repeated", side_effect=[first, second]
        ):
            self.compare.compare_mlx_dtypes(
                ["cumsum", "--cpu"], "float32", "float16", 1, results
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["dtypes"], ["float32", "float16"])
        self.assertEqual(results[0]["mlx"]["float32"], first)
        self.assertEqual(results[0]["mlx"]["float16"], second)
        self.assertEqual(results[0]["relative_speedup"], 0.5)

    def test_predicate_combines_positive_and_negative_filters(self):
        predicate = self.compare.make_predicate([r"^cumsum", r"--cpu$"], [r"axis 1"])

        self.assertTrue(predicate("cumsum --size 8x8 --axis 0 --cpu"))
        self.assertFalse(predicate("cumsum --size 8x8 --axis 1 --cpu"))
        self.assertFalse(predicate("sum_axis --size 8x8 --axis 0 --cpu"))


class BenchmarkCompareCliTest(unittest.TestCase):
    def test_invalid_repeat_count_fails_before_running_benchmarks(self):
        result = subprocess.run(
            [sys.executable, COMPARE_PATH, "--repeats", "0"],
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--repeats must be at least 1", result.stderr)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for comparative benchmarks")
class BenchmarkCompareIntegrationTest(unittest.TestCase):
    def run_compare(self, *args):
        return subprocess.run(
            [sys.executable, COMPARE_PATH, *args],
            capture_output=True,
            check=True,
            cwd=REPO_ROOT,
            text=True,
            timeout=120,
        )

    def test_json_output_from_real_mlx_and_torch_benchmarks(self):
        result = self.run_compare(
            "--filter",
            r"^cumsum --size 128x1024 --axis 0$",
            "--cpu",
            "--repeats",
            "2",
            "--json",
        )
        report = json.loads(result.stdout)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["metadata"]["repeats"], 2)
        self.assertEqual(len(report["results"]), 1)

        benchmark = report["results"][0]
        self.assertEqual(benchmark["benchmark"], "cumsum")
        self.assertEqual(
            benchmark["arguments"],
            ["--size", "128x1024", "--axis", "0", "--cpu"],
        )
        self.assertGreater(benchmark["mlx"]["median_seconds"], 0)
        self.assertGreater(benchmark["torch"]["median_seconds"], 0)
        self.assertEqual(len(benchmark["mlx"]["samples_seconds"]), 2)
        self.assertEqual(len(benchmark["torch"]["samples_seconds"]), 2)

    def test_default_output_from_real_mlx_and_torch_benchmarks(self):
        result = self.run_compare(
            "--filter",
            r"^cumsum --size 128x1024 --axis 0$",
            "--cpu",
        )

        speedup, benchmark = result.stdout.rstrip().split("\t", maxsplit=1)
        float(speedup)
        self.assertEqual(benchmark, "cumsum --size 128x1024 --axis 0 --cpu")


if __name__ == "__main__":
    unittest.main()
