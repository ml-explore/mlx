# ABOUTME: Tests for paged attention microbench sweep utilities.
# ABOUTME: Validates block/TG/vector sweep expansion logic.

import importlib.util
from pathlib import Path


def _load_bench_module():
    bench_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "python"
        / "paged_attention_bench.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paged_attention_bench", bench_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expand_sweeps_defaults():
    bench = _load_bench_module()
    parser = bench.build_parser()
    args = parser.parse_args([])
    configs = bench.expand_sweep_args(args)
    assert len(configs) == 1
    config = configs[0]
    assert config.block_size == args.block_size
    assert config.tg_size == args.tg_size
    assert config.vec_width == args.vec_width


def test_expand_sweeps_cartesian_product():
    bench = _load_bench_module()
    parser = bench.build_parser()
    args = parser.parse_args(
        [
            "--block-sizes",
            "32",
            "64",
            "--tg-sizes",
            "64",
            "128",
            "--vec-widths",
            "1",
            "2",
        ]
    )
    configs = bench.expand_sweep_args(args)
    assert len(configs) == 8
    seen = {(c.block_size, c.tg_size, c.vec_width) for c in configs}
    expected = {
        (b, t, v)
        for b in (32, 64)
        for t in (64, 128)
        for v in (1, 2)
    }
    assert seen == expected
