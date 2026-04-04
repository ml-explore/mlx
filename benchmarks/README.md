# MLX Benchmarks

This directory contains various benchmark scripts for MLX operations.

## Running Benchmarks

To run benchmarks, you need to have MLX installed. If you're working with this fork, ensure MLX is built and installed:

```bash
# From the project root directory
pip install -e .
```

Or if you have a built version:

```bash
python setup.py install
```

## Available Benchmarks

### Python Benchmarks (`benchmarks/python/`)

#### M5 Max Specific
- `m5_max_bench.py` - Comprehensive benchmark suite for Apple Silicon M5 Max

#### General Benchmarks
- `single_ops.py` - Single operation benchmarks
- `batch_matmul_bench.py` - Batch matmul operations
- `conv_bench.py` - Convolution benchmarks
- `gemm_bench.py` - GEMM (General Matrix Multiply) benchmarks

## Usage Examples

### M5 Max Benchmark Suite

```bash
# Run GPU benchmarks (default)
python -m benchmarks.python.m5_max_bench

# Run CPU benchmarks
python -m benchmarks.python.m5_max_bench --cpu

# Save results to JSON file
python -m benchmarks.python.m5_max_bench --output m5_max_results.json

# View all options
python -m benchmarks.python.m5_max_bench --help
```

### Single Operations Benchmark

```bash
# Run on GPU
python -m benchmarks.python.single_ops --gpu

# Run on CPU
python -m benchmarks.python.single_ops
```

### Matmul Benchmarks

```bash
# Run matmul benchmarks on GPU
python -m benchmarks.python.batch_matmul_bench --gpu

# Run matmul benchmarks on CPU
python -m benchmarks.python.batch_matmul_bench
```

## Output Format

The M5 Max benchmark produces JSON output:

```json
{
  "timestamp": "2026-04-04T12:00:00.000000",
  "device": "Apple M5 Max",
  "benchmarks": {
    "matmul": [
      {
        "test": "large_nn",
        "shape": "1024x1024 @ 1024x1024",
        "mean": 1.234,
        "min": 1.200,
        "max": 1.300,
        "std": 0.025,
        "num_iters": 100
      }
    ],
    "reduce": [...],
    "element_wise": [...]
  }
}
```

## Interpreting Results

- **mean**: Average execution time in milliseconds
- **min**: Minimum execution time (best case)
- **max**: Maximum execution time (worst case)
- **std**: Standard deviation (indicates consistency)

Lower times indicate better performance.

## Comparing Results

To compare performance between different configurations:

```bash
# Run baseline (before optimization)
python -m benchmarks.python.m5_max_bench --output before.json

# Run with optimizations
python -m benchmarks.python.m5_max_bench --output after.json

# Compare (requires custom script or manual comparison)
```

## Tips for Accurate Benchmarking

1. **Warm-up**: Benchmarks automatically perform warm-up iterations
2. **Multiple runs**: 100 iterations are performed for statistical significance
3. **Consistent environment**: Run benchmarks in the same environment for fair comparison
4. **Disable GPU throttling**: Ensure your Mac is plugged in and not throttling
5. **Close other apps**: Minimize background processes for consistent results

## Troubleshooting

### "No module named mlx.core"
- Ensure MLX is installed: `pip install mlx`
- Or build from source and install

### "No module named benchmarks.python.m5_max_bench"
- Ensure you're running from the project root directory
- Check that `benchmarks/python` is in your Python path

### Inconsistent results
- Run multiple times and average the results
- Check for system load during benchmarking
- Ensure thermal conditions are consistent

## Contributing Benchmarks

To add a new benchmark:

1. Create a new file in `benchmarks/python/`
2. Follow the pattern of existing benchmarks
3. Use `time_utils.py` for consistent timing
4. Add your benchmark to the M5 Max benchmark suite if it's general enough
