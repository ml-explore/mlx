# MLX Benchmarks

This directory contains various benchmark scripts for MLX operations.

## Prerequisites for Building MLX from Source

Before building MLX, ensure you have the required dependencies:

### macOS (Apple Silicon M5 Max)
MLX requires Apple's Metal framework for GPU acceleration. To build MLX with Metal support, you need:

1. **Full Xcode (not just Command Line Tools)**:
```bash
# Important: You need the full Xcode IDE (not just command line tools)
# Download and install from Mac App Store
# Command Line Tools alone do NOT include the Metal compiler

# After installing Xcode, set it as your developer directory:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Accept the license:
sudo xcodebuild -license accept
```

2. **Verify Metal is available**:
```bash
# Check if metal compiler is in PATH
xcrun --find metal

# Should output something like:
# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
```

3. **Build MLX**:
```bash
# From the project root directory
pip install -e .
```

⚠️ **Common Error**: If you get `xcrun: error: unable to find utility "metal"`, this means:
- You only have Command Line Tools installed (not full Xcode)
- Solution: Install full Xcode from App Store

### Alternative: Install Pre-built MLX

If building from source fails, you can install the official MLX wheel:

```bash
# Install MLX from PyPI (pre-built for macOS)
pip install mlx

# Verify installation
python -c "import mlx.core as mx; print(mx.__version__)"
```

Note: The pre-built wheels may not include your custom M5 Max optimizations. For full optimization support, build from source.

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

### "Failed to build mlx" - Metal Compiler Not Found

If you encounter this error when building MLX from source:

```
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
make[2]: *** [mlx/backend/metal/kernels/arg_reduce.air] Error 72
```

**Cause**: Command Line Tools alone don't include the Metal compiler. You need full **Xcode** (the IDE).

#### Solution Options:

**Option 1: Install Full Xcode (Recommended)**

The command line tools you have installed don't include the Metal compiler. Install Xcode from the App Store:

```bash
# 1. Download and install Xcode from Mac App Store
#    (Search for "Xcode" in App Store and install)

# 2. Switch to Xcode's developer directory
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# 3. Accept the license
sudo xcodebuild -license accept

# 4. Verify Metal compiler is available
xcrun --find metal
# Should output: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
```

**Option 2: Use Pre-built MLX Wheel**

If installing Xcode isn't feasible, use the pre-built wheel (may not have custom optimizations):

```bash
pip install mlx
```

**Option 3: Build without Metal (CPU-only)**

If you only need CPU support:

```bash
# Install MLX with CPU backend only (no Metal)
pip install mlx[cpu]
```

After installing Xcode, retry building MLX:

```bash
cd /Users/martinolsson/Documents/GitHub/mlx
pip install -e .
```

### "Failed building editable for mlx"
This typically means the build process failed during compilation. Common causes:

1. **Missing Metal compiler** - See previous error
2. **Insufficient disk space** - Ensure at least 5GB free
3. **Outdated Xcode** - Update to latest version from App Store

Build with verbose output for debugging:

```bash
pip install -e . --verbose
```

## Contributing Benchmarks

To add a new benchmark:

1. Create a new file in `benchmarks/python/`
2. Follow the pattern of existing benchmarks
3. Use `time_utils.py` for consistent timing
4. Add your benchmark to the M5 Max benchmark suite if it's general enough
