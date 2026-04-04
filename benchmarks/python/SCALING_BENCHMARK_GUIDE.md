# MLX Distributed Scaling Benchmark Guide

Comprehensive benchmarking and visualization for MLX distributed operations from 1 to 40 GPUs.

## 📊 Quick Start

### Install Required Dependencies

```bash
# For benchmarking
pip install -r benchmarks/python/requirements.txt

# Or manually:
pip install numpy pandas matplotlib
```

### Run Benchmarks

```bash
# GPU weak scaling (fixed size per GPU)
mpirun -n 8 python benchmarks/python/scaling_benchmark.py --weak-scaling

# GPU strong scaling (fixed total size)
mpirun -n 16 python benchmarks/python/scaling_benchmark.py --strong-scaling

# CPU scaling (single process, multi-threaded)
python benchmarks/python/scaling_benchmark.py --cpu-scaling

# Full benchmark suite
mpirun -n 40 python benchmarks/python/scaling_benchmark.py --full
```

### Generate Visualizations

```bash
# Generate all plots from benchmark results
python benchmarks/python/plot_scaling.py --json scaling_benchmark.json

# Generate specific plot type
python benchmarks/python/plot_scaling.py --json results.json --type efficiency

# Generate summary report only
python benchmarks/python/plot_scaling.py --json results.json --summary

# Specify output directory
python benchmarks/python/plot_scaling.py -j results.json -o my_plots/
```

## 🎯 What Is Benchmarked

### 1. GPU Weak Scaling
- **Fixed**: Data per GPU (e.g., 4MB)
- **Variable**: Total data size (scales with GPUs)
- **Use Case**: Distributed training with proportional data

### 2. GPU Strong Scaling
- **Fixed**: Total data size (e.g., 4MB)
- **Variable**: Data per GPU (decreases with more GPUs)
- **Use Case**: Single computation with increasing resources

### 3. CPU Scaling
- **Fixed**: Data per core
- **Variable**: Number of CPU cores (1-16)
- **Use Case**: Multi-threaded CPU operations

## 📈 Performance Metrics

### Primary Metrics
- **Bandwidth** (GB/s): Data transfer rate
- **Latency** (ms): Time per operation
- **Throughput** (Ops/sec): Operations per second

### Secondary Metrics
- **Scalability Efficiency**: How well performance scales with resources
  - Ideal: 100% (linear speedup)
  - Good: >80%
  - Acceptable: >60%

## 📊 Example Outputs

### All-Reduce Scaling (8 GPUs, 4MB per GPU)

| GPUs | Latency (ms) | Bandwidth (GB/s) | Efficiency |
|------|--------------|------------------|------------|
| 1    | 7.8          | -                | 100%       |
| 2    | 4.2          | 3.8              | 95%        |
| 4    | 2.1          | 4.7              | 95%        |
| 8    | 1.2          | 5.3              | 92%        |

### Best Algorithm Selection

| Message Size | Best Algorithm    | Speedup over Ring |
|--------------|-------------------|-------------------|
| < 1MB        | Recursive Doubling | 1.5x - 2.0x      |
| 1MB - 10MB   | Tree              | 1.8x - 2.5x      |
| > 10MB       | Tree              | 2.0x - 3.0x      |

## 🚀 Usage Examples

### Complete Benchmark Workflow

```bash
# Step 1: Run benchmarks
mpirun -n 8 python benchmarks/python/scaling_benchmark.py \
    --weak-scaling \
    --min-gpus 2 \
    --max-gpus 8 \
    --output weak_scaling_results.json

# Step 2: Visualize results
python benchmarks/python/plot_scaling.py \
    --json weak_scaling_results.json \
    --output scaling_plots

# Step 3: Generate summary
python benchmarks/python/plot_scaling.py \
    --json weak_scaling_results.json \
    --summary
```

### Custom Scaling Benchmarks

```bash
# CPU-only benchmark (16 cores)
python benchmarks/python/scaling_benchmark.py \
    --cpu-scaling \
    --min-cores 1 \
    --max-cores 16

# Custom GPU range (2 to 40)
mpirun -n 40 python benchmarks/python/scaling_benchmark.py \
    --weak-scaling \
    --min-gpus 2 \
    --max-gpus 40

# Strong scaling with specific size
mpirun -n 16 python benchmarks/python/scaling_benchmark.py \
    --strong-scaling \
    --total-size 1048576
```

### Advanced Usage

```bash
# Run with specific algorithms
mpirun -n 4 python benchmarks/python/scaling_benchmark.py \
    --algo default,ring,tree \
    --weak-scaling

# Custom output file
python benchmarks/python/scaling_benchmark.py \
    -o my_scaling_results.json

# Generate specific plot type
python benchmarks/python/plot_scaling.py \
    -j scaling_results.json \
    -t bandwidth

# Create all plots with custom metric
python benchmarks/python/plot_scaling.py \
    -j results.json \
    --metric scalability_efficiency
```

## 📁 Output Files

The benchmark suite generates:

### JSON Results (`scaling_benchmark.json`)
```json
{
  "benchmark_type": "weak_scaling",
  "num_processes_range": [1, 2, 4, 8, 16, 32],
  "size_per_process_elements": 1048576,
  "experiments": {
    "all_reduce_weak_scaling": {...},
    "pipeline_weak_scaling": {...}
  }
}
```

### Visual Plots (`plots/` directory)

| Plot | Description |
|------|-------------|
| `scalability_*.png` | Performance vs GPU count (linear & log scale) |
| `efficiency_*.png` | Scalability efficiency scatter/barchart |
| `algorithm_comparison_*.png` | Multi-algorithm comparison |
| `bandwidth_vs_gpu_*.png` | Bandwidth scaling with theoretical max |
| `speedup_*.png` | Speedup vs ideal scaling |
| `heatmap_*.png` | Performance heatmap (GPU × Algorithm) |
| `throughput_vs_gpu_*.png` | Throughput vs GPU count |
| `latency_vs_gpu_*.png` | Latency vs GPU count |
| `summary_report.png` | Comprehensive summary |

## 🔍 Understanding the Results

### Efficiency Analysis

**Perfect Scalability (100%)**: Each GPU provides equal speedup
```
Time(N GPUs) = Time(1 GPU) / N
```

**Sublinear Scalability (<100%)**: Common due to communication overhead

**Superlinear Scalability (>100%)**: Rare, occurs with cache effects

### Identifying Bottlenecks

| Pattern | Likely Cause |
|---------|-------------|
| Bandwidth plateaus | Network saturation |
| Efficiency drops sharply | Communication overhead |
| Non-monotonic performance | Resource contention |

## 📚 API Reference

### Python API

```python
import mlx.core as mx

# Optimized all-reduce with automatic algorithm selection
result = mx.distributed.all_reduce(x, op="sum")

# Explicit algorithm control
result = mx.distributed.all_reduce_opt(x, algo="tree")

# All-gather with algorithm selection
result = mx.distributed.all_gather_opt(x, algo="tree")

# Reduce-scatter with algorithm selection
result = mx.distributed.reduce_scatter_opt(x, op="sum", algo="tree")

# Pipeline parallelism
stages = [
    mx.distributed.PipelineStage(i, num_stages, compute_fn)
    for i in range(num_stages)
]
output = mx.distributed.execute_pipeline(stages, input_data)
```

### Command-Line Options

```bash
# Scaling benchmark options
python benchmarks/python/scaling_benchmark.py --help

# Plotting options
python benchmarks/python/plot_scaling.py --help
```

## 🎨 Plot Gallery

### Scalability Curves
![Scalability Curve](plots/scalability_all_reduce_weak_scaling.png)
Shows linear and log-log scaling behavior.

### Efficiency Plots
![Efficiency Plot](plots/efficiency_all_reduce_weak_scaling.png)
Scalability efficiency scatter with reference lines.

### Algorithm Comparison
![Algorithm Comparison](plots/algorithm_comparison_all_reduce.png)
Side-by-side comparison of different algorithms.

### Bandwidth Utilization
![Bandwidth Plot](plots/bandwidth_vs_gpu_*.png)
Actual vs theoretical bandwidth limits.

### Speedup Analysis
![Speedup Plot](plots/speedup_*.png)
Actual speedup vs ideal linear speedup.

### Performance Heatmap
![Heatmap](plots/heatmap_bandwidth_gbps.png)
Color-coded performance matrix.

## 📈 Performance Targets

### Good Scaling Characteristics
- **Efficiency**: >80% up to 16 GPUs
- **Bandwidth**: >80% of theoretical max
- **Speedup**: >75% of ideal

### Scalable System Requirements
-NVLink or high-bandwidth interconnect
-Optimized collective algorithms
-Minimal communication overhead

## 🛠️ Troubleshooting

### MPI Issues
```bash
# Check if MPI is installed
mpirun --version

# Test with 2 processes
mpirun -n 2 python benchmarks/python/scaling_benchmark.py --cpu-scaling
```

### Memory Issues
```bash
# Reduce per-GPU size
python benchmarks/python/scaling_benchmark.py \
    --size 262144  # 1MB instead of default
```

### Plot Generation Errors
```bash
# Ensure dependencies are installed
pip install matplotlib numpy pandas

# Check JSON validity
python -c "import json; json.load(open('results.json'))"
```

## 📚 Further Reading

- [MLX Documentation](https://mlx.readthedocs.io)
- [Distributed Computing Guide](https://docs.openhands.dev)
- Collective Algorithms: [arXiv:1802.09945](https://arxiv.org/abs/1802.09945)

## 🐛 Known Limitations

- Requires MPI for distributed benchmarks
- CPU scaling limited to single-process multi-threaded execution
- Theoretical bandwidth assumes NVLink (25 GB/s per GPU)

## 📄 License

Copyright © 2024 Apple Inc. All rights reserved.
