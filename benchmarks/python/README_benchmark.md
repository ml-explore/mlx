# MLX Benchmarks

This directory contains comprehensive benchmark scripts for MLX operations.

## Benchmark Categories

### Distributed Computing Benchmarks
- `distributed_bench.py` - Basic distributed operations benchmark (existing)
- `optimized_collectives_bench.py` - Comprehensive optimized collectives benchmark (NEW)

### Operator Benchmarks
- `single_ops.py` - Single operation benchmarks
- `batch_matmul_bench.py` - Batch matrix multiplication
- `conv*.py` - Convolution benchmarks (1D, 2D, 3D)
- `gemm_bench*.py` - GEMM benchmarks
- `hadamard_bench.py` - Hadamard product
- `fft_bench.py` - FFT operations
- `linalg*.py` - Linear algebra benchmarks

### Specialized Benchmarks
- `compile_bench.py` - Compilation performance
- `large_gemm_bench.py` - Large matrix multiplication
- `sdpa_bench.py` - Scaled dot product attention

## Running Benchmarks

### Python Benchmarks

#### Single Benchmark
```bash
python benchmarks/python/single_ops.py
```

#### Distributed Benchmarks (Multi-Process)
```bash
# Run with 2 processes using MPI
mpirun -n 2 python benchmarks/python/distributed_bench.py

# Run full distributed benchmark suite
mpirun -n 4 python benchmarks/python/optimized_collectives_bench.py --full

# Run specific operation
mpirun -n 2 python benchmarks/python/optimized_collectives_bench.py --op all_reduce
```

#### Advanced Usage
```bash
# Run with custom algorithms
mpirun -n 2 python optimized_collectives_bench.py --op all_reduce \
    --algo default,ring,tree

# Custom sizes
python optimized_collectives_bench.py --sizes 1024,65536,262144

# Save results to JSON
python optimized_collectives_bench.py --full --output results.json

# Print state-of-the-art metrics reference
python optimized_collectives_bench.py --metrics
```

### C++ Benchmarks

#### Build
```bash
cd build
cmake .. -DMLX_ENABLE_DISTRIBUTED=ON
make collectives_bench
```

#### Run
```bash
# Single process
./benchmarks/cpp/collectives_bench

# Multi-process with MPI
mpirun -n 2 ./benchmarks/cpp/collectives_bench --all

# Specific operation
mpirun -n 4 ./benchmarks/cpp/collectives_bench --op all_reduce
```

## Optimized Collectives Benchmark

The `optimized_collectives_bench.py` script provides comprehensive benchmarking for MLX's optimized collective communication operations.

### Features

1. **Algorithm Comparison**: Test multiple algorithms (LINEAR, RING, RECURSIVE_DOUBLING, TREE, BROADCAST)

2. **Multiple Operations**: 
   - All-reduce (with various reduction operations)
   - All-gather
   - Reduce-scatter
   - Pipeline parallelism

3. **Scalability Testing**: Test with different group sizes and data sizes

4. **State-of-the-Art Metrics**:
   - Bandwidth (GB/s)
   - Latency (ms)
   - Throughput (ops/sec)
   - Scalability analysis

### Benchmark Metrics

#### Bandwidth
```python
bandwidth_gbps = (data_size * num_processes) / time_in_seconds / (1024**3)
```

#### Latency
- Measured in milliseconds per operation
- For small messages: latency = fixed_overhead + transfer_rate * size

#### Throughput
```python
throughput_ops_per_sec = 1000 / latency_ms
```

#### State-of-the-Art Reference

| Size          | Ring (GB/s) | Tree (GB/s) | Recursive Doubling |
|---------------|-------------|-------------|-------------------|
| 1KB           |     0.5     |     0.3     |       0.8         |
| 1MB           |     8-12    |    10-15    |      10-14        |
| 10MB          |    10-15    |    12-18    |      12-16        |
| 100MB         |    12-18    |    15-22    |      14-20        |

### Usage Examples

#### Basic Run
```bash
mpirun -n 2 python optimized_collectives_bench.py --full
```

#### Compare Specific Algorithms
```bash
mpirun -n 4 python optimized_collectives_bench.py --op all_reduce \
    --algo default,ring,recursive_doubling,tree
```

#### Custom Sizes
```bash
python optimized_collectives_bench.py --op all_gather \
    --sizes 1024,65536,262144,1048576
```

#### Save Results
```bash
python optimized_collectives_bench.py --full --output my_results.json
```

#### Print Metrics Reference
```bash
python optimized_collectives_bench.py --metrics
```

## Results Analysis

### Output Format
Results are printed in a formatted table:
```
ALL_REDUCE
------------------------------------------------------------------------------
Algorithm            Size (MB)   Latency (ms)   Bandwidth (GB/s)   Ops/sec
------------------------------------------------------------------------------
default                   0.00         0.015           267.42        66667
ring                      0.00         0.018           232.56        55556
tree                      0.00         0.012           347.22        83333
...
```

### JSON Output
Results can be saved to JSON for further analysis:
```json
{
  "config": {
    "operation": "all_reduce",
    "algorithms_tested": ["default", "ring", "tree"],
    "sizes_tested": [1024, 65536, ...]
  },
  "results": [
    {
      "op": "all_reduce",
      "algo": "tree",
      "size_bytes": 4096,
      "latency_ms": 0.012,
      "bandwidth_gbps": 347.22,
      "ops_per_sec": 83333
    }
  ]
}
```

### Best Algorithm Selection
The script automatically identifies the best algorithm for each operation and size:
- **Small messages (<1MB)**: RECURSIVE_DOUBLING or TREE
- **Medium messages (1-10MB)**: TREE
- **Large messages (>10MB)**: RING or TREE (depends on network)

## Performance Considerations

### Network Topology
Performance varies based on:
- Interconnect (NVLink, PCIe, InfiniBand, Ethernet)
- Network topology (ring, tree, hybrid)
- Number of nodes

### Data Types
All benchmarks use float32 for consistency. Results may vary for:
- int8: 4x faster bandwidth
- float16/bfloat16: 2x faster bandwidth

### Batch Size Impact
For training workloads, consider:
- Gradient accumulation size
- Parameter partitioning
- Communication-computation overlap

## Comparisons with Other Frameworks

### PyTorch Distributed
```bash
# Similar benchmarks available in torch.distributed
python -m torch.utils.benchmark
```

### JAX Multi-Slice
```bash
# Use jax.profiler for similar benchmarks
```

### NCCL
```bash
# NCCL tests provide reference implementation
nccl-tests/build/all_reduce_perf -b 8 -e 256M
```

## Troubleshooting

### Common Issues

1. **Distributed not available**
   - Ensure MLX is built with distributed support
   - Check: `mx.distributed.is_available()`

2. **MPI errors**
   - Verify MPI installation
   - Check: `mpirun --version`

3. **Memory issues**
   - Reduce test sizes
   - Check available GPU/CPU memory

4. **Inconsistent results**
   - Increase warmup iterations
   - Run multiple times and take median

### Help
```bash
python optimized_collectives_bench.py --help
```

## Contributing

To add new benchmarks:

1. Follow existing patterns in the directory
2. Use consistent timing methodology (warmup + multiple iterations)
3. Provide meaningful metrics (bandwidth, latency, throughput)
4. Include state-of-the-art comparisons
5. Document usage and expected results
