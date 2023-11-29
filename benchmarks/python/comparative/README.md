Microbenchmarks comparing MLX to PyTorch
========================================

Implement the same microbenchmarks in MLX and PyTorch to compare and make a
list of the biggest possible performance improvements and/or regressions.

Run with `python bench_mlx.py sum_axis --size 8x1024x128 --axis 2 --cpu` for
instance to measure the times it takes to sum across the 3rd axis of the above
tensor on the cpu.

`compare.py` runs several benchmarks and compares the speed-up or lack thereof
in comparison to PyTorch.

Each bench script can be run with `--print-pid` to print the PID and wait for a
key in order to ease attaching a debugger.
