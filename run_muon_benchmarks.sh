#!/bin/bash

# MLX Muon Benchmark Runner
# Based on the MLX-Muon playbook validation criteria

echo "🚀 MLX Muon Optimizer Benchmark Suite"
echo "======================================"

# Create results directory
mkdir -p benchmarks/results

# Check if we're in the right directory
if [ ! -f "python/mlx/optimizers/optimizers.py" ]; then
    echo "❌ Error: Please run this script from the MLX root directory"
    exit 1
fi

# Set Python path to include our MLX installation
export PYTHONPATH="$PWD/python:$PYTHONPATH"

echo "📊 Running comparison benchmarks..."
python benchmarks/mlx_muon_benchmark.py --compare

echo ""
echo "✅ Benchmark complete! Check benchmarks/results/ for detailed results."
