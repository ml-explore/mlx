#!/bin/bash

# M5 Max Benchmark Runner Script
# Usage: ./run_benchmarks.sh [OPTIONS]
#
# Options:
#   --cpu       Run CPU benchmarks (default: GPU)
#   --output FILE  Save results to FILE
#   --help      Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse arguments
CPU_MODE=false
OUTPUT_FILE=""
ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            CPU_MODE=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_benchmarks.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu       Run CPU benchmarks (default: GPU)"
            echo "  --output FILE  Save results to FILE"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

echo "=========================================="
echo "MLX M5 Max Performance Benchmarks"
echo "=========================================="

# Check if MLX is available
python3 -c "import mlx.core as mx" 2>/dev/null || {
    echo "Error: MLX not found. Please install it first."
    echo ""
    echo "Install MLX from PyPI:"
    echo "  pip install mlx"
    echo ""
    echo "Or build and install from source:"
    echo "  cd /path/to/mlx"
    echo "  pip install -e ."
    exit 1
}

echo "Device: $(python3 -c 'import mlx.core as mx; print(mx.default_device())')"
echo ""

# Run benchmark
CMD="python3 -m benchmarks.python.m5_max_bench $ARGS"

if [ "$CPU_MODE" = true ]; then
    CMD="$CMD --cpu"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output $OUTPUT_FILE"
fi

echo "Running: $CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "=========================================="
if [ -n "$OUTPUT_FILE" ]; then
    echo "Results saved to: $OUTPUT_FILE"
fi
echo "=========================================="
