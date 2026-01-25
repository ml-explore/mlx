#!/bin/bash
# Script to test ROCm backend compilation using Docker
# No AMD GPU required - just tests that the code compiles

set -e

IMAGE="rocm/dev-ubuntu-22.04:6.0"

echo "=== MLX ROCm Backend Compilation Test ==="
echo "Using Docker image: $IMAGE"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Please start Docker Desktop"
    exit 1
fi

echo "Pulling ROCm development image (this may take a while on first run)..."
docker pull $IMAGE

echo ""
echo "Starting compilation test..."
echo ""

# Run the build in Docker
# Note: ROCm images are x86_64 only, so we use --platform linux/amd64
# This runs via emulation on Apple Silicon (slower but works)
docker run --rm \
    --platform linux/amd64 \
    -v "$(pwd)":/workspace \
    -w /workspace \
    $IMAGE \
    bash -c '
        set -e
        echo "=== Installing dependencies ==="
        apt-get update -qq
        apt-get install -y -qq build-essential python3-pip liblapack-dev liblapacke-dev libopenblas-dev git wget rocblas-dev rocthrust-dev rocprim-dev hiprand-dev > /dev/null 2>&1
        
        # Install ROCm libraries needed for MLX
        echo "=== Installing ROCm libraries ==="
        apt-get install -y -qq rocblas-dev rocthrust-dev rocprim-dev hiprand-dev > /dev/null 2>&1
        
        # Install newer CMake (3.25+)
        echo "=== Installing CMake 3.28 ==="
        wget -q https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.tar.gz
        tar -xzf cmake-3.28.0-linux-x86_64.tar.gz
        export PATH=$(pwd)/cmake-3.28.0-linux-x86_64/bin:$PATH
        cmake --version

        echo "=== Configuring CMake ==="
        rm -rf build_rocm_test
        mkdir build_rocm_test
        cd build_rocm_test
        
        # Set ROCm paths for CMake to find packages
        export ROCM_PATH=/opt/rocm-6.0.0
        export CMAKE_PREFIX_PATH=$ROCM_PATH:$ROCM_PATH/lib/cmake:$CMAKE_PREFIX_PATH
        
        cmake .. \
            -DMLX_BUILD_ROCM=ON \
            -DMLX_BUILD_METAL=OFF \
            -DMLX_BUILD_CUDA=OFF \
            -DMLX_BUILD_TESTS=OFF \
            -DMLX_BUILD_EXAMPLES=OFF \
            -DMLX_BUILD_BENCHMARKS=OFF \
            -DMLX_BUILD_PYTHON_BINDINGS=OFF \
            -DMLX_ROCM_ARCHITECTURES="gfx906;gfx1030" \
            2>&1
        
        echo ""
        echo "=== Building MLX with ROCm backend ==="
        make -j$(nproc) 2>&1
        
        echo ""
        echo "=== Build successful! ==="
    '

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo "✓ ROCm backend compilation test PASSED"
    echo ""
    echo "The build directory is at: ./build_rocm_test"
else
    echo ""
    echo "✗ ROCm backend compilation test FAILED"
    exit 1
fi
