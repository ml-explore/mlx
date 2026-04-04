// Copyright © 2024 Apple Inc.

/**
 * Comprehensive benchmark for MLX optimized collective communications.
 * 
 * This benchmark tests:
 * - All-reduce operations with different algorithms
 * - All-gather operations
 * - Reduce-scatter operations
 * - Scalability across different group sizes
 * 
 * Build and run:
 *   mkdir -p build && cd build
 *   cmake .. && make collectives_bench
 *   mpirun -n 2 ./benchmarks/cpp/collectives_bench
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>

#include "mlx/mlx.h"

namespace mx = mlx::core;

// Benchmark timing utility
double benchmark_fn(
    std::function<mx::array()> fn,
    int warmup = 5,
    int iterations = 20) {
  
  // Warmup
  for (int i = 0; i < warmup; ++i) {
    mx::eval(fn());
  }
  
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    mx::eval(fn());
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
  return duration_ms / iterations;
}

void print_header(const std::string& title) {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(80, '=') << std::endl;
}

void print_subheader(const std::string& title) {
  std::cout << "\n" << std::string(60, '-') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(60, '-') << std::endl;
}

void benchmark_all_reduce(const mx::Group& group) {
  print_subheader("All-Reduce Benchmark");
  
  // Test sizes in elements
  std::vector<int> sizes = {
    1024,           // 4KB
    65536,          // 256KB
    262144,         // 1MB
    1048576,        // 4MB
    4194304         // 16MB
  };
  
  std::vector<std::string> algorithms = {
    "default",
    "linear",
    "ring",
    "recursive_doubling",
    "tree"
  };
  
  std::cout << std::left
            << std::setw(20) << "Algorithm"
            << std::setw(15) << "Size (MB)"
            << std::setw(15) << "Latency (ms)"
            << std::setw(15) << "Bandwidth (GB/s)"
            << std::setw(10) << "Ops/sec" << std::endl;
  std::cout << std::string(75, '-') << std::endl;
  
  for (const auto& algo : algorithms) {
    for (int size : sizes) {
      // Create array
      auto x = mx::random::normal({size});
      mx::eval(x);
      
      // Benchmark
      auto fn = [&]() {
        return mx::distributed::all_reduce_opt(x, "sum", algo);
      };
      
      double latency_ms = benchmark_fn(fn);
      
      // Calculate metrics
      int size_bytes = size * 4; // float32 = 4 bytes
      double bandwidth_gbps = (size_bytes * group.size()) / (latency_ms / 1000.0) / (1e9);
      double ops_per_sec = 1000.0 / latency_ms;
      
      std::cout << std::left
                << std::setw(20) << algo
                << std::setw(15) << (size_bytes / (1024.0 * 1024.0))
                << std::setw(15) << latency_ms
                << std::setw(15) << bandwidth_gbps
                << static_cast<int>(ops_per_sec) << std::endl;
    }
  }
}

void benchmark_all_gather(const mx::Group& group) {
  print_subheader("All-Gather Benchmark");
  
  std::vector<int> sizes = {
    1024,
    65536,
    262144,
    1048576
  };
  
  std::vector<std::string> algorithms = {
    "default",
    "ring",
    "tree"
  };
  
  std::cout << std::left
            << std::setw(20) << "Algorithm"
            << std::setw(15) << "Size (MB)"
            << std::setw(15) << "Latency (ms)"
            << std::setw(15) << "Bandwidth (GB/s)"
            << std::setw(10) << "Ops/sec" << std::endl;
  std::cout << std::string(75, '-') << std::endl;
  
  for (const auto& algo : algorithms) {
    for (int size : sizes) {
      auto x = mx::random::normal({size});
      mx::eval(x);
      
      auto fn = [&]() {
        return mx::distributed::all_gather_opt(x, algo);
      };
      
      double latency_ms = benchmark_fn(fn);
      
      int size_bytes = size * 4;
      // All-gather: each rank contributes size, total is size * group.size()
      double bandwidth_gbps = (size_bytes * group.size()) / (latency_ms / 1000.0) / (1e9);
      double ops_per_sec = 1000.0 / latency_ms;
      
      std::cout << std::left
                << std::setw(20) << algo
                << std::setw(15) << (size_bytes / (1024.0 * 1024.0))
                << std::setw(15) << latency_ms
                << std::setw(15) << bandwidth_gbps
                << static_cast<int>(ops_per_sec) << std::endl;
    }
  }
}

void benchmark_reduce_scatter(const mx::Group& group) {
  print_subheader("Reduce-Scatter Benchmark");
  
  // Sizes must be divisible by group.size()
  std::vector<int> sizes = {
    65536,
    262144,
    1048576
  };
  
  std::vector<std::string> algorithms = {
    "default",
    "ring"
  };
  
  std::cout << std::left
            << std::setw(20) << "Algorithm"
            << std::setw(15) << "Size (MB)"
            << std::setw(15) << "Latency (ms)"
            << std::setw(15) << "Bandwidth (GB/s)"
            << std::setw(10) << "Ops/sec" << std::endl;
  std::cout << std::string(75, '-') << std::endl;
  
  for (const auto& algo : algorithms) {
    for (int size : sizes) {
      // Ensure size is divisible by group.size()
      int aligned_size = ((size / group.size()) + 1) * group.size();
      
      auto x = mx::random::normal({aligned_size});
      mx::eval(x);
      
      auto fn = [&]() {
        return mx::distributed::reduce_scatter_opt(x, "sum", algo);
      };
      
      double latency_ms = benchmark_fn(fn);
      
      int size_bytes = aligned_size * 4;
      // Reduce-scatter: each rank receives size/group.size(), total is size
      double bandwidth_gbps = (size_bytes * group.size()) / (latency_ms / 1000.0) / (1e9);
      double ops_per_sec = 1000.0 / latency_ms;
      
      std::cout << std::left
                << std::setw(20) << algo
                << std::setw(15) << (size_bytes / (1024.0 * 1024.0))
                << std::setw(15) << latency_ms
                << std::setw(15) << bandwidth_gbps
                << static_cast<int>(ops_per_sec) << std::endl;
    }
  }
}

void benchmark_pipeline(const mx::Group& group) {
  print_subheader("Pipeline Parallelism Benchmark");
  
  std::vector<int> sizes = {
    1024,
    65536
  };
  
  int num_stages = 4;
  
  std::cout << std::left
            << std::setw(20) << "Stages"
            << std::setw(15) << "Size (MB)"
            << std::setw(20) << "Latency (ms)"
            << std::setw(15) << "Ops/sec" << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  
  for (int size : sizes) {
    auto input = mx::random::normal({size});
    mx::eval(input);
    
    // Create pipeline stages
    std::vector<mx::distributed::PipelineStage> stages;
    for (int i = 0; i < num_stages; ++i) {
      auto stage_fn = [i](const mx::array& x) -> mx::array {
        // Simple computation
        auto result = x;
        for (int j = 0; j < 3; ++j) {
          result = mx::sin(result);
        }
        return result;
      };
      
      stages.emplace_back(i, num_stages, stage_fn);
    }
    
    auto fn = [&]() {
      return mx::distributed::execute_pipeline(stages, input);
    };
    
    double latency_ms = benchmark_fn(fn, 3, 10);
    double ops_per_sec = 1000.0 / latency_ms;
    
    int size_bytes = size * 4;
    
    std::cout << std::left
              << std::setw(20) << num_stages
              << std::setw(15) << (size_bytes / (1024.0 * 1024.0))
              << std::setw(20) << latency_ms
              << static_cast<int>(ops_per_sec) << std::endl;
  }
}

void benchmark_bandwidth_scaling(const mx::Group& group) {
  print_subheader("Bandwidth Scaling Benchmark");
  
  std::vector<int> sizes = {
    65536,
    262144,
    1048576
  };
  
  std::vector<std::string> algorithms = {
    "ring",
    "tree"
  };
  
  std::cout << "Group size: " << group.size() << std::endl;
  std::cout << std::left
            << std::setw(20) << "Algorithm"
            << std::setw(15) << "Size (MB)"
            << std::setw(20) << "Bandwidth (GB/s)" << std::endl;
  std::cout << std::string(55, '-') << std::endl;
  
  for (const auto& algo : algorithms) {
    for (int size : sizes) {
      auto x = mx::random::normal({size});
      mx::eval(x);
      
      auto fn = [&]() {
        return mx::distributed::all_reduce_opt(x, "sum", algo);
      };
      
      double latency_ms = benchmark_fn(fn, 3, 10);
      
      int size_bytes = size * 4;
      double bandwidth_gbps = (size_bytes * group.size()) / (latency_ms / 1000.0) / (1e9);
      
      std::cout << std::left
                << std::setw(20) << algo
                << std::setw(15) << (size_bytes / (1024.0 * 1024.0))
                << std::setw(20) << bandwidth_gbps << std::endl;
    }
  }
}

void print_state_of_the_art_metrics() {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "STATE-OF-THE-ART PERFORMANCE METRICS REFERENCE" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  
  std::string metrics = R"(
1. BANDWIDTH METRICS
   - GB/s (Gigabytes per second): Primary metric for collective operations
   - Effective bandwidth = (data_size * num_processes) / time
   - Ideal: Limited by interconnect bandwidth (NVLink, PCIe, InfiniBand)

2. LATENCY METRICS
   - ms (milliseconds): Time for small messages (<1KB)
   - Latency = T0 + T1 * data_size
   - T0: fixed overhead (protocol, scheduling)
   - T1: per-byte transfer time

3. THROUGHPUT METRICS
   - operations/sec: How many collectives per second
   - Important for training workloads with frequent syncs

4. SCALING METRICS
   - Weak scaling: Fixed workload per process
   - Strong scaling: Total work fixed, more processes = less per process
   - Ideal linear scaling: N processes = 1/N time

5. EXPECTED PERFORMANCE (approximate)
   | Size          | Ring (GB/s) | Tree (GB/s) | Recursive Doubling |
   |---------------|-------------|-------------|-------------------|
   | 1KB           |     0.5     |     0.3     |       0.8         |
   | 1MB           |     8-12    |    10-15    |      10-14        |
   | 10MB          |    10-15    |    12-18    |      12-16        |
   | 100MB         |    12-18    |    15-22    |      14-20        |
)";
  
  std::cout << metrics << std::endl;
}

int main(int argc, char** argv) {
  // Initialize distributed group
  mx::distributed::Group group = mx::distributed::init(false);
  
  std::cout << "\nMLX Optimized Collectives Benchmark" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "Number of processes: " << group.size() << std::endl;
  
  // Parse arguments
  bool run_all = false;
  std::string operation = "";
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--all" || arg == "-a") {
      run_all = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: ./collectives_bench [OPTIONS]" << std::endl;
      std::cout << "\nOptions:" << std::endl;
      std::cout << "  -a, --all       Run all benchmarks" << std::endl;
      std::cout << "  -m, --metrics   Print state-of-the-art metrics reference" << std::endl;
      std::cout << "  -h, --help      Print this help message" << std::endl;
      return 0;
    } else if (arg == "--metrics" || arg == "-m") {
      print_state_of_the_art_metrics();
      return 0;
    } else if (arg.size() > 2 && arg.substr(0, 2) == "--op") {
      operation = arg.substr(4);
    }
  }
  
  if (run_all) {
    print_header("FULL BENCHMARK SUITE");
    
    benchmark_all_reduce(group);
    benchmark_all_gather(group);
    benchmark_reduce_scatter(group);
    benchmark_pipeline(group);
    benchmark_bandwidth_scaling(group);
  } else if (operation == "all_reduce") {
    benchmark_all_reduce(group);
  } else if (operation == "all_gather") {
    benchmark_all_gather(group);
  } else if (operation == "reduce_scatter") {
    benchmark_reduce_scatter(group);
  } else if (operation == "pipeline") {
    benchmark_pipeline(group);
  } else if (operation == "bandwidth") {
    benchmark_bandwidth_scaling(group);
  } else {
    // Default: run all
    print_header("FULL BENCHMARK SUITE");
    
    benchmark_all_reduce(group);
    benchmark_all_gather(group);
    benchmark_reduce_scatter(group);
    benchmark_pipeline(group);
    benchmark_bandwidth_scaling(group);
  }
  
  print_header("BENCHMARK COMPLETE");
  
  return 0;
}
