// Copyright © 2025 Apple Inc.
//
// Monte Carlo Pi Estimation with JACCL
//
// This example demonstrates distributed Monte Carlo simulation using JACCL
// to estimate the value of π. Each rank generates random points independently
// and uses all-reduce to aggregate the results across all machines.
//
// The algorithm:
// 1. Each rank generates N random points in the unit square [0,1] x [0,1]
// 2. Count how many fall inside the quarter circle (x² + y² ≤ 1)
// 3. Use all_sum to aggregate hits and total points across all ranks
// 4. π ≈ 4 × (hits / total)
//
// Usage:
//   Set environment variables (see README.md), then run:
//
//     ./jaccl_monte_carlo_pi [-n <points_per_rank>]
//
//   Or with mlx.launch:
//
//     mlx.launch --hostfile hosts.json ./jaccl_monte_carlo_pi -n 10000000
//
// Example output (4 ranks, 10M points each):
//   Rank 2 of 4
//   Local: 7854321 hits out of 10000000 points
//   Global: 31416789 hits out of 40000000 points
//   Estimated π = 3.141679 (error: 0.000086)

#include <jaccl/jaccl.h>
#include <jaccl/types.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  -n <points>  Points per rank (default: 1000000)\n"
            << "  -s <seed>    Random seed base (default: 42)\n"
            << "  -h           Show this help\n";
}

struct MonteCarloResult {
  int64_t hits;
  int64_t total;
};

MonteCarloResult estimate_pi_local(int64_t num_points, unsigned int seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  int64_t hits = 0;
  for (int64_t i = 0; i < num_points; i++) {
    double x = dist(rng);
    double y = dist(rng);
    if (x * x + y * y <= 1.0) {
      hits++;
    }
  }

  return {hits, num_points};
}

int main(int argc, char** argv) {
  int64_t points_per_rank = 1000000;
  unsigned int seed_base = 42;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else if (arg == "-n" && i + 1 < argc) {
      points_per_rank = std::atoll(argv[++i]);
    } else if (arg == "-s" && i + 1 < argc) {
      seed_base = static_cast<unsigned int>(std::atoi(argv[++i]));
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  auto group = jaccl::init();
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  int rank = group->rank();
  int nranks = group->size();

  std::printf("Rank %d of %d\n", rank, nranks);
  std::printf(
      "Generating %ld random points (seed: %u)...\n",
      static_cast<long>(points_per_rank),
      seed_base + static_cast<unsigned int>(rank));

  auto t0 = std::chrono::high_resolution_clock::now();

  MonteCarloResult local = estimate_pi_local(
      points_per_rank, seed_base + static_cast<unsigned int>(rank));

  auto t1 = std::chrono::high_resolution_clock::now();
  double local_time =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf(
      "Rank %d: %ld hits out of %ld points (%.2f ms)\n",
      rank,
      static_cast<long>(local.hits),
      static_cast<long>(local.total),
      local_time);

  MonteCarloResult global = {0, 0};

  group->all_sum(&local.hits, &global.hits, sizeof(int64_t), jaccl::Int64);
  group->all_sum(&local.total, &global.total, sizeof(int64_t), jaccl::Int64);

  if (rank == 0) {
    double pi_estimate = 4.0 * static_cast<double>(global.hits) /
        static_cast<double>(global.total);
    double error = std::abs(pi_estimate - M_PI);

    std::printf("\n=== Results ===\n");
    std::printf(
        "Global: %ld hits out of %ld points\n",
        static_cast<long>(global.hits),
        static_cast<long>(global.total));
    std::printf("Estimated π = %.10f\n", pi_estimate);
    std::printf("True π      = %.10f\n", M_PI);
    std::printf("Error       = %.10f (%.6f%%)\n", error, 100.0 * error / M_PI);

    double total_time =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\nPerformance:\n");
    std::printf("Total points: %ld\n", static_cast<long>(global.total));
    std::printf("Time: %.2f ms\n", total_time);
    std::printf(
        "Points/sec: %.0f\n",
        static_cast<double>(global.total) / (total_time / 1000.0));
  }

  return 0;
}
