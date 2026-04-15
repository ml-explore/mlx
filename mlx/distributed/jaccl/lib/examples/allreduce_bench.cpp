// Copyright © 2025 Apple Inc.
//
// JACCL All-Reduce Benchmark
//
// Measures bandwidth and latency of all_sum across a sweep of message sizes,
// similar in spirit to the NCCL all-reduce benchmark (nccl-tests).
//
// Usage:
//   Set the environment variables described in jaccl.h, then run:
//
//     ./jaccl_allreduce_bench [-w <warmup_iters>] [-n <iters>]
//                             [-b <min_bytes>] [-e <max_bytes>]
//                             [-f <step_factor>] [-d <datatype>]
//                             [-c] [-h]
//
//   Or use the MLX launcher:
//
//     mlx.launch --hostfile hosts.json ./jaccl_allreduce_bench
//
//   The arguments are:
//
//   -w  Warmup iterations per message size  (default: 5)
//   -n  Timed iterations per message size   (default: 20)
//   -b  Minimum message size in bytes       (default: 1K)
//   -e  Maximum message size in bytes       (default: 256M)
//   -f  Multiplicative step factor          (default: 2)
//   -d  Datatype: float32, float16, bfloat16 (default: float32)
//   -c  Check correctness                   (default: off)
//   -h  Print this help message

#include <jaccl/jaccl.h>
#include <jaccl/types.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  -w <warmup>    Warmup iterations          (default: 5)\n"
      << "  -n <iters>     Timed iterations            (default: 20)\n"
      << "  -b <min_bytes> Minimum message size        (default: 1K)\n"
      << "  -e <max_bytes> Maximum message size        (default: 256M)\n"
      << "  -f <factor>    Multiplicative step factor  (default: 2)\n"
      << "  -d <dtype>     float32|float16|bfloat16    (default: float32)\n"
      << "  -c             Check correctness\n"
      << "  -h             Show this help\n";
}

static size_t parse_size(const char* s) {
  char* end = nullptr;
  double val = std::strtod(s, &end);
  if (end && (*end == 'K' || *end == 'k'))
    val *= 1024;
  else if (end && (*end == 'M' || *end == 'm'))
    val *= 1024 * 1024;
  else if (end && (*end == 'G' || *end == 'g'))
    val *= 1024 * 1024 * 1024;
  return static_cast<size_t>(val);
}

static std::string fmt_bytes(size_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int idx = 0;
  double val = static_cast<double>(bytes);
  while (val >= 1024.0 && idx < 4) {
    val /= 1024.0;
    idx++;
  }
  char buf[32];
  if (val == static_cast<int>(val))
    std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(val), units[idx]);
  else
    std::snprintf(buf, sizeof(buf), "%.2f %s", val, units[idx]);
  return buf;
}

// Conversion from algorithm bandwidth to bus bandwidth for a ring reduce.
static double bus_factor(int nranks) {
  return 2.0 * (nranks - 1) / static_cast<double>(nranks);
}

int main(int argc, char** argv) {
  int warmup_iters = 5;
  int timed_iters = 20;
  size_t min_bytes = 1024;
  size_t max_bytes = 256 * 1024 * 1024;
  int step_factor = 2;
  std::string dtype_str = "float32";
  bool check = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else if (arg == "-w" && i + 1 < argc) {
      warmup_iters = std::atoi(argv[++i]);
    } else if (arg == "-n" && i + 1 < argc) {
      timed_iters = std::atoi(argv[++i]);
    } else if (arg == "-b" && i + 1 < argc) {
      min_bytes = parse_size(argv[++i]);
    } else if (arg == "-e" && i + 1 < argc) {
      max_bytes = parse_size(argv[++i]);
    } else if (arg == "-f" && i + 1 < argc) {
      step_factor = std::atoi(argv[++i]);
    } else if (arg == "-d" && i + 1 < argc) {
      dtype_str = argv[++i];
    } else if (arg == "-c") {
      check = true;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  jaccl::Dtype dtype;
  size_t elem_size;
  if (dtype_str == "float32") {
    dtype = jaccl::Float32;
    elem_size = 4;
  } else if (dtype_str == "float16") {
    dtype = jaccl::Float16;
    elem_size = 2;
  } else if (dtype_str == "bfloat16") {
    dtype = jaccl::BFloat16;
    elem_size = 2;
  } else {
    std::cerr << "Unsupported dtype: " << dtype_str << "\n";
    return 1;
  }

  auto group = jaccl::init(/* strict= */ true);

  int rank = group->rank();
  int nranks = group->size();

  if (rank == 0) {
    std::cout << "# JACCL All-Reduce Benchmark\n"
              << "# Ranks:   " << nranks << "\n"
              << "# Dtype:   " << dtype_str << "\n"
              << "# Warmup:  " << warmup_iters << " iters\n"
              << "# Timed:   " << timed_iters << " iters\n"
              << "# Sizes:   " << fmt_bytes(min_bytes) << " .. "
              << fmt_bytes(max_bytes) << " (x" << step_factor << ")\n"
              << "#\n";

    // Table header (NCCL-style)
    std::cout << std::left << std::setw(14) << "#    size" << std::right
              << std::setw(12) << "count" << std::setw(12) << "type"
              << std::setw(14) << "time (us)" << std::setw(14) << "algo BW"
              << std::setw(14) << "bus BW";
    if (check)
      std::cout << std::setw(10) << "check";
    std::cout << "\n";
    std::cout << std::left << std::setw(14) << "#   (bytes)" << std::right
              << std::setw(12) << "(elems)" << std::setw(12) << ""
              << std::setw(14) << "" << std::setw(14) << "(GB/s)"
              << std::setw(14) << "(GB/s)";
    if (check)
      std::cout << std::setw(10) << "";
    std::cout << "\n";
  }

  size_t max_elems = max_bytes / elem_size;
  max_bytes = max_elems * elem_size;

  std::vector<char> sendbuf(max_bytes);

  // Fill send buffer with a simple deterministic pattern (rank + 1) casted to
  // the target type, so correctness checks are straightforward: after all_sum
  // every element should equal sum_{r=0}^{nranks-1} (r + 1) =
  // nranks*(nranks+1)/2.
  auto fill_buffer = [&](char* buf, size_t n_bytes) {
    float val = static_cast<float>(rank + 1);
    if (dtype == jaccl::Float32) {
      auto* p = reinterpret_cast<float*>(buf);
      size_t n = n_bytes / sizeof(float);
      for (size_t i = 0; i < n; i++) {
        p[i] = val;
      }
    } else if (dtype == jaccl::Float16) {
      // Write via the library's float16_t
      auto* p = reinterpret_cast<jaccl::float16_t*>(buf);
      size_t n = n_bytes / sizeof(jaccl::float16_t);
      for (size_t i = 0; i < n; i++) {
        p[i] = jaccl::float16_t(val);
      }
    } else if (dtype == jaccl::BFloat16) {
      auto* p = reinterpret_cast<jaccl::bfloat16_t*>(buf);
      size_t n = n_bytes / sizeof(jaccl::bfloat16_t);
      for (size_t i = 0; i < n; i++) {
        p[i] = jaccl::bfloat16_t(val);
      }
    }
  };

  auto check_buffer = [&](const char* buf, size_t n_bytes) -> bool {
    float expected = static_cast<float>(nranks) * (nranks + 1) / 2.0f;
    float tol = (dtype == jaccl::Float32) ? 1e-5f : 1e-1f;
    size_t n = n_bytes / elem_size;

    for (size_t i = 0; i < n; i++) {
      float val;
      if (dtype == jaccl::Float32) {
        val = reinterpret_cast<const float*>(buf)[i];
      } else if (dtype == jaccl::Float16) {
        val = static_cast<float>(
            reinterpret_cast<const jaccl::float16_t*>(buf)[i]);
      } else {
        val = static_cast<float>(
            reinterpret_cast<const jaccl::bfloat16_t*>(buf)[i]);
      }
      if (std::abs(val - expected) > tol) {
        return false;
      }
    }
    return true;
  };

  double bf = bus_factor(nranks);

  for (size_t nbytes = min_bytes; nbytes <= max_bytes;
       nbytes *= static_cast<size_t>(step_factor)) {
    // Round down to element boundary
    size_t n = std::max((nbytes / elem_size) * elem_size, elem_size);
    size_t count = n / elem_size;

    fill_buffer(sendbuf.data(), n);

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
      group->all_sum(sendbuf.data(), sendbuf.data(), n, dtype);
    }

    // Timed iterations
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_iters; i++) {
      group->all_sum(sendbuf.data(), sendbuf.data(), n, dtype);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = elapsed_us / timed_iters;

    // Bandwidth in GB/s
    double algo_bw = (static_cast<double>(n) / avg_us) / 1e3;
    double bus_bw = algo_bw * bf;

    // Correctness check
    std::string check_result;
    if (check) {
      fill_buffer(sendbuf.data(), n);
      group->all_sum(sendbuf.data(), sendbuf.data(), n, dtype);
      check_result = check_buffer(sendbuf.data(), n) ? "OK" : "FAIL";
    }

    if (rank == 0) {
      std::cout << std::left << std::setw(14) << n << std::right
                << std::setw(12) << count << std::setw(12) << dtype_str
                << std::setw(14) << std::fixed << std::setprecision(1) << avg_us
                << std::setw(14) << std::fixed << std::setprecision(2)
                << algo_bw << std::setw(14) << std::fixed
                << std::setprecision(2) << bus_bw;
      if (check)
        std::cout << std::setw(10) << check_result;
      std::cout << "\n";
    }
  }

  if (rank == 0) {
    std::cout << "# Done.\n";
  }

  return 0;
}
