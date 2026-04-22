// Copyright © 2025 Apple Inc.
//
// File Broadcast with JACCL
//
// This example demonstrates distributed file transfer using JACCL's all_sum
// operation to broadcast a file from any rank to all other machines.
//
// The algorithm:
// 1. The sender rank reads the file into memory
// 2. All other ranks allocate zero-filled buffers of the same size
// 3. Use all_sum to broadcast: sender has data, others have zeros
// 4. After all_sum, all ranks have the file data
// 5. All ranks write the file to disk
//
// For large files, the transfer is chunked to manage memory efficiently.
//
// Usage:
//   Set environment variables (see README.md), then run:
//
//     ./jaccl_file_broadcast -f <file> [-s <sender_rank>] [-o <output_dir>]
//
//   Or with mlx.launch:
//
//     mlx.launch --hostfile hosts.json ./jaccl_file_broadcast -f myfile.bin
//
// Example output (4 ranks, sender rank 2):
//   Rank 0 of 4: Received 10485760 bytes from rank 2 (982.5 MB/s)
//   Rank 1 of 4: Received 10485760 bytes from rank 2 (985.2 MB/s)
//   Rank 2 of 4: Sent 10485760 bytes (980.1 MB/s)
//   Rank 3 of 4: Received 10485760 bytes from rank 2 (978.9 MB/s)

#include <jaccl/jaccl.h>
#include <jaccl/types.h>

#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  -f <file>       File to broadcast (required)\n"
      << "  -s <rank>       Sender rank (default: 0)\n"
      << "  -o <dir>        Output directory (default: current dir)\n"
      << "  -c <bytes>      Chunk size in bytes (default: 67108864 = 64MB)\n"
      << "  -v              Verbose output\n"
      << "  -h              Show this help\n";
}

static bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static std::int64_t file_size(const std::string& path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return -1;
  }
  return static_cast<std::int64_t>(buffer.st_size);
}

static bool create_directory(const std::string& path) {
  if (path.empty() || path == ".") {
    return true;
  }
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

static std::string basename(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

struct BroadcastStats {
  std::int64_t total_bytes;
  std::int64_t chunks_sent;
  std::int64_t chunks_received;
  double total_time_ms;
  int sender_rank;
};

int main(int argc, char** argv) {
  std::string input_file;
  std::string output_dir = ".";
  int sender_rank = 0;
  std::int64_t chunk_size = 67108864;
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else if (arg == "-f" && i + 1 < argc) {
      input_file = argv[++i];
    } else if (arg == "-s" && i + 1 < argc) {
      sender_rank = std::atoi(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "-c" && i + 1 < argc) {
      chunk_size = std::atoll(argv[++i]);
    } else if (arg == "-v" || arg == "--verbose") {
      verbose = true;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  if (input_file.empty()) {
    std::cerr << "Error: Input file is required (-f <file>)\n";
    usage(argv[0]);
    return 1;
  }

  auto group = jaccl::init();
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  int rank = group->rank();
  int nranks = group->size();

  if (sender_rank < 0 || sender_rank >= nranks) {
    std::cerr << "Error: Sender rank " << sender_rank << " is out of range [0, "
              << nranks << ")\n";
    return 1;
  }

  std::int64_t total_file_size = 0;
  if (rank == sender_rank) {
    if (!file_exists(input_file)) {
      std::cerr << "Error: File not found: " << input_file << "\n";
      return 1;
    }
    total_file_size = file_size(input_file);
    if (total_file_size < 0) {
      std::cerr << "Error: Cannot read file size: " << input_file << "\n";
      return 1;
    }
  }

  group->all_sum(
      &total_file_size, &total_file_size, sizeof(int64_t), jaccl::Int64);

  if (!create_directory(output_dir)) {
    std::cerr << "Error: Cannot create output directory: " << output_dir
              << "\n";
    return 1;
  }

  std::string output_file = output_dir == "."
      ? basename(input_file)
      : output_dir + "/" + basename(input_file);

  if (verbose) {
    std::printf(
        "Rank %d of %d: Broadcasting '%s' (%ld bytes) from rank %d\n",
        rank,
        nranks,
        input_file.c_str(),
        static_cast<long>(total_file_size),
        sender_rank);
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  std::int64_t num_chunks = (total_file_size + chunk_size - 1) / chunk_size;
  if (num_chunks == 0) {
    num_chunks = 1;
  }

  const int num_buffers = 4;
  std::vector<std::vector<std::uint8_t>> buffers(
      num_buffers, std::vector<std::uint8_t>(chunk_size, 0));

  std::ifstream infile;
  std::ofstream outfile;

  if (rank == sender_rank) {
    infile.open(input_file, std::ios::binary);
    if (!infile.good()) {
      std::cerr << "Error: Cannot open input file: " << input_file << "\n";
      return 1;
    }
  }

  outfile.open(output_file, std::ios::binary);
  if (!outfile.good()) {
    std::cerr << "Error: Cannot open output file: " << output_file << "\n";
    return 1;
  }

  std::atomic<std::int64_t> next_read_chunk{0};
  std::atomic<std::int64_t> next_comm_chunk{0};
  std::atomic<std::int64_t> next_write_chunk{0};
  std::atomic<bool> read_done{false};
  std::atomic<bool> comm_done{false};

  std::vector<std::atomic<bool>> buffer_ready(num_buffers);
  std::vector<std::atomic<bool>> buffer_written(num_buffers);
  for (int i = 0; i < num_buffers; i++) {
    buffer_ready[i] = false;
    buffer_written[i] = false;
  }

  std::vector<std::int64_t> chunk_sizes(num_chunks);
  for (std::int64_t i = 0; i < num_chunks; i++) {
    chunk_sizes[i] = std::min(chunk_size, total_file_size - i * chunk_size);
  }

  std::thread reader_thread;
  if (rank == sender_rank) {
    reader_thread = std::thread([&]() {
      while (true) {
        std::int64_t chunk_idx = next_read_chunk.fetch_add(1);
        if (chunk_idx >= num_chunks) {
          break;
        }
        std::int64_t offset = chunk_idx * chunk_size;
        std::int64_t this_chunk_size = chunk_sizes[chunk_idx];
        int buffer_idx = chunk_idx % num_buffers;

        infile.seekg(offset, std::ios::beg);
        infile.read(
            reinterpret_cast<char*>(buffers[buffer_idx].data()),
            this_chunk_size);

        std::fill(
            buffers[buffer_idx].begin() + this_chunk_size,
            buffers[buffer_idx].end(),
            0);

        buffer_ready[buffer_idx] = true;
      }
      read_done = true;
    });
  } else {
    read_done = true;
  }

  std::thread writer_thread([&]() {
    while (true) {
      std::int64_t chunk_idx = next_write_chunk.load();
      if (chunk_idx >= num_chunks && comm_done) {
        break;
      }
      if (chunk_idx >= num_chunks) {
        std::this_thread::yield();
        continue;
      }

      int buffer_idx = chunk_idx % num_buffers;
      if (!buffer_written[buffer_idx]) {
        std::this_thread::yield();
        continue;
      }

      std::int64_t this_chunk_size = chunk_sizes[chunk_idx];
      outfile.write(
          reinterpret_cast<const char*>(buffers[buffer_idx].data()),
          this_chunk_size);

      buffer_written[buffer_idx] = false;
      next_write_chunk.fetch_add(1);
    }
  });

  for (std::int64_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    std::int64_t this_chunk_size = chunk_sizes[chunk_idx];
    int buffer_idx = chunk_idx % num_buffers;

    if (rank == sender_rank) {
      while (!buffer_ready[buffer_idx] && !read_done) {
        std::this_thread::yield();
      }
    }

    std::fill(
        buffers[buffer_idx].begin() + this_chunk_size,
        buffers[buffer_idx].end(),
        0);

    group->all_sum(
        buffers[buffer_idx].data(),
        buffers[buffer_idx].data(),
        this_chunk_size,
        jaccl::UInt8);

    buffer_written[buffer_idx] = true;
    next_comm_chunk.fetch_add(1);

    if (verbose) {
      double progress = 100.0 * (chunk_idx + 1) / num_chunks;
      std::printf(
          "Rank %d: Progress %.1f%% (%ld/%ld chunks)\n",
          rank,
          progress,
          static_cast<long>(chunk_idx + 1),
          static_cast<long>(num_chunks));
    }
  }

  comm_done = true;

  if (reader_thread.joinable()) {
    reader_thread.join();
  }
  writer_thread.join();

  infile.close();
  outfile.close();

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();
  double elapsed_sec = elapsed_ms / 1000.0;
  double bandwidth_mbps = (total_file_size / (1024.0 * 1024.0)) / elapsed_sec;

  if (rank == sender_rank) {
    std::printf(
        "Rank %d of %d: Sent %ld bytes from '%s' (%.1f MB/s)\n",
        rank,
        nranks,
        static_cast<long>(total_file_size),
        input_file.c_str(),
        bandwidth_mbps);
  } else {
    std::printf(
        "Rank %d of %d: Received %ld bytes from rank %d to '%s' (%.1f MB/s)\n",
        rank,
        nranks,
        static_cast<long>(total_file_size),
        sender_rank,
        output_file.c_str(),
        bandwidth_mbps);
  }

  if (verbose) {
    std::printf(
        "Rank %d: Total time: %.2f ms, Chunks: %ld, Chunk size: %ld bytes\n",
        rank,
        elapsed_ms,
        static_cast<long>(num_chunks),
        static_cast<long>(chunk_size));
  }

  return 0;
}
