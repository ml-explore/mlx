// Copyright © 2023 Apple Inc.

#pragma once

#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "mlx/threadpool.h"

// Strictly we need to operate on files in binary mode (to avoid \r getting
// automatically inserted), but every modern system except for Windows no
// longer differentiates between binary and text files and for them define
// the flag as no-op.
#ifndef O_BINARY
#define O_BINARY 0
#endif

namespace mlx::core {

namespace io {

ThreadPool& thread_pool();

class Reader {
 public:
  virtual bool is_open() const = 0;
  virtual bool good() const = 0;
  virtual size_t tell() = 0; // tellp is non-const in iostream
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) = 0;
  virtual void read(char* data, size_t n) = 0;
  virtual void read(char* data, size_t n, size_t offset) = 0;
  virtual std::string label() const = 0;
  virtual ~Reader() = default;
};

class Writer {
 public:
  virtual bool is_open() const = 0;
  virtual bool good() const = 0;
  virtual size_t tell() = 0;
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) = 0;
  virtual void write(const char* data, size_t n) = 0;
  virtual std::string label() const = 0;
  virtual void open() {}

  virtual ~Writer() = default;
};

class ParallelFileReader : public Reader {
 public:
  explicit ParallelFileReader(std::string file_path)
      : fd_(open(file_path.c_str(), O_RDONLY | O_BINARY)),
        label_(std::move(file_path)) {}

  ~ParallelFileReader() override {
    close(fd_);
  }

  bool is_open() const override {
    return fd_ > 0;
  }

  bool good() const override {
    return is_open();
  }

  size_t tell() override {
#ifdef _WIN32
    return _lseeki64(fd_, 0, SEEK_CUR);
#else
    return lseek(fd_, 0, SEEK_CUR);
#endif
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    int origin;
    if (way == std::ios_base::beg) {
      origin = SEEK_SET;
    } else if (way == std::ios_base::end) {
      origin = SEEK_END;
    } else {
      origin = SEEK_CUR;
    }
#ifdef _WIN32
    _lseeki64(fd_, off, origin);
#else
    lseek(fd_, off, origin);
#endif
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void read(char* data, size_t n) override;

  void read(char* data, size_t n, size_t offset) override;

  std::string label() const override {
    return "file " + label_;
  }

 private:
  // Reads larger than this are split in batches and read in parallel.
  static constexpr size_t batch_size_ = 1 << 25;

  // The pool that reads the batches, held from the first batched read until
  // the reader is destroyed. It cannot be io::thread_pool(), the tasks that
  // run there wait for these batches and would deadlock in the same pool.
  ThreadPool& thread_pool();

  int fd_;
  std::string label_;
  std::once_flag pool_once_;
  std::shared_ptr<ThreadPool> pool_;
};

class FileWriter : public Writer {
 public:
  explicit FileWriter() {}
  explicit FileWriter(std::string file_path)
      : file_path_(std::move(file_path)) {}

  FileWriter(const FileWriter&) = delete;
  FileWriter& operator=(const FileWriter&) = delete;
  FileWriter(FileWriter&& other)
      : fd_(std::exchange(other.fd_, -1)),
        file_path_(std::move(other.file_path_)) {
    other.file_path_.clear();
  }

  ~FileWriter() override {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  // Kept separate from construction so lazy inputs can be evaluated first,
  // they may still read from the file.
  void open() override {
    if (fd_ < 0 && !file_path_.empty()) {
      fd_ = ::open(
          file_path_.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_BINARY, 0644);
    }
  }

  bool is_open() const override {
    return fd_ >= 0;
  }

  bool good() const override {
    return is_open();
  }

  size_t tell() override {
    check_open();
#ifdef _WIN32
    return _lseeki64(fd_, 0, SEEK_CUR);
#else
    return lseek(fd_, 0, SEEK_CUR);
#endif
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    check_open();
    int origin;
    if (way == std::ios_base::beg) {
      origin = SEEK_SET;
    } else if (way == std::ios_base::end) {
      origin = SEEK_END;
    } else {
      origin = SEEK_CUR;
    }
#ifdef _WIN32
    _lseeki64(fd_, off, origin);
#else
    lseek(fd_, off, origin);
#endif
  }

  void write(const char* data, size_t n) override {
    check_open();
    while (n != 0) {
      auto m = ::write(fd_, data, std::min(n, static_cast<size_t>(INT32_MAX)));
      if (m <= 0) {
        std::ostringstream msg;
        msg << "[write] Unable to write " << n << " bytes to file.";
        throw std::runtime_error(msg.str());
      }
      data += m;
      n -= m;
    }
  }

  std::string label() const override {
    return "file " + file_path_;
  }

 private:
  void check_open() const {
    if (!is_open()) {
      throw std::runtime_error("[write] File " + file_path_ + " is not open.");
    }
  }

  int fd_{-1};
  std::string file_path_;
};

} // namespace io
} // namespace mlx::core
