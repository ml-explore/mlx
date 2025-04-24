// Copyright © 2023 Apple Inc.

#pragma once

#include <memory>
#include <sstream>

#include <fcntl.h>
#ifdef _MSC_VER
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
    return lseek(fd_, 0, SEEK_CUR);
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    if (way == std::ios_base::beg) {
      lseek(fd_, off, 0);
    } else {
      lseek(fd_, off, SEEK_CUR);
    }
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void read(char* data, size_t n) override;

  void read(char* data, size_t n, size_t offset) override;

  std::string label() const override {
    return "file " + label_;
  }

 private:
  static constexpr size_t batch_size_ = 1 << 25;
  static ThreadPool& thread_pool();
  int fd_;
  std::string label_;
};

class FileWriter : public Writer {
 public:
  explicit FileWriter(std::string file_path)
      : fd_(open(
            file_path.c_str(),
            O_CREAT | O_WRONLY | O_TRUNC | O_BINARY,
            0644)),
        label_(std::move(file_path)) {}

  FileWriter(const FileWriter&) = delete;
  FileWriter& operator=(const FileWriter&) = delete;
  FileWriter(FileWriter&& other) {
    std::swap(fd_, other.fd_);
  }

  ~FileWriter() override {
    if (fd_ != 0) {
      close(fd_);
    }
  }

  bool is_open() const override {
    return fd_ >= 0;
  }

  bool good() const override {
    return is_open();
  }

  size_t tell() override {
    return lseek(fd_, 0, SEEK_CUR);
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    if (way == std::ios_base::beg) {
      lseek(fd_, off, 0);
    } else {
      lseek(fd_, off, SEEK_CUR);
    }
  }

  void write(const char* data, size_t n) override {
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
    return "file " + label_;
  }

 private:
  int fd_{0};
  std::string label_;
};

} // namespace io
} // namespace mlx::core
