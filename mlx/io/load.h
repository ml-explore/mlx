// Copyright © 2023 Apple Inc.

#pragma once

#include <fstream>
#include <istream>
#include <memory>

namespace mlx::core {

namespace io {

class Reader {
 public:
  virtual bool is_open() const = 0;
  virtual bool good() const = 0;
  virtual size_t tell() = 0; // tellp is non-const in iostream
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) = 0;
  virtual void read(char* data, size_t n) = 0;
  virtual std::string label() const = 0;
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
};

class FileReader : public Reader {
 public:
  explicit FileReader(std::ifstream is)
      : is_(std::move(is)), label_("stream") {}
  explicit FileReader(std::string file_path)
      : is_(std::ifstream(file_path, std::ios::binary)),
        label_(std::move(file_path)) {}

  bool is_open() const override {
    return is_.is_open();
  }

  bool good() const override {
    return is_.good();
  }

  size_t tell() override {
    return is_.tellg();
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    is_.seekg(off, way);
  }

  void read(char* data, size_t n) override {
    is_.read(data, n);
  }

  std::string label() const override {
    return "file " + label_;
  }

 private:
  std::ifstream is_;
  std::string label_;
};

class FileWriter : public Writer {
 public:
  explicit FileWriter(std::ofstream os)
      : os_(std::move(os)), label_("stream") {}
  explicit FileWriter(std::string file_path)
      : os_(std::ofstream(file_path, std::ios::binary)),
        label_(std::move(file_path)) {}

  bool is_open() const override {
    return os_.is_open();
  }

  bool good() const override {
    return os_.good();
  }

  size_t tell() override {
    return os_.tellp();
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    os_.seekp(off, way);
  }

  void write(const char* data, size_t n) override {
    os_.write(data, n);
  }

  std::string label() const override {
    return "file " + label_;
  }

 private:
  std::ofstream os_;
  std::string label_;
};

} // namespace io
} // namespace mlx::core
