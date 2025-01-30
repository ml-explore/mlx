// Copyright © 2023 Apple Inc.

#pragma once

#include <memory>
#include <sstream>
#include <variant>

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
  static ThreadPool thread_pool_;
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

struct json;

using json_array = std::vector<json>;
using json_object = std::unordered_map<std::string, json>;

struct json {
  json() : data(json_object()) {}

  template <typename T>
  json(T x) : data(x) {}

  // TODO: templatize this
  json(std::vector<size_t> x) {
    json_array arr;
    for (long val : x) {
      arr.push_back(val);
    }
    this->data = arr;
  }

  json(std::vector<int> x) {
    json_array arr;
    for (long val : x) {
      arr.push_back(val);
    }
    this->data = arr;
  }

  json(std::vector<long> x) {
    json_array arr;
    for (long val : x) {
      arr.push_back(val);
    }
    this->data = arr;
  }

  json(std::vector<float> x) {
    json_array arr;
    for (double val : x) {
      arr.push_back(val);
    }
    this->data = arr;
  }

  json(std::vector<double> x) {
    json_array arr;
    for (double val : x) {
      arr.push_back(val);
    }
    this->data = arr;
  }

  operator std::string() const {
    return std::get<std::string>(this->data);
  }

  operator double() const {
    return std::get<double>(this->data);
  }

  operator long() const {
    return std::get<long>(this->data);
  }

  operator size_t() const {
    return std::get<long>(this->data);
  }

  operator int() const {
    return std::get<long>(this->data);
  }

  operator float() const {
    return std::get<double>(this->data);
  }

  operator bool() const {
    return std::get<bool>(this->data);
  }

  template <typename T>
  operator std::vector<T>() const {
    json_array arr = std::get<json_array>(this->data);
    std::vector<T> out;
    for (T val : arr) {
      out.push_back(val);
    }
    return out;
  }

  json& operator[](const std::string& key) {
    auto& m = std::get<json_object>(this->data);
    if (m.find(key) == m.end()) {
      m.insert({key, nullptr});
    }
    return m.at(key);
  }

  const json& operator[](const std::string& key) const {
    return std::get<json_object>(this->data).at(key);
  }

  json& operator[](const char* key) {
    std::cout << "KEY " << key << std::endl;
    auto& m = std::get<json_object>(this->data);
    if (m.find(key) == m.end()) {
      m.insert({key, nullptr});
    }
    return m.at(key);
  }

  const json& operator[](const char* key) const {
    return std::get<json_object>(this->data).at(key);
  }

  json& operator[](const int index) {
    return std::get<json_array>(this->data).at(index);
  }

  const json& operator[](const int index) const {
    return std::get<json_array>(this->data).at(index);
  }

  auto begin() {
    return std::get<json_array>(this->data).begin();
  }

  auto begin() const {
    return std::get<json_array>(this->data).begin();
  }

  auto end() {
    return std::get<json_array>(this->data).end();
  }

  auto end() const {
    return std::get<json_array>(this->data).end();
  }

  json_object& items() {
    return std::get<json_object>(this->data);
  }

  const json_object& items() const {
    return std::get<json_object>(this->data);
  }

  template <typename T>
  bool is() {
    return std::holds_alternative<T>(this->data);
  }

  template <typename T>
  bool is() const {
    return std::holds_alternative<T>(this->data);
  }

  // Produces valid json
  friend std::ostream& operator<<(std::ostream& os, const json& obj);

 private:
  std::variant<
      std::nullptr_t,
      bool,
      long,
      double,
      std::string,
      json_array,
      json_object>
      data;
};

json parse_json(std::istream& s, bool allow_extra = false);

json parse_json(const std::string& s, bool allow_extra = false);

} // namespace io
} // namespace mlx::core
