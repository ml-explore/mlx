// Copyright © 2025 Apple Inc.

#pragma once

#include <istream>
#include <sstream>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx::core {

namespace io {

struct json {
  using json_array = std::vector<json>;
  using json_object = std::unordered_map<std::string, json>;

  json() : data(json_object()) {}

  template <typename T>
  json(T x) : data(x) {}

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

  std::string dump() {
    std::ostringstream os;
    os << *this;
    return os.str();
  }

  // Produces valid JSON
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

json parse_json(std::istream& s);

json parse_json(char* s, int length);

json parse_json(std::string& s);

} // namespace io

} // namespace mlx::core
