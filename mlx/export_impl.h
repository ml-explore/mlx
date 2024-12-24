// Copyright © 2024 Apple Inc.

#include "mlx/io/load.h"

#pragma once

namespace mlx::core {

struct FunctionTable;

struct FunctionExporter {
  void operator()(const std::initializer_list<array>& args) {
    this->operator()(Args(args));
  }
  void operator()(const Args& args);
  void operator()(const Kwargs& kwargs);
  void operator()(const Args& args, const Kwargs& kwargs);

  void close();

  FunctionExporter(const FunctionExporter&) = delete;
  FunctionExporter& operator=(const FunctionExporter&) = delete;
  FunctionExporter(FunctionExporter&& other) = default;

 private:
  friend FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Args&)>&,
      bool shapeless);

  friend FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Kwargs&)>&,
      bool shapeless);

  friend FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Args&, const Kwargs&)>&,
      bool shapeless);

  FunctionExporter(
      const std::string& file,
      std::function<std::vector<array>(const Args&, const Kwargs&)> fun,
      bool shapeless);
  io::FileWriter os;
  std::function<std::vector<array>(const Args&, const Kwargs& kwargs)> fun;
  void export_function(const Args& args, const Kwargs& kwargs);
  std::set<std::uintptr_t> constants;
  int count{0};
  bool closed{false};
  std::shared_ptr<FunctionTable> ftable;
};

struct ImportedFunction {
  std::vector<array> operator()(
      const std::initializer_list<array>& args) const {
    return this->operator()(Args(args));
  }
  std::vector<array> operator()(const Args& args) const;
  std::vector<array> operator()(const Kwargs& kwargs) const;
  std::vector<array> operator()(const Args& args, const Kwargs& kwargs) const;

 private:
  ImportedFunction(const std::string& file);
  friend ImportedFunction import_function(const std::string&);
  ImportedFunction();

  std::shared_ptr<FunctionTable> ftable;
};

} // namespace mlx::core
