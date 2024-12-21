// Copyright © 2024 Apple Inc.

#include "mlx/io/load.h"

#pragma once

namespace mlx::core {

namespace {}

struct FunctionExporter {
  void operator()(const std::initializer_list<array>& args) {
    this->operator()(Args(args));
  }
  void operator()(const Args& args);
  void operator()(const Kwargs& kwargs);
  void operator()(const Args& args, const Kwargs& kwargs);

 private:
  struct FunctionInfo {
    std::vector<Shape> shapes;
    std::vector<Dtype> types;
    std::vector<std::string> kwarg_keys;
  };

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
      const std::string& path,
      std::function<std::vector<array>(const Args&, const Kwargs&)> fun,
      bool shapeless);

  io::FileWriter os;
  std::function<std::vector<array>(const Args&, const Kwargs& kwargs)> fun;
  bool shapeless;
  void export_function(const Args& args, const Kwargs& kwargs);
  std::set<std::uintptr_t> constants;
  int count{0};
  // std::vector<ExportedFunctionInfo> functions;
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
  ImportedFunction(const std::string& path);
  struct Function {
    std::vector<std::string> kwarg_keys;
    std::vector<array> trace_inputs;
    std::vector<array> trace_outputs;
    std::vector<array> tape;
  };

  bool shapeless;
  friend ImportedFunction import_function(const std::string&);
  ImportedFunction();

  // Index functions by number of inputs as a heuristic for reasonably
  // fast lookup
  std::unordered_map<int, std::vector<Function>> functions;
};

} // namespace mlx::core
