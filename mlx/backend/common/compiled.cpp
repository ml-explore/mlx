// Copyright © 2023-2024 Apple Inc.

#include <dlfcn.h>
#include <filesystem>
#include <iostream> // TODO

#include "mlx/primitives.h"

namespace mlx::core {

// #include <fstream>
// #include <stdlib.h>

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name);
}

struct DynamicLibrary {
  DynamicLibrary(const std::string& libname) {
    lib = dlopen(libname.c_str(), RTLD_LAZY);
    if (!lib) {
      std::ostringstream msg;
      msg << "Could not load C++ shared library " << dlerror();
      throw std::runtime_error(msg.str());
    }
  }

  ~DynamicLibrary() {
    dlclose(lib);
  }
  void* lib;
};

// Return a pointer to a compiled function
void* compile(
    const std::string& kernel_name,
    const std::string& source_code = "") {
  // Statics to cache compiled libraries and functions
  static std::vector<DynamicLibrary> libs;
  static std::unordered_map<std::string, void*> kernels;
  if (auto it = kernels.find(kernel_name); it != kernels.end()) {
    return it->second;
  }
  if (source_code.empty()) {
    return nullptr;
  }

  // TODO we can probably reuse the .so if they are still around
  std::ostringstream source_file_name;
  source_file_name << kernel_name << ".cpp";
  auto source_file_path = get_temp_file(source_file_name.str());
  std::cout << "SOURCE FILE: " << source_file_path << std::endl;

  std::ostringstream shared_lib_name;
  shared_lib_name << "lib" << kernel_name << ".so";
  auto shared_lib_path = get_temp_file(shared_lib_name.str());

  // Open source file and write source code to it
  std::ofstream source_file(source_file_path);
  source_file
      << "#include <iostream>\nextern \"C\" void fun() { std::cout << \"Hello world\" << std::endl; }\n";
  source_file.close();

  {
    std::ostringstream build_command;
    build_command << "g++ -std=c++17 -O2 -Wall -shared " << source_file_path
                  << " -o " << shared_lib_path;
    std::string build_command_str = build_command.str();
    system(build_command_str.c_str());
  }

  // load library
  libs.emplace_back(shared_lib_path);

  // Load function
  void* fun = dlsym(libs.back().lib, "fun");

  // Example call:
  void (*fun2)(void) = (void (*)(void))(fun);
  fun2();

  return fun;
}

void Compiled::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (kernel_lib_.empty()) {
    kernel_lib_ = "test_cpu_compile"; // build_lib_name(inputs_, outputs_,
                                      // tape_, constant_ids_);
  }

  // Check if it has already been compiled
  auto fn = compile(kernel_lib_);
  if (fn == nullptr) {
    // Build the code and compile it
    compile(kernel_lib_, "sourcecode");
  }

  // Allocate space for the outputs
  for (auto& out : outputs) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }
}

} // namespace mlx::core
