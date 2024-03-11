// Copyright © 2023-2024 Apple Inc.

#include <dlfcn.h>
#include <filesystem>
#include <list>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/compiled_preamble.h"
#include "mlx/device.h"
#include "mlx/graph_utils.h"

namespace mlx::core {

// GPU compile is always available if the GPU is available and since we are in
// this file CPU compile is also available.
namespace detail {
bool compile_available_for_device(const Device& device) {
  return true;
}
} // namespace detail

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name);
}

// Return a pointer to a compiled function
void* compile(
    const std::string& kernel_name,
    const std::string& source_code = "") {
  struct DLib {
    DLib(const std::string& libname) {
      lib = dlopen(libname.c_str(), RTLD_NOW);
      if (!lib) {
        std::ostringstream msg;
        msg << "Could not load C++ shared library " << dlerror();
        throw std::runtime_error(msg.str());
      }
    }

    ~DLib() {
      dlclose(lib);
    }
    void* lib;
  };
  // Statics to cache compiled libraries and functions
  static std::list<DLib> libs;
  static std::unordered_map<std::string, void*> kernels;
  if (auto it = kernels.find(kernel_name); it != kernels.end()) {
    return it->second;
  }
  if (source_code.empty()) {
    return nullptr;
  }

  std::ostringstream shared_lib_name;
  shared_lib_name << "lib" << kernel_name << ".so";
  auto shared_lib_path = get_temp_file(shared_lib_name.str());
  bool lib_exists = false;
  {
    std::ifstream f(shared_lib_path.c_str());
    lib_exists = f.good();
  }

  if (!lib_exists) {
    // Open source file and write source code to it
    std::ostringstream source_file_name;
    source_file_name << kernel_name << ".cpp";
    auto source_file_path = get_temp_file(source_file_name.str());

    std::ofstream source_file(source_file_path);
    source_file << source_code;
    source_file.close();

    std::ostringstream build_command;
    build_command << "g++ -std=c++17 -O2 -Wall -fPIC -shared "
                  << source_file_path << " -o " << shared_lib_path;
    std::string build_command_str = build_command.str();
    auto return_code = system(build_command_str.c_str());
    if (return_code) {
      std::ostringstream msg;
      msg << "[Compile::eval_cpu] Failed to compile function " << kernel_name
          << " with error code " << return_code << "." << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  // load library
  libs.emplace_back(shared_lib_path);

  // Load function
  void* fun = dlsym(libs.back().lib, kernel_name.c_str());
  if (!fun) {
    std::ostringstream msg;
    msg << "[Compile::eval_cpu] Failed to load compiled function "
        << kernel_name << std::endl
        << dlerror();
    throw std::runtime_error(msg.str());
  }
  kernels.insert({kernel_name, fun});
  return fun;
}

inline void build_kernel(
    std::ostream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous,
    int ndim) {
  // All outputs should have the exact same shape and will be row contiguous
  auto output_shape = outputs[0].shape();
  auto output_strides = outputs[0].strides();

  // Constants are scalars that are captured by value and cannot change
  auto is_constant = [&constant_ids](const array& x) {
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  NodeNamer namer;

  // Start the kernel
  os << "void " << kernel_name << "(void** args) {" << std::endl;

  // Add the input arguments
  int cnt = 0;
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    // Skip constants from the input list
    if (is_constant(x)) {
      continue;
    }

    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << xname << " = (" << tstr << "*)args[" << cnt++
       << "];" << std::endl;
    // Scalars and contiguous need no strides
    if (!is_scalar(x) && !contiguous) {
      os << "  const size_t* " << xname << "_strides = (size_t*)args[" << cnt++
         << "];" << std::endl;
    }
  }

  // Add the output arguments
  for (auto& x : outputs) {
    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << namer.get_name(x) << " = (" << tstr
       << "*)args[" << cnt++ << "];" << std::endl;
  }
  // Add output strides and shape to extract the indices.
  if (!contiguous) {
    os << "  const int* shape = (int*)args[" << cnt++ << "];" << std::endl;
  } else {
    os << "  const size_t size = (size_t)args[" << cnt++ << "];" << std::endl;
  }

  if (contiguous) {
    os << "  for (size_t i = 0; i < size; ++i) {" << std::endl;
  } else {
    for (int d = 0; d < ndim; ++d) {
      os << "  for (int i" << d << " = 0; i" << d << " < shape[" << d
         << "]; ++i" << d << ") {" << std::endl;
    }
  }

  // Read the inputs in tmps
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    if (is_constant(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = ";
      print_constant(os, x);
      os << ";" << std::endl;
    } else if (is_scalar(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[0];" << std::endl;
    } else if (contiguous) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[i];" << std::endl;
    } else {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = *"
         << xname << ";" << std::endl;
    }
  }

  // Actually write the computation
  for (auto& x : tape) {
    os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
       << " = ";
    if (is_static_cast(x.primitive())) {
      os << "static_cast<" << get_type_string(x.dtype()) << ">(tmp_"
         << namer.get_name(x.inputs()[0]) << ");" << std::endl;
    } else {
      x.primitive().print(os);
      os << "()(";
      for (int i = 0; i < x.inputs().size() - 1; i++) {
        os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
      }
      os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
    }
  }

  // Write the outputs from tmps
  for (auto& x : outputs) {
    if (contiguous) {
      os << "  " << namer.get_name(x) << "[i] = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    } else {
      os << "  *" << namer.get_name(x) << "++ = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    }
  }

  // Close loops
  if (contiguous) {
    os << "  }" << std::endl;
  } else {
    for (int d = ndim - 1; d >= 0; --d) {
      // Update pointers
      for (auto& x : inputs) {
        if (is_constant(x) || is_scalar(x)) {
          continue;
        }
        auto& xname = namer.get_name(x);
        os << "  " << xname << " += " << xname << "_strides[" << d << "];"
           << std::endl;
        if (d < ndim - 1) {
          os << "  " << xname << " -= " << xname << "_strides[" << d + 1 << "]"
             << " * shape[" << d + 1 << "];" << std::endl;
        }
      }
      os << "  }" << std::endl;
    }
  }

  // Finish the kernel
  os << "}" << std::endl;
}

void Compiled::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (kernel_lib_.empty()) {
    kernel_lib_ = build_lib_name(inputs_, outputs_, tape_, constant_ids_);
  }

  // Figure out which kernel we are using
  auto& shape = outputs[0].shape();
  bool contiguous = true;
  {
    bool all_contig = true;
    bool all_row_contig = true;
    bool all_col_contig = true;
    int non_scalar_inputs = 0;
    for (auto& x : inputs) {
      if (is_scalar(x)) {
        continue;
      }
      non_scalar_inputs++;
      bool shape_eq = x.shape() == shape;
      all_contig &= (x.flags().contiguous && shape_eq);
      all_row_contig &= (x.flags().row_contiguous && shape_eq);
      all_col_contig &= (x.flags().col_contiguous && shape_eq);
    }
    if (non_scalar_inputs > 1 && !all_row_contig && !all_col_contig) {
      contiguous = false;
    } else if (non_scalar_inputs == 1 && !all_contig) {
      contiguous = false;
    }
  }

  // Handle all broadcasting and collect function input arguments
  std::vector<void*> args;
  std::vector<std::vector<size_t>> strides;
  for (int i = 0; i < inputs.size(); i++) {
    // Skip constants.
    if (constant_ids_.find(inputs_[i].id()) != constant_ids_.end()) {
      continue;
    }
    auto& x = inputs[i];
    args.push_back((void*)x.data<void>());

    if (contiguous || is_scalar(x)) {
      continue;
    }

    // Broadcast the input to the output shape.
    std::vector<size_t> xstrides;
    int j = 0;
    for (; j < shape.size() - x.ndim(); j++) {
      if (shape[j] == 1) {
        xstrides.push_back(outputs[0].strides()[j]);
      } else {
        xstrides.push_back(0);
      }
    }
    for (int i = 0; i < x.ndim(); i++, j++) {
      if (x.shape(i) == 1) {
        if (shape[j] == 1) {
          xstrides.push_back(outputs[0].strides()[j]);
        } else {
          xstrides.push_back(0);
        }
      } else {
        xstrides.push_back(x.strides()[i]);
      }
    }
    strides.push_back(std::move(xstrides));
    args.push_back(strides.back().data());
  }

  // Get the kernel name from the lib
  int ndim = shape.size();
  auto kernel_name = kernel_lib_ + (contiguous ? "_contiguous" : "_strided_");
  if (!contiguous) {
    kernel_name += std::to_string(shape.size());
  }

  // Get the function
  auto fn_ptr = compile(kernel_name);

  // If it doesn't exist, compile it
  if (fn_ptr == nullptr) {
    std::ostringstream kernel;
    kernel << get_kernel_preamble() << std::endl;
    kernel << "extern \"C\"  {" << std::endl;
    build_kernel(
        kernel,
        kernel_name,
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        contiguous,
        ndim);
    // Close extern "C"
    kernel << "}" << std::endl;

    // Compile and get function pointer
    fn_ptr = compile(kernel_name, kernel.str());
  }

  // Allocate space for the outputs possibly with input donation
  if (contiguous) {
    int o = 0;
    std::vector<size_t> strides;
    size_t data_size;
    array::Flags flags;
    for (int i = 0; i < inputs.size() && o < outputs.size(); ++i) {
      auto& in = inputs[i];
      // Conditions for donation
      // - Contiguous
      // - Donatable
      // - Correct size
      // - Not a constant
      if (in.flags().contiguous && !is_scalar(in) && in.is_donatable() &&
          constant_ids_.find(inputs_[i].id()) == constant_ids_.end()) {
        outputs[o++].copy_shared_buffer(in);
      }
      // Get representative input flags to properly set non-donated outputs
      if (strides.empty() && in.size() == outputs[0].size()) {
        strides = in.strides();
        flags = in.flags();
        data_size = in.data_size();
      }
    }
    for (; o < outputs.size(); ++o) {
      outputs[o].set_data(
          allocator::malloc_or_wait(data_size * outputs[o].itemsize()),
          data_size,
          strides,
          flags);
    }
  } else {
    int o = 0;
    for (int i = 0; i < inputs.size() && o < outputs.size(); ++i) {
      auto& in = inputs[i];
      // Conditions for donation
      // - Row contiguous
      // - Donatable
      // - Correct size
      // - Not a constant
      if (in.flags().row_contiguous && in.nbytes() == outputs[o].nbytes() &&
          in.is_donatable() &&
          constant_ids_.find(inputs_[i].id()) == constant_ids_.end()) {
        outputs[o].copy_shared_buffer(
            in, outputs[o].strides(), in.flags(), in.data_size());
        o++;
      }
    }
    for (; o < outputs.size(); ++o) {
      outputs[o].set_data(allocator::malloc_or_wait(outputs[o].nbytes()));
    }
  }

  for (auto& x : outputs) {
    args.push_back(x.data<void>());
  }
  if (!contiguous) {
    args.push_back((void*)outputs[0].shape().data());
  } else {
    args.push_back((void*)outputs[0].data_size());
  }
  auto fun = (void (*)(void**))fn_ptr;
  fun(args.data());
}

} // namespace mlx::core
