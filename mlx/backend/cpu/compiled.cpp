// Copyright © 2023-2024 Apple Inc.

#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <shared_mutex>

#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cpu/compiled_preamble.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/jit_compiler.h"
#include "mlx/device.h"
#include "mlx/graph_utils.h"
#include "mlx/version.h"

namespace mlx::core {

struct CompilerCache {
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
  std::list<DLib> libs;
  std::unordered_map<std::string, void*> kernels;
  std::shared_mutex mtx;
};

static CompilerCache& cache() {
  static CompilerCache cache_;
  return cache_;
};

// GPU compile is always available if the GPU is available and since we are in
// this file CPU compile is also available.
namespace detail {
bool compile_available_for_device(const Device& device) {
  return true;
}

} // namespace detail

// Return a pointer to a compiled function
void* compile(
    const std::string& kernel_name,
    const std::function<std::string(void)>& source_builder) {
  {
    std::shared_lock lock(cache().mtx);
    if (auto it = cache().kernels.find(kernel_name);
        it != cache().kernels.end()) {
      return it->second;
    }
  }

  std::unique_lock lock(cache().mtx);
  if (auto it = cache().kernels.find(kernel_name);
      it != cache().kernels.end()) {
    return it->second;
  }
  std::string source_code = source_builder();
  std::string kernel_file_name;

  // Deal with long kernel names. Maximum length for filename on macOS is 255
  // characters, and on Windows the maximum length for whole path is 260. Clip
  // file name with a little extra room and append a 16 character hash.
#ifdef _WIN32
  constexpr int max_file_name_length = 140;
#else
  constexpr int max_file_name_length = 245;
#endif
  if (kernel_name.size() > max_file_name_length) {
    std::ostringstream file_name;
    file_name
        << std::string_view(kernel_name).substr(0, max_file_name_length - 16);
    auto file_id =
        std::hash<std::string>{}(kernel_name.substr(max_file_name_length - 16));
    file_name << "_" << std::hex << std::setw(16) << file_id << std::dec;
    kernel_file_name = file_name.str();
  } else {
    kernel_file_name = kernel_name;
  }

  auto output_dir =
      std::filesystem::temp_directory_path() / "mlx" / version() / "cpu";
  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }

  std::string shared_lib_name = "lib" + kernel_file_name + ".so";
  auto shared_lib_path = (output_dir / shared_lib_name).string();
  bool lib_exists = false;
  {
    std::ifstream f(shared_lib_path.c_str());
    lib_exists = f.good();
  }

  if (!lib_exists) {
    // Open source file and write source code to it
    std::string source_file_name = kernel_file_name + ".cpp";
    auto source_file_path = (output_dir / source_file_name).string();

    std::ofstream source_file(source_file_path);
    source_file << source_code;
    source_file.close();

    try {
      JitCompiler::exec(JitCompiler::build_command(
          output_dir, source_file_name, shared_lib_name));
    } catch (const std::exception& error) {
      throw std::runtime_error(fmt::format(
          "[Compile::eval_cpu] Failed to compile function {0}: {1}",
          kernel_name,
          error.what()));
    }
  }

  // load library
  cache().libs.emplace_back(shared_lib_path);

  // Load function
  void* fun = dlsym(cache().libs.back().lib, kernel_name.c_str());
  if (!fun) {
    std::ostringstream msg;
    msg << "[Compile::eval_cpu] Failed to load compiled function "
        << kernel_name << std::endl
        << dlerror();
    throw std::runtime_error(msg.str());
  }
  cache().kernels.insert({kernel_name, fun});
  return fun;
}

inline void build_kernel(
    std::ostream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    int ndim) {
  NodeNamer namer;

#ifdef _MSC_VER
  // Export the symbol
  os << "__declspec(dllexport) ";
#endif

  // Start the kernel
  os << "void " << kernel_name
     << "(int* shape, int64_t** strides, void** args) {" << std::endl;

  // Add the input arguments
  int cnt = 0;
  int strides_index = 1;
  for (size_t i = 0; i < inputs.size(); ++i) {
    // Skip constants from the input list
    if (is_constant(i)) {
      continue;
    }

    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);

    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << xname << " = (" << tstr << "*)args[" << cnt++
       << "];" << std::endl;
    // Scalars and contiguous need no strides
    if (!is_scalar(x) && !contiguous) {
      os << "  const int64_t* " << xname << "_strides = strides["
         << strides_index++ << "];" << std::endl;
    }
  }

  // Add the output arguments
  for (auto& x : outputs) {
    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << namer.get_name(x) << " = (" << tstr
       << "*)args[" << cnt++ << "];" << std::endl;
  }
  // Add output size
  if (contiguous) {
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
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);

    if (is_constant(i)) {
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
      os << x.primitive().name();
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
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        if (is_constant(i) || is_scalar(x)) {
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
  auto& encoder = cpu::get_command_encoder(stream());

  // Collapse contiguous dims to route to a faster kernel if possible. Also
  // handle all broadcasting.
  auto [contiguous, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Collect function input arguments.
  std::vector<void*> args;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    encoder.set_input_array(x);
    args.push_back((void*)x.data<void>());
  }

  // Get the kernel name from the lib
  int ndim = shape.size();
  auto kernel_name = kernel_lib_ + (contiguous ? "_contiguous" : "_strided_");
  if (!contiguous) {
    kernel_name += std::to_string(ndim);
  }

  // Get the function
  auto fn_ptr = compile(kernel_name, [&, contiguous = contiguous]() {
    std::ostringstream kernel;
    kernel << get_kernel_preamble() << std::endl;
    kernel << "extern \"C\"  {" << std::endl;
    build_kernel(
        kernel,
        kernel_name,
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        contiguous,
        ndim);
    // Close extern "C"
    kernel << "}" << std::endl;
    return kernel.str();
  });

  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);

  for (auto& x : outputs) {
    args.push_back(x.data<void>());
    encoder.set_output_array(x);
  }
  if (contiguous) {
    args.push_back((void*)outputs[0].data_size());
  }
  auto fun = reinterpret_cast<void (*)(int*, int64_t**, void**)>(fn_ptr);
  encoder.dispatch([fun,
                    args = std::move(args),
                    strides = std::move(strides),
                    shape = std::move(shape)]() mutable {
    SmallVector<int64_t*> strides_ptrs;
    for (auto& s : strides) {
      strides_ptrs.push_back(s.data());
    }
    fun(shape.data(), strides_ptrs.data(), args.data());
  });
}

} // namespace mlx::core
