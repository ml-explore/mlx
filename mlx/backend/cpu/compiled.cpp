// Copyright © 2023-2026 Apple Inc.

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
#include "mlx/backend/cpu/threading/common.h"
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
  // Leak - see Scheduler singleton comment in scheduler.cpp.
  // DLib destructors call dlclose() which unmaps JIT .so files;
  // StreamThreads may still be executing that code at exit.
  static CompilerCache* cache_ = new CompilerCache;
  return *cache_;
};

// Check if JIT compilation is available for the given device.
// On Windows, this probes for a usable C++ compiler (MSVC or clang-cl).
namespace detail {
bool compile_available_for_device(const Device& device) {
  return JitCompiler::available();
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

#ifdef _WIN32
  std::string shared_lib_name = kernel_file_name + ".dll";
#else
  std::string shared_lib_name = "lib" + kernel_file_name + ".so";
#endif
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
      JitCompiler::exec(
          JitCompiler::build_command(
              output_dir, source_file_name, shared_lib_name));
    } catch (const std::exception& error) {
      throw std::runtime_error(
          fmt::format(
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
    auto out_tstr = get_type_string(outputs[0].dtype());

    // SIMD main loop (processes max_size elements per iteration)
    os << "  constexpr int _S = simd::max_size<" << out_tstr << ">;"
       << std::endl;
    os << "  size_t _vi = 0;" << std::endl;
    os << "  if constexpr (_S > 1) {" << std::endl;
    os << "  for (; _vi + _S <= size; _vi += _S) {" << std::endl;

    // SIMD loads
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      auto& xname = namer.get_name(x);
      auto tstr = get_type_string(x.dtype());

      if (is_constant(i)) {
        os << "  auto tmp_" << xname << " = Simd<" << tstr << ", _S>((" << tstr
           << ")";
        print_constant(os, x);
        os << ");" << std::endl;
      } else if (is_scalar(x)) {
        os << "  auto tmp_" << xname << " = Simd<" << tstr << ", _S>(" << xname
           << "[0]);" << std::endl;
      } else {
        os << "  auto tmp_" << xname << " = simd::load<" << tstr << ", _S>("
           << xname << " + _vi);" << std::endl;
      }
    }

    // SIMD computation
    for (auto& x : tape) {
      os << "  auto tmp_" << namer.get_name(x) << " = ";
      if (is_static_cast(x.primitive())) {
        auto tstr = get_type_string(x.dtype());
        os << "Simd<" << tstr << ", _S>(tmp_" << namer.get_name(x.inputs()[0])
           << ");" << std::endl;
      } else {
        os << x.primitive().name();
        os << "()(";
        for (int i = 0; i < x.inputs().size() - 1; i++) {
          os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
        }
        os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
      }
    }

    // SIMD stores
    for (auto& x : outputs) {
      os << "  simd::store(" << namer.get_name(x) << " + _vi, tmp_"
         << namer.get_name(x) << ");" << std::endl;
    }

    os << "  }" << std::endl; // close SIMD for
    os << "  }" << std::endl; // close if constexpr

    // Scalar tail loop
    os << "  for (size_t i = _vi; i < size; ++i) {" << std::endl;

    // Scalar reads (contiguous)
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
      } else {
        os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
           << xname << "[i];" << std::endl;
      }
    }

    // Scalar computation (contiguous tail)
    for (auto& x : tape) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
         << " = ";
      if (is_static_cast(x.primitive())) {
        os << "static_cast<" << get_type_string(x.dtype()) << ">(tmp_"
           << namer.get_name(x.inputs()[0]) << ");" << std::endl;
      } else {
        os << x.primitive().name() << "()(";
        for (int i = 0; i < x.inputs().size() - 1; i++) {
          os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
        }
        os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
      }
    }

    // Scalar writes (contiguous tail)
    for (auto& x : outputs) {
      os << "  " << namer.get_name(x) << "[i] = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    }

    os << "  }" << std::endl; // close scalar tail

  } else {
    // ===== STRIDED PATH WITH SIMD =====
    auto out_tstr = get_type_string(outputs[0].dtype());
    int inner_d = ndim - 1;

    os << "  constexpr int _S = simd::max_size<" << out_tstr << ">;"
       << std::endl;

    // Open outer loops (d=0 to d=ndim-2)
    for (int d = 0; d < ndim - 1; ++d) {
      os << "  for (int i" << d << " = 0; i" << d << " < shape[" << d
         << "]; ++i" << d << ") {" << std::endl;
    }

    // ---- SIMD inner loop ----
    os << "  int i" << inner_d << " = 0;" << std::endl;
    os << "  if constexpr (_S > 1) {" << std::endl;
    os << "  for (; i" << inner_d << " + _S <= shape[" << inner_d << "]; i"
       << inner_d << " += _S) {" << std::endl;

    // SIMD loads (with stride check for gather)
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      auto& xname = namer.get_name(x);
      auto tstr = get_type_string(x.dtype());

      if (is_constant(i)) {
        os << "  auto tmp_" << xname << " = Simd<" << tstr << ", _S>((" << tstr
           << ")";
        print_constant(os, x);
        os << ");" << std::endl;
      } else if (is_scalar(x)) {
        os << "  auto tmp_" << xname << " = Simd<" << tstr << ", _S>(" << xname
           << "[0]);" << std::endl;
      } else {
        // Runtime stride check: contiguous inner dim -> simd::load, else gather
        os << "  Simd<" << tstr << ", _S> tmp_" << xname << ";" << std::endl;
        os << "  if (" << xname << "_strides[" << inner_d << "] == 1) {"
           << std::endl;
        os << "    tmp_" << xname << " = simd::load<" << tstr << ", _S>("
           << xname << " + i" << inner_d << ");" << std::endl;
        os << "  } else {" << std::endl;
        os << "    " << tstr << " _gather_" << xname << "[_S];" << std::endl;
        os << "    for (int _k = 0; _k < _S; _k++)" << std::endl;
        os << "      _gather_" << xname << "[_k] = " << xname << "[(i"
           << inner_d << " + _k) * " << xname << "_strides[" << inner_d << "]];"
           << std::endl;
        os << "    tmp_" << xname << " = simd::load<" << tstr
           << ", _S>(_gather_" << xname << ");" << std::endl;
        os << "  }" << std::endl;
      }
    }

    // SIMD computation (strided)
    for (auto& x : tape) {
      os << "  auto tmp_" << namer.get_name(x) << " = ";
      if (is_static_cast(x.primitive())) {
        auto tstr = get_type_string(x.dtype());
        os << "Simd<" << tstr << ", _S>(tmp_" << namer.get_name(x.inputs()[0])
           << ");" << std::endl;
      } else {
        os << x.primitive().name() << "()(";
        for (int i = 0; i < x.inputs().size() - 1; i++) {
          os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
        }
        os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
      }
    }

    // SIMD stores (output is always contiguous)
    for (auto& x : outputs) {
      os << "  simd::store(" << namer.get_name(x) << ", tmp_"
         << namer.get_name(x) << ");" << std::endl;
      os << "  " << namer.get_name(x) << " += _S;" << std::endl;
    }

    os << "  }" << std::endl; // close SIMD for
    os << "  }" << std::endl; // close if constexpr

    // ---- Scalar tail (strided) ----
    os << "  for (; i" << inner_d << " < shape[" << inner_d << "]; ++i"
       << inner_d << ") {" << std::endl;

    // Scalar reads (indexed by inner dim)
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      auto& xname = namer.get_name(x);
      auto tstr = get_type_string(x.dtype());

      if (is_constant(i)) {
        os << "  " << tstr << " tmp_" << xname << " = ";
        print_constant(os, x);
        os << ";" << std::endl;
      } else if (is_scalar(x)) {
        os << "  " << tstr << " tmp_" << xname << " = " << xname << "[0];"
           << std::endl;
      } else {
        os << "  " << tstr << " tmp_" << xname << " = " << xname << "[i"
           << inner_d << " * " << xname << "_strides[" << inner_d << "]];"
           << std::endl;
      }
    }

    // Scalar computation (strided tail)
    for (auto& x : tape) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
         << " = ";
      if (is_static_cast(x.primitive())) {
        os << "static_cast<" << get_type_string(x.dtype()) << ">(tmp_"
           << namer.get_name(x.inputs()[0]) << ");" << std::endl;
      } else {
        os << x.primitive().name() << "()(";
        for (int i = 0; i < x.inputs().size() - 1; i++) {
          os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
        }
        os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
      }
    }

    // Scalar writes (output contiguous, pointer auto-advances)
    for (auto& x : outputs) {
      os << "  *" << namer.get_name(x) << "++ = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    }

    os << "  }" << std::endl; // close scalar tail

    // Close outer loops (d=ndim-2 down to 0)
    // Inner dim uses indexed access (no pointer accumulation), so outer loops
    // only undo accumulation from dimensions > inner_d (i.e., d+1 < inner_d).
    for (int d = ndim - 2; d >= 0; --d) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        if (is_constant(i) || is_scalar(x)) {
          continue;
        }
        auto& xname = namer.get_name(x);
        os << "  " << xname << " += " << xname << "_strides[" << d << "];"
           << std::endl;
        if (d < ndim - 2) {
          os << "  " << xname << " -= " << xname << "_strides[" << d + 1
             << "] * shape[" << d + 1 << "];" << std::endl;
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
  // Metadata for parallel dispatch
  SmallVector<size_t, 8> arg_itemsize;
  SmallVector<bool, 8> arg_is_scalar;
  size_t num_data_args = 0; // input + output args (excludes trailing size arg)

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i))
      continue;
    arg_itemsize.push_back(inputs[i].itemsize());
    arg_is_scalar.push_back(is_scalar(inputs_[i]));
    num_data_args++;
  }
  for (auto& x : outputs) {
    arg_itemsize.push_back(x.itemsize());
    arg_is_scalar.push_back(false);
    num_data_args++;
  }

  if (contiguous) {
    args.push_back((void*)outputs[0].data_size());
  }

  auto fun = reinterpret_cast<void (*)(int*, int64_t**, void**)>(fn_ptr);

  if (contiguous) {
    size_t total_size = outputs[0].data_size();
    auto& pool = cpu::ThreadPool::instance();
    int n_threads = cpu::effective_threads(total_size, pool.max_threads());

    if (n_threads > 1) {
      // Multi-threaded: parallel_for splits work across threads
      encoder.dispatch([fun,
                        args,
                        arg_itemsize,
                        arg_is_scalar,
                        num_data_args,
                        total_size,
                        n_threads]() mutable {
        auto& pool = cpu::ThreadPool::instance();
        pool.parallel_for(n_threads, [&](int tid, int nth) {
          size_t chunk = (total_size + nth - 1) / nth;
          size_t start = chunk * tid;
          size_t end = std::min(start + chunk, total_size);
          if (start >= end)
            return;

          SmallVector<void*, 8> thread_args(args.size());
          for (size_t j = 0; j < num_data_args; ++j) {
            if (arg_is_scalar[j]) {
              thread_args[j] = args[j];
            } else {
              thread_args[j] =
                  static_cast<char*>(args[j]) + start * arg_itemsize[j];
            }
          }
          thread_args[num_data_args] = (void*)(end - start);
          fun(nullptr, nullptr, thread_args.data());
        });
      });
    } else {
      // Single-threaded contiguous
      encoder.dispatch([fun, args = std::move(args)]() mutable {
        fun(nullptr, nullptr, args.data());
      });
    }
  } else {
    // Strided path with threading
    size_t total_size = outputs[0].data_size();
    auto& pool = cpu::ThreadPool::instance();
    int n_threads = cpu::effective_threads(total_size, pool.max_threads());
    if (!shape.empty()) {
      n_threads = std::min(n_threads, (int)shape[0]);
    }

    if (n_threads > 1 && !shape.empty() && shape[0] > 1) {
      // Compute outermost stride for each arg (for pointer offsetting)
      SmallVector<int64_t, 8> arg_outer_stride(num_data_args);
      {
        int si = 1; // strides[0] is output, inputs start at 1
        size_t arg_j = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
          if (is_constant_(i))
            continue;
          if (arg_is_scalar[arg_j]) {
            arg_outer_stride[arg_j] = 0;
          } else {
            arg_outer_stride[arg_j] = strides[si][0];
            si++;
          }
          arg_j++;
        }
        // Output args use strides[0] (output strides)
        for (size_t i = arg_j; i < num_data_args; ++i) {
          arg_outer_stride[i] = strides[0][0];
        }
      }

      encoder.dispatch([fun,
                        args,
                        strides = std::move(strides),
                        shape = std::move(shape),
                        arg_outer_stride,
                        arg_itemsize,
                        num_data_args,
                        n_threads]() mutable {
        SmallVector<int64_t*> strides_ptrs;
        for (auto& s : strides) {
          strides_ptrs.push_back(s.data());
        }
        auto& pool = cpu::ThreadPool::instance();
        pool.parallel_for(n_threads, [&](int tid, int nth) {
          int chunk = (shape[0] + nth - 1) / nth;
          int start = chunk * tid;
          int end = std::min(start + chunk, shape[0]);
          if (start >= end)
            return;

          SmallVector<void*, 8> thread_args(args.begin(), args.end());
          for (size_t j = 0; j < num_data_args; ++j) {
            if (arg_outer_stride[j] != 0) {
              thread_args[j] = static_cast<char*>(args[j]) +
                  start * arg_outer_stride[j] *
                      static_cast<int64_t>(arg_itemsize[j]);
            }
          }

          auto thread_shape = shape;
          thread_shape[0] = end - start;

          fun(thread_shape.data(), strides_ptrs.data(), thread_args.data());
        });
      });
    } else {
      // Single-threaded strided
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
  }
}

} // namespace mlx::core
