// Copyright © 2026 Apple Inc.

#include "mlx/backend/common/utils.h"

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

#include <cuda_runtime.h>

namespace mlx::core::cu {

// Defined in dirs.cpp to avoid invalidating compile cache.
const char* cuda_bin_dir();
const char* cudnn_bin_dir();

namespace fs = std::filesystem;

#if CUDART_VERSION == 0
#error "CUDART_VERSION not defined"
#elif CUDART_VERSION < 13000
#define BIN_DIR(name) "../nvidia/" name "/bin"
#elif CUDART_VERSION < 14000
#define BIN_DIR(name) "../nvidia/cu13/bin/x86_64"
#else
#define BIN_DIR(name) "../nvidia/cu14/bin/x86_64"
#endif

inline fs::path relative_to_current_binary(const char* relative) {
  return fs::absolute(current_binary_dir() / relative);
}

inline fs::path cublas_dir() {
  if (const char* dir = cuda_bin_dir()) {
    return fs::path(dir);
  }
  return relative_to_current_binary(BIN_DIR("cublas"));
}

fs::path load_nvrtc() {
  fs::path nvrtc_dir;
  if (const char* dir = cuda_bin_dir()) {
    nvrtc_dir = fs::path(dir);
  } else {
    nvrtc_dir = relative_to_current_binary(BIN_DIR("cuda_nvrtc"));
  }
  // Internally nvrtc loads some libs dynamically, add to search dirs.
  ::AddDllDirectory(nvrtc_dir.c_str());
#if CUDART_VERSION >= 13000
  // The search dirs are not used when nvrtc loads nvrtc-builtins, so load it
  // beforehand, otherwise compiling kernels fails with "failed to open
  // nvrtc-builtins64_*.dll".
  for (const auto& dll : fs::directory_iterator(nvrtc_dir)) {
    if (dll.path().filename().string().starts_with("nvrtc-builtins64") &&
        dll.path().extension() == ".dll") {
      ::LoadLibraryW(dll.path().c_str());
      break;
    }
  }
#endif
  return nvrtc_dir;
}

fs::path load_cudnn() {
  fs::path cudnn_dir;
  if (const char* dir = cudnn_bin_dir()) {
    cudnn_dir = fs::path(dir);
  } else {
    cudnn_dir = relative_to_current_binary("../nvidia/cudnn/bin");
  }
  // Must load cudnn_graph64_9.dll before locating symbols, otherwise We would
  // get errors like "Invalid handle. Cannot load symbol cudnnCreate".
  for (const auto& dll : fs::directory_iterator(cudnn_dir)) {
    if (dll.path().filename().string().starts_with("cudnn_graph") &&
        dll.path().extension() == ".dll") {
      ::LoadLibraryW(dll.path().c_str());
      break;
    }
  }
  // Internally cuDNN loads some libs dynamically, add to search dirs.
  load_nvrtc();
  ::AddDllDirectory(cudnn_dir.c_str());
  ::AddDllDirectory(cublas_dir().c_str());
  return cudnn_dir;
}

fs::path load_cusolver() {
  fs::path cusolver_dir;
  if (const char* dir = cuda_bin_dir()) {
    cusolver_dir = dir;
  } else {
    cusolver_dir = relative_to_current_binary(BIN_DIR("cusolver"));
    ::AddDllDirectory(cublas_dir().c_str());
    ::AddDllDirectory(relative_to_current_binary(BIN_DIR("cusparse")).c_str());
    ::AddDllDirectory(relative_to_current_binary(BIN_DIR("nvjitlink")).c_str());
  }
  ::AddDllDirectory(cusolver_dir.c_str());
  return cusolver_dir;
}

// Called by system before loading a lazy-loaded DLL.
FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  HMODULE mod = NULL;
  try {
    if (dliNotify == dliNotePreLoadLibrary) {
      std::string dll = pdli->szDll;
      if (dll.starts_with("cudnn")) {
        static auto cudnn_dir = load_cudnn();
        mod = ::LoadLibraryW((cudnn_dir / dll).c_str());
      } else if (dll.starts_with("cublas")) {
        mod = ::LoadLibraryW((cublas_dir() / dll).c_str());
      } else if (dll.starts_with("cusolver")) {
        static auto cusolver_dir = load_cusolver();
        mod = ::LoadLibraryW((cusolver_dir / dll).c_str());
      } else if (dll.starts_with("nvrtc")) {
        static auto nvrtc_dir = load_nvrtc();
        mod = ::LoadLibraryW((nvrtc_dir / dll).c_str());
      } else if (const char* dir = cuda_bin_dir()) {
        mod = ::LoadLibraryW((fs::path(dir) / dll).c_str());
      }
    }
  } catch (...) {
    // Return control to system on error.
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace mlx::core::cu

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::cu::delayload_helper;
