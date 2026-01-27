// Copyright © 2026 Apple Inc.

#include "mlx/backend/common/utils.h"

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

namespace mlx::core {

namespace fs = std::filesystem;

inline fs::path relative_to_current_binary(const char* relative) {
  return fs::absolute(current_binary_dir() / relative);
}

inline fs::path cublas_bin_dir() {
#if defined(MLX_CUDA_BIN_DIR)
  return MLX_CUDA_BIN_DIR;
#else
  return relative_to_current_binary("../nvidia/cublas/bin");
#endif
}

fs::path load_nvrtc() {
#if defined(MLX_CUDA_BIN_DIR)
  fs::path nvrtc_bin_dir = MLX_CUDA_BIN_DIR;
#else
  fs::path nvrtc_bin_dir =
      relative_to_current_binary("../nvidia/cuda_nvrtc/bin");
#endif
  // Internally nvrtc loads some libs dynamically, add to search dirs.
  ::AddDllDirectory(nvrtc_bin_dir.c_str());
  return nvrtc_bin_dir;
}

fs::path load_cudnn() {
#if defined(MLX_CUDNN_BIN_DIR)
  fs::path cudnn_bin_dir = MLX_CUDNN_BIN_DIR;
#else
  fs::path cudnn_bin_dir = relative_to_current_binary("../nvidia/cudnn/bin");
#endif
  // Must load cudnn_graph64_9.dll before locating symbols, otherwise We would
  // get errors like "Invalid handle. Cannot load symbol cudnnCreate".
  for (const auto& dll : fs::directory_iterator(cudnn_bin_dir)) {
    if (dll.path().filename().string().starts_with("cudnn_graph") &&
        dll.path().extension() == ".dll") {
      ::LoadLibraryW(dll.path().c_str());
      break;
    }
  }
  // Internally cuDNN loads some libs dynamically, add to search dirs.
  load_nvrtc();
  ::AddDllDirectory(cudnn_bin_dir.c_str());
  ::AddDllDirectory(cublas_bin_dir().c_str());
  return cudnn_bin_dir;
}

// Called by system when failed to locate a lazy-loaded DLL.
FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  HMODULE mod = NULL;
  if (dliNotify == dliNotePreLoadLibrary) {
    std::string dll = pdli->szDll;
    if (dll.starts_with("cudnn")) {
      static auto cudnn_bin_dir = load_cudnn();
      mod = ::LoadLibraryW((cudnn_bin_dir / dll).c_str());
    } else if (dll.starts_with("cublas")) {
      mod = ::LoadLibraryW((cublas_bin_dir() / dll).c_str());
    } else if (dll.starts_with("nvrtc")) {
      static auto nvrtc_bin_dir = load_nvrtc();
      mod = ::LoadLibraryW((nvrtc_bin_dir / dll).c_str());
    }
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace mlx::core

extern "C" const PfnDliHook __pfnDliNotifyHook2 = mlx::core::delayload_helper;
