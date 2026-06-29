// Copyright © 2026 Apple Inc.

#include "mlx/backend/common/utils.h"

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

namespace mlx::core::cu {

// Defined in dirs.cpp to avoid invalidating compile cache.
const char* cuda_bin_dir();
const char* cudnn_bin_dir();

namespace fs = std::filesystem;

inline fs::path relative_to_current_binary(const char* relative) {
  return fs::absolute(current_binary_dir() / relative);
}

inline fs::path cublas_dir() {
  return cuda_bin_dir() ? fs::path(cuda_bin_dir())
                        : relative_to_current_binary("../nvidia/cublas/bin");
}

fs::path load_nvrtc() {
  fs::path nvrtc_dir = cuda_bin_dir()
      ? fs::path(cuda_bin_dir())
      : relative_to_current_binary("../nvidia/cuda_nvrtc/bin");
  // Internally nvrtc loads some libs dynamically, add to search dirs.
  ::AddDllDirectory(nvrtc_dir.c_str());
  return nvrtc_dir;
}

fs::path load_cudnn() {
  fs::path cudnn_dir = cudnn_bin_dir()
      ? fs::path(cudnn_bin_dir())
      : relative_to_current_binary("../nvidia/cudnn/bin");
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

// Called by system when failed to locate a lazy-loaded DLL.
FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  HMODULE mod = NULL;
  if (dliNotify == dliNotePreLoadLibrary) {
    std::string dll = pdli->szDll;
    if (dll.starts_with("cudnn")) {
      static auto cudnn_dir = load_cudnn();
      mod = ::LoadLibraryW((cudnn_dir / dll).c_str());
    } else if (dll.starts_with("cublas")) {
      mod = ::LoadLibraryW((cublas_dir() / dll).c_str());
    } else if (dll.starts_with("nvrtc")) {
      static auto nvrtc_dir = load_nvrtc();
      mod = ::LoadLibraryW((nvrtc_dir / dll).c_str());
    }
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace mlx::core::cu

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::cu::delayload_helper;
