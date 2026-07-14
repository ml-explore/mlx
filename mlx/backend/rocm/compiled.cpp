// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <climits>
#include <sstream>

namespace mlx::core {

namespace rocm {

struct FusedKernelBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& outputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build(const char* name, bool contiguous) {
    NodeNamer namer;

    // Function parameters.
    std::vector<std::string> params;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        continue;
      }
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      params.push_back(
          std::string("const ") + dtype_to_hip_type(x.dtype()) + "* " + xname);
      if (!is_scalar(x) && !contiguous) {
        params.push_back(
            std::string("const hip::std::array<int64_t, NDIM> ") + xname +
            "_strides");
      }
    }
    for (const auto& x : outputs) {
      params.push_back(
          std::string(dtype_to_hip_type(x.dtype())) + "* " + namer.get_name(x));
    }
    if (!contiguous) {
      params.push_back("const hip::std::array<int32_t, NDIM> shape");
    }
    params.push_back("IdxT size");

    // Build function signature.
    if (contiguous) {
      os += "template <typename IdxT = uint32_t, int work_per_thread = 1>\n";
    } else {
      os +=
          "template <int NDIM, typename IdxT = uint32_t, int work_per_thread = 1>\n";
    }
    os += "__global__ void " + kernel_name + name + "(\n";
    for (size_t i = 0; i < params.size(); ++i) {
      os += "    ";
      os += params[i];
      if (i != params.size() - 1) {
        os += ",\n";
      }
    }
    os += ") {\n";

    // Index. For non contiguous kernels we create a separate index
    // variable per variable otherwise everyone uses `index`.
    os +=
        "  IdxT index = (blockIdx.x * blockDim.x + threadIdx.x) * work_per_thread;\n"
        "  if (index >= size) {\n"
        "    return;\n"
        "  }\n";
    if (!contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "  IdxT " + xname + "_idx = 0;\n";
      }
      os += "  {\n";
      os += "    IdxT loc = index;\n";
      os +=
          "    #pragma unroll\n"
          "    for (int i = NDIM - 1; i >= 0; i--) {\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "      " + xname + "_idx += (loc \% shape[i]) * IdxT(" + xname +
            "_strides[i]);\n";
      }
      os +=
          "      loc /= shape[i];\n"
          "    }\n"
          "  }\n";
    }

    // Vectorized contiguous reads (CUDA parity): pack work_per_thread elements
    // into AlignedVector via wide load when the pointer is aligned.
    if (contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        const std::string& xname = namer.get_name(x);
        std::string type = dtype_to_hip_type(x.dtype());
        os += "  auto vec_" + xname + " = load_vector<work_per_thread, " +
            type + ">(" + xname + " + index, 0, size - index, " + type +
            "(0));\n";
      }
    }

    // Output vectors (both contiguous and strided paths store via store_vector
    // so the last partial vector is bounded by size - index).
    for (const auto& x : outputs) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_hip_type(x.dtype());
      os += "  AlignedVector<" + type + ", work_per_thread> vec_" + xname +
          ";\n";
    }

    // Work loop
    if (!contiguous) {
      os +=
          "\n"
          "  for (int i = 0; i < work_per_thread && index < size; i++) {\n";
    } else {
      os +=
          "\n"
          "  #pragma unroll\n"
          "  for (int i = 0; i < work_per_thread; i++) {\n";
    }

    // Read inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_hip_type(x.dtype());
      std::string value;
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        value = std::string("static_cast<") + type + ">(" + ss.str() + ")";
      } else if (is_scalar(x)) {
        value = xname + "[0]";
      } else if (contiguous) {
        value = "vec_" + xname + "[i]";
      } else {
        value = xname + "[" + xname + "_idx]";
      }
      os +=
          std::string("    ") + type + " tmp_" + xname + " = " + value + ";\n";
    }

    // Write tape.
    for (const auto& x : tape) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_hip_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = std::string("static_cast<") + type + ">(tmp_" +
            namer.get_name(x.inputs()[0]) + ")";
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += "tmp_" + namer.get_name(x.inputs()[i]) + ", ";
        }
        value += "tmp_" + namer.get_name(x.inputs().back()) + ")";
      }
      os +=
          std::string("    ") + type + " tmp_" + xname + " = " + value + ";\n";
    }

    // Write output into the local vector.
    for (const auto& x : outputs) {
      os += "    vec_" + namer.get_name(x) + "[i] = tmp_" + namer.get_name(x) +
          ";\n";
    }

    // End of work loop
    if (!contiguous) {
      os += "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += std::string("    ") + xname + "_idx += " + xname +
            "_strides[NDIM - 1];\n";
      }
    }
    os += "  }\n";

    // Wide store of outputs (handles partial last vector via size bound).
    // Outputs of Compiled are contiguous in linear order; strided path still
    // packs work_per_thread consecutive linear elements starting at `index`
    // (CUDA parity — work_per_thread forced to 1 when last dim not divisible).
    for (const auto& x : outputs) {
      os += "  store_vector(" + namer.get_name(x) + " + index, 0, vec_" +
          namer.get_name(x) + ", size - index);\n";
    }

    os += "}\n";
  }
};

} // namespace rocm

constexpr const char* g_jit_includes = R"(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// Non-finite scalar constants are emitted by the shared codegen
// (mlx/backend/common/compiled.h: print_float_constant) as the bare tokens
// NAN / INFINITY / -INFINITY, which it expects the backend toolchain to define
// (Metal via <metal_stdlib>, CPU via <cmath>). hiprtc does NOT pull in <cmath>,
// so those tokens are undeclared and any compiled kernel carrying a non-finite
// constant (e.g. isinf/isfinite lowering, reduction identities) fails to build
// with "use of undeclared identifier 'INFINITY'". Provide them here, matching
// what the CPU backend injects into its generated source.
#ifndef INFINITY
#define INFINITY (__builtin_huge_valf())
#endif
#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif

// Standard type definitions for JIT compilation
using uint32_t = unsigned int;
using int32_t = signed int;
using uint64_t = unsigned long long;
using int64_t = signed long long;
using uint16_t = unsigned short;
using int16_t = signed short;
using uint8_t = unsigned char;
using int8_t = signed char;
using size_t = unsigned long;

// Simple array type for JIT compilation (hip/std/array not available in hiprtc)
namespace hip {
namespace std {
template <typename T, int N>
struct array {
  T data_[N];
  __device__ T& operator[](int i) { return data_[i]; }
  __device__ const T& operator[](int i) const { return data_[i]; }
};

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
  __device__ static float infinity() { return __int_as_float(0x7f800000); }
};
} // namespace std
} // namespace hip

// Math function overloads for bfloat16 and half types
// HIP doesn't provide native math functions for these types,
// so we convert to float, compute, and convert back.

__device__ inline hip_bfloat16 abs(hip_bfloat16 x) {
  return hip_bfloat16(fabsf(static_cast<float>(x)));
}
__device__ inline __half abs(__half x) {
  return __float2half(fabsf(__half2float(x)));
}

__device__ inline hip_bfloat16 exp(hip_bfloat16 x) {
  return hip_bfloat16(expf(static_cast<float>(x)));
}
__device__ inline __half exp(__half x) {
  return __float2half(expf(__half2float(x)));
}

__device__ inline hip_bfloat16 log(hip_bfloat16 x) {
  return hip_bfloat16(logf(static_cast<float>(x)));
}
__device__ inline __half log(__half x) {
  return __float2half(logf(__half2float(x)));
}

__device__ inline hip_bfloat16 sqrt(hip_bfloat16 x) {
  return hip_bfloat16(sqrtf(static_cast<float>(x)));
}
__device__ inline __half sqrt(__half x) {
  return __float2half(sqrtf(__half2float(x)));
}

__device__ inline hip_bfloat16 rsqrt(hip_bfloat16 x) {
  return hip_bfloat16(rsqrtf(static_cast<float>(x)));
}
__device__ inline __half rsqrt(__half x) {
  return __float2half(rsqrtf(__half2float(x)));
}

__device__ inline hip_bfloat16 sin(hip_bfloat16 x) {
  return hip_bfloat16(sinf(static_cast<float>(x)));
}
__device__ inline __half sin(__half x) {
  return __float2half(sinf(__half2float(x)));
}

__device__ inline hip_bfloat16 cos(hip_bfloat16 x) {
  return hip_bfloat16(cosf(static_cast<float>(x)));
}
__device__ inline __half cos(__half x) {
  return __float2half(cosf(__half2float(x)));
}

__device__ inline hip_bfloat16 tan(hip_bfloat16 x) {
  return hip_bfloat16(tanf(static_cast<float>(x)));
}
__device__ inline __half tan(__half x) {
  return __float2half(tanf(__half2float(x)));
}

__device__ inline hip_bfloat16 sinh(hip_bfloat16 x) {
  return hip_bfloat16(sinhf(static_cast<float>(x)));
}
__device__ inline __half sinh(__half x) {
  return __float2half(sinhf(__half2float(x)));
}

__device__ inline hip_bfloat16 cosh(hip_bfloat16 x) {
  return hip_bfloat16(coshf(static_cast<float>(x)));
}
__device__ inline __half cosh(__half x) {
  return __float2half(coshf(__half2float(x)));
}

__device__ inline hip_bfloat16 tanh(hip_bfloat16 x) {
  return hip_bfloat16(tanhf(static_cast<float>(x)));
}
__device__ inline __half tanh(__half x) {
  return __float2half(tanhf(__half2float(x)));
}

__device__ inline hip_bfloat16 asin(hip_bfloat16 x) {
  return hip_bfloat16(asinf(static_cast<float>(x)));
}
__device__ inline __half asin(__half x) {
  return __float2half(asinf(__half2float(x)));
}

__device__ inline hip_bfloat16 acos(hip_bfloat16 x) {
  return hip_bfloat16(acosf(static_cast<float>(x)));
}
__device__ inline __half acos(__half x) {
  return __float2half(acosf(__half2float(x)));
}

__device__ inline hip_bfloat16 atan(hip_bfloat16 x) {
  return hip_bfloat16(atanf(static_cast<float>(x)));
}
__device__ inline __half atan(__half x) {
  return __float2half(atanf(__half2float(x)));
}

__device__ inline hip_bfloat16 asinh(hip_bfloat16 x) {
  return hip_bfloat16(asinhf(static_cast<float>(x)));
}
__device__ inline __half asinh(__half x) {
  return __float2half(asinhf(__half2float(x)));
}

__device__ inline hip_bfloat16 acosh(hip_bfloat16 x) {
  return hip_bfloat16(acoshf(static_cast<float>(x)));
}
__device__ inline __half acosh(__half x) {
  return __float2half(acoshf(__half2float(x)));
}

__device__ inline hip_bfloat16 atanh(hip_bfloat16 x) {
  return hip_bfloat16(atanhf(static_cast<float>(x)));
}
__device__ inline __half atanh(__half x) {
  return __float2half(atanhf(__half2float(x)));
}

__device__ inline hip_bfloat16 ceil(hip_bfloat16 x) {
  return hip_bfloat16(ceilf(static_cast<float>(x)));
}
__device__ inline __half ceil(__half x) {
  return __float2half(ceilf(__half2float(x)));
}

__device__ inline hip_bfloat16 floor(hip_bfloat16 x) {
  return hip_bfloat16(floorf(static_cast<float>(x)));
}
__device__ inline __half floor(__half x) {
  return __float2half(floorf(__half2float(x)));
}

__device__ inline hip_bfloat16 rint(hip_bfloat16 x) {
  return hip_bfloat16(rintf(static_cast<float>(x)));
}
__device__ inline __half rint(__half x) {
  return __float2half(rintf(__half2float(x)));
}

__device__ inline hip_bfloat16 log2(hip_bfloat16 x) {
  return hip_bfloat16(log2f(static_cast<float>(x)));
}
__device__ inline __half log2(__half x) {
  return __float2half(log2f(__half2float(x)));
}

__device__ inline hip_bfloat16 log10(hip_bfloat16 x) {
  return hip_bfloat16(log10f(static_cast<float>(x)));
}
__device__ inline __half log10(__half x) {
  return __float2half(log10f(__half2float(x)));
}

__device__ inline hip_bfloat16 log1pf(hip_bfloat16 x) {
  return hip_bfloat16(::log1pf(static_cast<float>(x)));
}
__device__ inline __half log1pf(__half x) {
  return __float2half(::log1pf(__half2float(x)));
}

__device__ inline hip_bfloat16 expm1f(hip_bfloat16 x) {
  return hip_bfloat16(::expm1f(static_cast<float>(x)));
}
__device__ inline __half expm1f(__half x) {
  return __float2half(::expm1f(__half2float(x)));
}

__device__ inline hip_bfloat16 erff(hip_bfloat16 x) {
  return hip_bfloat16(::erff(static_cast<float>(x)));
}
__device__ inline __half erff(__half x) {
  return __float2half(::erff(__half2float(x)));
}

__device__ inline hip_bfloat16 erfinvf(hip_bfloat16 x) {
  return hip_bfloat16(::erfinvf(static_cast<float>(x)));
}
__device__ inline __half erfinvf(__half x) {
  return __float2half(::erfinvf(__half2float(x)));
}

__device__ inline hip_bfloat16 powf(hip_bfloat16 base, hip_bfloat16 exp) {
  return hip_bfloat16(::powf(static_cast<float>(base), static_cast<float>(exp)));
}
__device__ inline __half powf(__half base, __half exp) {
  return __float2half(::powf(__half2float(base), __half2float(exp)));
}

__device__ inline hip_bfloat16 fmodf(hip_bfloat16 x, hip_bfloat16 y) {
  return hip_bfloat16(::fmodf(static_cast<float>(x), static_cast<float>(y)));
}
__device__ inline __half fmodf(__half x, __half y) {
  return __float2half(::fmodf(__half2float(x), __half2float(y)));
}

__device__ inline hip_bfloat16 truncf(hip_bfloat16 x) {
  return hip_bfloat16(::truncf(static_cast<float>(x)));
}
__device__ inline __half truncf(__half x) {
  return __float2half(::truncf(__half2float(x)));
}

__device__ inline hip_bfloat16 atan2f(hip_bfloat16 y, hip_bfloat16 x) {
  return hip_bfloat16(::atan2f(static_cast<float>(y), static_cast<float>(x)));
}
__device__ inline __half atan2f(__half y, __half x) {
  return __float2half(::atan2f(__half2float(y), __half2float(x)));
}

// Include device operations
namespace mlx::core::rocm {

// Vectorized load/store for fused Compiled kernels (CUDA FusedKernelBuilder
// parity). hiprtc cannot #include device/utils.hpp, so the helpers live here.
template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];
  __device__ T& operator[](int i) { return val[i]; }
  __device__ T operator[](int i) const { return val[i]; }
};

template <int N, typename T>
__device__ inline bool is_aligned_ptr(const T* x) {
  return (reinterpret_cast<size_t>(x) % (N * sizeof(T))) == 0;
}
template <int N, typename T>
__device__ inline bool is_aligned_ptr(T* x) {
  return (reinterpret_cast<size_t>(x) % (N * sizeof(T))) == 0;
}

template <int N, typename T, typename SizeT>
__device__ inline AlignedVector<T, N>
load_vector(const T* ptr, uint32_t offset, SizeT size, T fallback) {
  if (is_aligned_ptr<N>(ptr) && SizeT((offset + 1) * N) <= size) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  }
  AlignedVector<T, N> v;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    v[i] = SizeT(N * offset + i) < size ? ptr[offset * N + i] : fallback;
  }
  return v;
}

template <int N, typename T, typename SizeT>
__device__ inline void store_vector(
    T* ptr,
    uint32_t offset,
    const AlignedVector<T, N>& vec,
    SizeT size) {
  if (is_aligned_ptr<N>(ptr) && SizeT((offset + 1) * N) <= size) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
    for (int i = 0; SizeT(offset * N + i) < size && i < N; ++i) {
      ptr[offset * N + i] = vec[i];
    }
  }
}

// Binary ops — promote half/bfloat16 through float to avoid precision loss
// that compounds across 28-36 transformer layers in LLM inference.
struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) + static_cast<float>(y));
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) - static_cast<float>(y));
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) * static_cast<float>(y));
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) / static_cast<float>(y));
  }
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) { return x > y ? x : y; }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) { return x < y ? x : y; }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) {
    return T(powf(static_cast<float>(base), static_cast<float>(exp)));
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x == y; }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x != y; }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x > y; }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x >= y; }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x < y; }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x <= y; }
};

struct LogicalAnd {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x && y; }
};

struct LogicalOr {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x || y; }
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return T(atan2f(static_cast<float>(y), static_cast<float>(x)));
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(fmodf(static_cast<float>(x), static_cast<float>(y)));
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(truncf(static_cast<float>(x) / static_cast<float>(y)));
  }
};

struct LogAddExp {
  __device__ hip_bfloat16 operator()(hip_bfloat16 x, hip_bfloat16 y) {
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float maxval = fx > fy ? fx : fy;
    float minval = fx > fy ? fy : fx;
    return hip_bfloat16(maxval + log1pf(expf(minval - maxval)));
  }

  __device__ __half operator()(__half x, __half y) {
    float fx = __half2float(x);
    float fy = __half2float(y);
    float maxval = fx > fy ? fx : fy;
    float minval = fx > fy ? fy : fx;
    return __float2half(maxval + log1pf(expf(minval - maxval)));
  }

  template <typename T>
  __device__ T operator()(T x, T y) {
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float maxval = fx > fy ? fx : fy;
    float minval = fx > fy ? fy : fx;
    return T(maxval + log1pf(expf(minval - maxval)));
  }
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) { return x & y; }
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) { return x | y; }
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) { return x ^ y; }
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) { return x << y; }
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) { return x >> y; }
};

// All unary math ops promote through float to support half/bfloat16.
// For float inputs the static_cast is a no-op.
#define UNARY_FLOAT_OP(name, op) \
struct name { \
  template <typename T> \
  __device__ T operator()(T x) { \
    return T(op(static_cast<float>(x))); \
  } \
};

// Unary ops
UNARY_FLOAT_OP(Abs, fabsf)
UNARY_FLOAT_OP(Exp, expf)
UNARY_FLOAT_OP(Log, logf)
UNARY_FLOAT_OP(Sqrt, sqrtf)

struct Negative {
  template <typename T>
  __device__ T operator()(T x) { return -x; }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    float fx = static_cast<float>(x);
    return T(fx * fx);
  }
};

struct Sigmoid {
  __device__ hip_bfloat16 operator()(hip_bfloat16 x) {
    float fx = static_cast<float>(x);
    float y = 1.0f / (1.0f + expf(-fabsf(fx)));
    return hip_bfloat16((fx < 0.0f) ? 1.0f - y : y);
  }

  __device__ __half operator()(__half x) {
    float fx = __half2float(x);
    float y = 1.0f / (1.0f + expf(-fabsf(fx)));
    return __float2half((fx < 0.0f) ? 1.0f - y : y);
  }

  template <typename T>
  __device__ T operator()(T x) {
    float fx = static_cast<float>(x);
    float y = 1.0f / (1.0f + expf(-fabsf(fx)));
    return T((fx < 0.0f) ? 1.0f - y : y);
  }
};

UNARY_FLOAT_OP(Tanh, tanhf)
UNARY_FLOAT_OP(Sin, sinf)
UNARY_FLOAT_OP(Cos, cosf)
UNARY_FLOAT_OP(Tan, tanf)
UNARY_FLOAT_OP(Sinh, sinhf)
UNARY_FLOAT_OP(Cosh, coshf)
UNARY_FLOAT_OP(Erf, erff)
UNARY_FLOAT_OP(ErfInv, erfinvf)
UNARY_FLOAT_OP(Expm1, expm1f)
UNARY_FLOAT_OP(Log1p, log1pf)
UNARY_FLOAT_OP(Log2, log2f)
UNARY_FLOAT_OP(Log10, log10f)
UNARY_FLOAT_OP(Ceil, ceilf)
UNARY_FLOAT_OP(Floor, floorf)
UNARY_FLOAT_OP(Round, rintf)
UNARY_FLOAT_OP(Rsqrt, rsqrtf)

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    float fx = static_cast<float>(x);
    return T((fx > 0.0f) - (fx < 0.0f));
  }
};

UNARY_FLOAT_OP(Asin, asinf)
UNARY_FLOAT_OP(Acos, acosf)
UNARY_FLOAT_OP(Atan, atanf)
UNARY_FLOAT_OP(Asinh, asinhf)
UNARY_FLOAT_OP(Acosh, acoshf)
UNARY_FLOAT_OP(Atanh, atanhf)

// Primitive::name() uses Arc* / BitwiseInvert (Metal/CUDA device headers match).
// hiprtc fused kernels instantiate by that name — alias the short C lib names.
using ArcSin = Asin;
using ArcCos = Acos;
using ArcTan = Atan;
using ArcSinh = Asinh;
using ArcCosh = Acosh;
using ArcTanh = Atanh;

struct LogicalNot {
  template <typename T>
  __device__ bool operator()(T x) { return !x; }
};

struct BitwiseNot {
  template <typename T>
  __device__ T operator()(T x) { return ~x; }
};
using BitwiseInvert = BitwiseNot;

// Equal::name() may return "NaNEqual" when equal_nan is set.
struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    // Match CPU NaNEqual: true if both NaN or x==y
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    bool xn = !(fx == fx);
    bool yn = !(fy == fy);
    return (xn && yn) || (fx == fy);
  }
};

// complex64 helpers used by some compiled graphs (identity on reals)
struct Real {
  template <typename T>
  __device__ T operator()(T x) { return x; }
};
struct Imag {
  template <typename T>
  __device__ T operator()(T x) { return T(0); }
};
struct Conjugate {
  template <typename T>
  __device__ T operator()(T x) { return x; }
};

#undef UNARY_FLOAT_OP

struct Reciprocal {
  template <typename T>
  __device__ T operator()(T x) { return T(1.0f / static_cast<float>(x)); }
};

// Ternary ops
struct Select {
  template <typename T>
  __device__ T operator()(bool c, T x, T y) { return c ? x : y; }
};

// Broadcast is a no-op in fused kernels (handled by indexing)
struct Broadcast {
  template <typename T>
  __device__ T operator()(T x) { return x; }
};

} // namespace mlx::core::rocm

#define inf hip::std::numeric_limits<float>::infinity()
)";

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  // Determine the work per thread for the vectorized reads/writes.
  int max_size = 1;
  for (const auto& x : outputs) {
    max_size = (max_size > x.itemsize()) ? max_size : x.itemsize();
  }
  int work_per_thread = 16 / max_size;

  rocm::JitModule& mod = rocm::get_jit_module(s.device, lib_name(), [&]() {
    // Build source code.
    rocm::FusedKernelBuilder builder{
        g_jit_includes, lib_name(), inputs_, outputs_, tape_, is_constant_};
    builder.os += "namespace mlx::core::rocm {\n\n";
    builder.build("_contiguous", true);
    builder.os += "\n";
    builder.build("_strided", false);
    builder.os += "\n} // namespace mlx::core::rocm\n";

    // Build kernel names.
    std::vector<std::string> kernel_names;
    kernel_names.push_back(
        std::string("mlx::core::rocm::") + lib_name() +
        "_contiguous<uint32_t, " + std::to_string(work_per_thread) + ">");
    kernel_names.push_back(
        std::string("mlx::core::rocm::") + lib_name() +
        "_contiguous<int64_t, " + std::to_string(work_per_thread) + ">");
    for (auto wpt : std::array<int, 2>{1, work_per_thread}) {
      for (int i = 1; i <= MAX_NDIM; ++i) {
        kernel_names.push_back(
            std::string("mlx::core::rocm::") + lib_name() + "_strided<" +
            std::to_string(i) + ", uint32_t, " + std::to_string(wpt) + ">");
        kernel_names.push_back(
            std::string("mlx::core::rocm::") + lib_name() + "_strided<" +
            std::to_string(i) + ", int64_t, " + std::to_string(wpt) + ">");
      }
    }

    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  // Collapse contiguous dims to route to a faster kernel if possible.
  auto [contiguous, shape, strides_vec] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index.
  bool large = compiled_use_large_index(inputs, outputs, contiguous);

  rocm::KernelArgs args;
  // Put inputs.
  int strides_index = 1;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    args.append(x);
    if (!contiguous && !is_scalar(x)) {
      args.append_ptr(strides_vec[strides_index++].data());
    }
  }

  // Put outputs.
  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);
  for (auto& x : outputs) {
    args.append(x);
  }

  // Put shape and size.
  if (!contiguous) {
    args.append_ptr(shape.data());
  }
  if (large) {
    args.append<int64_t>(outputs[0].data_size());
  } else {
    args.append<uint32_t>(outputs[0].data_size());
  }

  // Choose work per thread
  if (!contiguous && shape.back() % work_per_thread != 0) {
    work_per_thread = 1;
  }

  // Launch kernel.
  const char* index_type = large ? "int64_t" : "uint32_t";
  std::string kernel_name = std::string("mlx::core::rocm::") + lib_name();
  if (contiguous) {
    kernel_name += std::string("_contiguous<") + index_type + ", " +
        std::to_string(work_per_thread) + ">";
  } else {
    kernel_name += std::string("_strided<") + std::to_string(shape.size()) +
        ", " + index_type + ", " + std::to_string(work_per_thread) + ">";
  }

  auto& encoder = rocm::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }

  auto kernel = mod.get_kernel(kernel_name);

  // Calculate launch configuration — use runtime CU/warp-aware block size so
  // reduced-CU devices (e.g. gfx1152 partitions) pack better occupancy.
  int64_t total_work =
      (outputs[0].data_size() + work_per_thread - 1) / work_per_thread;
  int block_size = encoder.device().preferred_block_size(
      total_work > INT_MAX ? -1 : static_cast<int>(total_work));
  int num_blocks =
      static_cast<int>((total_work + block_size - 1) / block_size);
  if (num_blocks < 1) {
    num_blocks = 1;
  }

  launch_module_kernel(
      encoder,
      kernel,
      dim3(num_blocks, 1, 1),
      dim3(block_size, 1, 1),
      0,
      std::move(args));
}

} // namespace mlx::core
