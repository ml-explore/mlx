// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cuda/std/cmath>

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x;
    } else {
      return cuda::std::abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::atanh(x);
  }
};

struct BitwiseInvert {
  template <typename T>
  __device__ T operator()(T x) {
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{cuda::std::ceil(x.real()), cuda::std::ceil(x.imag())};
    } else {
      return cuda::std::ceil(x);
    }
  }
};

struct Conjugate {
  template <typename T>
  __device__ complex_t<T> operator()(complex_t<T> x) {
    return cuda::std::conj(x);
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::cos(x);
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::cosh(x);
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erfinv(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erfinv(__bfloat162float(x));
    } else {
      return erfinv(x);
    }
  }
};

struct BesselI0e {
  template <typename T>
  __device__ T operator()(T x) {
    float v = static_cast<float>(x);
    float ax = v < 0.0f ? -v : v;
    // Cephes Chebyshev coefficients for i0e, |x| <= 8
    float A[30] = {
        -4.41534164647933937950e-18f, 3.33079451882223809783e-17f,
        -2.43127984654795469359e-16f, 1.71539128555513303061e-15f,
        -1.16853328779934516808e-14f, 7.67618549860493561688e-14f,
        -4.85644678311192946090e-13f, 2.95505266312963983461e-12f,
        -1.72682629144155570723e-11f, 9.67580903537323691224e-11f,
        -5.18979560163526290666e-10f, 2.65982372468238665035e-09f,
        -1.30002500998624804212e-08f, 6.04699502254191894932e-08f,
        -2.67079385394061173391e-07f, 1.11738753912010371815e-06f,
        -4.41673835845875056359e-06f, 1.64484480707288970893e-05f,
        -5.75419501008210370398e-05f, 1.88502885095841655729e-04f,
        -5.76375574538582365885e-04f, 1.63947561694133579842e-03f,
        -4.32430999505057594430e-03f, 1.05464603945949983183e-02f,
        -2.37374148058994688156e-02f, 4.93052842396707084878e-02f,
        -9.49010970480476444210e-02f, 1.71620901522208775349e-01f,
        -3.04682672343198398683e-01f, 6.76795274409476084995e-01f,
    };
    // Cephes Chebyshev coefficients for i0e, |x| > 8
    float B[25] = {
        -7.23318048787475395456e-18f, -4.83050448594418207126e-18f,
        4.46562142029675999901e-17f,  3.46122286769746109310e-17f,
        -2.82762398051658348494e-16f, -3.42548561967721913462e-16f,
        1.77256013305652638360e-15f,  3.81168066935262242075e-15f,
        -9.55484669882830764870e-15f, -4.15056934728722208663e-14f,
        1.54008621752140982691e-14f,  3.85277838274214270114e-13f,
        7.18012445138366623367e-13f,  -1.79417853150680611778e-12f,
        -1.32158118404477131188e-11f, -3.14991652796324136454e-11f,
        1.18891471078464383424e-11f,  4.94060238822496958910e-10f,
        3.39623202570838634515e-09f,  2.26666899049817806459e-08f,
        2.04891858946906374183e-07f,  2.89137052083475648297e-06f,
        6.88975834691682398426e-05f,  3.36911647825569408990e-03f,
        8.04490411014108831608e-01f,
    };
    // Clenshaw recurrence
    auto chbevl = [](float x, float* coeffs, int n) {
      float b0 = coeffs[0];
      float b1 = 0.0f;
      float b2;
      for (int i = 1; i < n; i++) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coeffs[i];
      }
      return 0.5f * (b0 - b2);
    };
    float result;
    if (ax <= 8.0f) {
      result = chbevl(ax * 0.5f - 2.0f, A, 30);
    } else {
      result = chbevl(32.0f / ax - 2.0f, B, 25) / ::sqrtf(ax);
    }
    return static_cast<T>(result);
  }
};

struct BesselI1e {
  template <typename T>
  __device__ T operator()(T x) {
    float v = static_cast<float>(x);
    float ax = v < 0.0f ? -v : v;
    // Cephes Chebyshev coefficients for i1e, |x| <= 8
    float A[29] = {
        2.77791411276104639959e-18f,  -2.11142121435816608115e-17f,
        1.55363195773620046921e-16f,  -1.10559694773538630805e-15f,
        7.60068429473540693410e-15f,  -5.04218550472791168711e-14f,
        3.22379336594557470981e-13f,  -1.98397439776494371520e-12f,
        1.17361862988909016308e-11f,  -6.66348972350202774223e-11f,
        3.62559028155211703701e-10f,  -1.88724975172282928790e-09f,
        9.38153738649577178388e-09f,  -4.44505912879632808065e-08f,
        2.00329475355213526229e-07f,  -8.56872026469545474066e-07f,
        3.47025130813767847674e-06f,  -1.32731636560394358279e-05f,
        4.78156510755005422638e-05f,  -1.61760815825896745588e-04f,
        5.12285956168575772895e-04f,  -1.51357245063125314899e-03f,
        4.15642294431288815669e-03f,  -1.05640848946261981558e-02f,
        2.47264490306265168283e-02f,  -5.29459812080949914269e-02f,
        1.02643658689847095384e-01f,  -1.76416518357834055153e-01f,
        2.52587186443633654823e-01f,
    };
    // Cephes Chebyshev coefficients for i1e, |x| > 8
    float B[25] = {
        7.51729631084210481353e-18f,  4.41434832307170791151e-18f,
        -4.65030536848935832153e-17f, -3.20952592199342395980e-17f,
        2.96262899764595013876e-16f,  3.30820231092092828324e-16f,
        -1.88035477551078244854e-15f, -3.81440307243700780478e-15f,
        1.04202769841288027642e-14f,  4.27244001671195135429e-14f,
        -2.10154184277266431302e-14f, -4.08355111109219731823e-13f,
        -7.19855177624590851209e-13f, 2.03562854414708950722e-12f,
        1.41258074366137813316e-11f,  3.25260358301548823856e-11f,
        -1.89749581235054123450e-11f, -5.58974346219658380687e-10f,
        -3.83538038596423702205e-09f, -2.63146884688951950684e-08f,
        -2.51223623787020892529e-07f, -3.88256480887769039346e-06f,
        -1.10588938762623716291e-04f, -9.76109749136146840777e-03f,
        7.78576235018280120474e-01f,
    };
    auto chbevl = [](float x, float* coeffs, int n) {
      float b0 = coeffs[0];
      float b1 = 0.0f;
      float b2;
      for (int i = 1; i < n; i++) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coeffs[i];
      }
      return 0.5f * (b0 - b2);
    };
    float result;
    if (ax <= 8.0f) {
      result = chbevl(ax * 0.5f - 2.0f, A, 29) * ax;
    } else {
      result = chbevl(32.0f / ax - 2.0f, B, 25) / ::sqrtf(ax);
    }
    if (v < 0.0f)
      result = -result;
    return static_cast<T>(result);
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::exp(x);
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::expm1(x);
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{cuda::std::floor(x.real()), cuda::std::floor(x.imag())};
    } else {
      return cuda::std::floor(x);
    }
  }
};

struct Imag {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.imag();
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      auto y = Log{}(x);
      return {y.real() / CUDART_LN2_F, y.imag() / CUDART_LN2_F};
    } else {
      return cuda::std::log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::log10(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T z) {
    if constexpr (is_complex_v<T>) {
      float x = z.real();
      float y = z.imag();
      float zabs = Abs{}(z).real();
      float theta = atan2f(y, x + 1);
      if (zabs < 0.5f) {
        float r = x * (2 + x) + y * y;
        if (r == 0) { // handle underflow
          return {x, theta};
        }
        return {0.5f * log1pf(r), theta};
      } else {
        float z0 = hypotf(x + 1, y);
        return {logf(z0), theta};
      }
    } else {
      return cuda::std::log1p(z);
    }
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return T{0, 0} - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.real();
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return {cuda::std::rint(x.real()), cuda::std::rint(x.imag())};
    } else {
      return cuda::std::rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + cuda::std::exp(cuda::std::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (is_complex_v<T>) {
      if (x.real() == 0 && x.imag() == 0) {
        return x;
      } else {
        return x / Abs()(x);
      }
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::sin(x);
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::sinh(x);
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return 1.0f / Sqrt{}(x);
    } else if constexpr (cuda::std::is_same_v<T, __half>) {
      return rsqrt(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return rsqrt(__bfloat162float(x));
    } else {
      return rsqrt(x);
    }
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::tan(x);
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::tanh(x);
  }
};

struct ToFP8 {
  template <typename T>
  __device__ uint8_t operator()(T x) {
    return __nv_fp8_e4m3(x).__x;
  }
};

struct FromFP8 {
  __device__ float operator()(uint8_t x) {
    return float(*(__nv_fp8_e4m3*)(&x));
  }
};

} // namespace mlx::core::cu
