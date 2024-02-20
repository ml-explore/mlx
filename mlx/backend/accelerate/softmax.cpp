// Copyright © 2023 Apple Inc.

#include <cassert>
#include <limits>

#include <arm_neon.h>
#include <simd/math.h>
#include <simd/vector.h>

#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

/**
 * Compute exp(x) in an optimizer friendly way as follows:
 *
 * First change the problem to computing 2**y where y = x / ln(2).
 *
 * Now we will compute 2**y as 2**y1 * 2**y2 where y1 is the integer part
 * `ipart` and y2 is fractional part. For the integer part we perform bit
 * shifting and for the fractional part we use a polynomial approximation.
 *
 * The algorithm and constants of the polynomial taken from
 * https://github.com/akohlmey/fastermath/blob/master/src/exp.c which took them
 * from Cephes math library.
 *
 * Note: The implementation below is a general fast exp. There could be faster
 *       implementations for numbers strictly < 0.
 */
inline simd_float16 simd_fast_exp(simd_float16 x) {
  x *= 1.442695; // multiply with log_2(e)
  simd_float16 ipart, fpart;
  simd_int16 epart;
  x = simd_clamp(x, -80, 80);
  ipart = simd::floor(x + 0.5);
  fpart = x - ipart;

  x = 1.535336188319500e-4f;
  x = x * fpart + 1.339887440266574e-3f;
  x = x * fpart + 9.618437357674640e-3f;
  x = x * fpart + 5.550332471162809e-2f;
  x = x * fpart + 2.402264791363012e-1f;
  x = x * fpart + 6.931472028550421e-1f;
  x = x * fpart + 1.000000000000000f;

  // generate 2**ipart in the floating point representation using integer
  // bitshifting
  epart = (simd_int(ipart) + 127) << 23;

  return (*(simd_float16*)&epart) * x;
}

/**
 * The ARM neon equivalent of the fast exp above.
 */
inline float16x8_t neon_fast_exp(float16x8_t x) {
  x = vmulq_f16(x, vdupq_n_f16(1.442695)); // multiply with log_2(e)
  x = vmaxq_f16(x, vdupq_n_f16(-14)); // clamp under with -14
  x = vminq_f16(x, vdupq_n_f16(14)); // clamp over with 14

  float16x8_t ipart = vrndmq_f16(vaddq_f16(x, vdupq_n_f16(0.5)));
  float16x8_t fpart = vsubq_f16(x, ipart);

  x = vdupq_n_f16(1.535336188319500e-4f);
  x = vfmaq_f16(vdupq_n_f16(1.339887440266574e-3f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(1.339887440266574e-3f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(9.618437357674640e-3f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(5.550332471162809e-2f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(2.402264791363012e-1f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(6.931472028550421e-1f), x, fpart);
  x = vfmaq_f16(vdupq_n_f16(1.000000000000000f), x, fpart);

  // generate 2**ipart in the floating point representation using integer
  // bitshifting
  int16x8_t epart = vcvtq_s16_f16(ipart);
  epart = vaddq_s16(epart, vdupq_n_s16(15));
  epart = vshlq_n_s16(epart, 10);

  return vmulq_f16(vreinterpretq_f16_s16(epart), x);
}

/**
 * Implementation of folding maximum for ARM neon. This should possibly be
 * refactored out of softmax.cpp at some point.
 */
inline float16_t neon_reduce_max(float16x8_t x) {
  float16x4_t y;
  y = vpmax_f16(vget_low_f16(x), vget_high_f16(x));
  y = vpmax_f16(y, y);
  y = vpmax_f16(y, y);
  return vget_lane_f16(y, 0);
}

/**
 * Implementation of folding sum for ARM neon. This should possibly be
 * refactored out of softmax.cpp at some point.
 */
inline float16_t neon_reduce_add(float16x8_t x) {
  float16x4_t y;
  float16x4_t zero = vdup_n_f16(0);
  y = vpadd_f16(vget_low_f16(x), vget_high_f16(x));
  y = vpadd_f16(y, zero);
  y = vpadd_f16(y, zero);
  return vget_lane_f16(y, 0);
}

template <typename T, typename VT>
struct AccelerateSimdOps {
  VT init(T a) {
    return a;
  }

  VT load(const T* a) {
    return *(VT*)a;
  }

  void store(T* dst, VT x) {
    *(VT*)dst = x;
  }

  VT max(VT a, VT b) {
    return simd_max(a, b);
  };

  VT exp(VT x) {
    return simd_fast_exp(x);
  }

  VT add(VT a, VT b) {
    return a + b;
  }

  VT sub(VT a, T b) {
    return a - b;
  }

  VT mul(VT a, VT b) {
    return a * b;
  }

  VT mul(VT a, T b) {
    return a * b;
  }

  T reduce_max(VT x) {
    return simd_reduce_max(x);
  }

  T reduce_add(VT x) {
    return simd_reduce_add(x);
  }
};

template <typename T, typename VT>
struct NeonFp16SimdOps {
  VT init(T a) {
    return vdupq_n_f16(a);
  }

  VT load(const T* a) {
    return vld1q_f16(a);
  }

  void store(T* dst, VT x) {
    vst1q_f16(dst, x);
  }

  VT max(VT a, VT b) {
    return vmaxq_f16(a, b);
  };

  VT exp(VT x) {
    return neon_fast_exp(x);
  }

  VT add(VT a, VT b) {
    return vaddq_f16(a, b);
  }

  VT sub(VT a, T b) {
    return vsubq_f16(a, vdupq_n_f16(b));
  }

  VT mul(VT a, VT b) {
    return vmulq_f16(a, b);
  }

  VT mul(VT a, T b) {
    return vmulq_f16(a, vdupq_n_f16(b));
  }

  T reduce_max(VT x) {
    return neon_reduce_max(x);
  }

  T reduce_add(VT x) {
    return neon_reduce_add(x);
  }
};

template <typename T, typename VT, typename Ops, int N>
void softmax(const array& in, array& out) {
  Ops ops;

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();
  int M = in.shape().back();
  int L = in.data_size() / M;
  const T* current_in_ptr;
  T* current_out_ptr;

  for (int i = 0; i < L; i++, in_ptr += M, out_ptr += M) {
    // Find the maximum
    current_in_ptr = in_ptr;
    VT vmaximum = ops.init(-std::numeric_limits<float>::infinity());
    size_t s = M;
    while (s >= N) {
      vmaximum = ops.max(ops.load(current_in_ptr), vmaximum);
      current_in_ptr += N;
      s -= N;
    }
    T maximum = ops.reduce_max(vmaximum);
    while (s-- > 0) {
      maximum = std::max(maximum, *current_in_ptr);
      current_in_ptr++;
    }

    // Compute the normalizer and the exponentials
    VT vnormalizer = ops.init(0.0);
    current_out_ptr = out_ptr;
    current_in_ptr = in_ptr;
    s = M;
    while (s >= N) {
      VT vexp = ops.exp(ops.sub(*(VT*)current_in_ptr, maximum));
      ops.store(current_out_ptr, vexp);
      *(VT*)current_out_ptr = vexp;
      vnormalizer = ops.add(vnormalizer, vexp);
      current_in_ptr += N;
      current_out_ptr += N;
      s -= N;
    }
    T normalizer = ops.reduce_add(vnormalizer);
    while (s-- > 0) {
      T _exp = std::exp(*current_in_ptr - maximum);
      *current_out_ptr = _exp;
      normalizer += _exp;
      current_in_ptr++;
      current_out_ptr++;
    }
    normalizer = 1 / normalizer;

    // Normalize
    current_out_ptr = out_ptr;
    s = M;
    while (s >= N) {
      ops.store(current_out_ptr, ops.mul(*(VT*)current_out_ptr, normalizer));
      current_out_ptr += N;
      s -= N;
    }
    while (s-- > 0) {
      *current_out_ptr *= normalizer;
      current_out_ptr++;
    }
  }
}

} // namespace

void Softmax::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // Make sure that the last dimension is contiguous
  auto check_input = [](array x) {
    bool no_copy = x.strides()[x.ndim() - 1] == 1;
    if (x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      copy(x, x_copy, CopyType::General);
      return x_copy;
    }
  };
  array in = check_input(std::move(inputs[0]));
  out.set_data(
      allocator::malloc_or_wait(in.data_size() * in.itemsize()),
      in.data_size(),
      in.strides(),
      in.flags());

  switch (in.dtype()) {
    case bool_:
    case uint8:
    case uint16:
    case uint32:
    case uint64:
    case int8:
    case int16:
    case int32:
    case int64:
      throw std::invalid_argument(
          "Softmax is defined only for floating point types");
      break;
    case float32:
      softmax<float, simd_float16, AccelerateSimdOps<float, simd_float16>, 16>(
          in, out);
      break;
    case float16:
      softmax<
          float16_t,
          float16x8_t,
          NeonFp16SimdOps<float16_t, float16x8_t>,
          8>(in, out);
      break;
    case bfloat16:
      eval(inputs, out);
      break;
    case complex64:
      eval(inputs, out);
      break;
  }
}

} // namespace mlx::core
