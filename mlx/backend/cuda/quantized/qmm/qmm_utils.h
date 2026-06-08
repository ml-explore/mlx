// Copyright © 2026 Apple Inc.

#include "mlx/dtype_utils.h"

namespace mlx::core {

inline auto
make_problem_shape(const array& x, const array& w, const array& out) {
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  bool broadcast_b = (w.ndim() <= 2) || (w.size() != w.data_size());
  return std::make_tuple(m, n, k, l, broadcast_b);
}

inline const char* get_weight_cutlass_type(const Dtype& dtype) {
  switch (dtype) {
    case float16:
      return "cutlass::half_t";
    case bfloat16:
      return "cutlass::bfloat16_t";
    case float32:
      return "float";
    default:
      throw std::invalid_argument(
          fmt::format(
              "[quantized_matmul] Unsupported dtype: {}.",
              dtype_to_string(dtype)));
  }
}

inline std::tuple<const char*, const char*>
get_quant_cutlass_types(const char* ctype_x, int bits, QuantizationMode mode) {
  if (mode == QuantizationMode::Mxfp4) {
    return {"cutlass::float_e2m1_t", "cutlass::float_ue8m0_t"};
  } else if (mode == QuantizationMode::Mxfp8) {
    return {"cutlass::float_e4m3_t", "cutlass::float_ue8m0_t"};
  } else if (mode == QuantizationMode::Nvfp4) {
    return {"cutlass::float_e2m1_t", "cutlass::float_e4m3_t"};
  } else {
    if (bits == 2) {
      return {"cutlass::uint2b_t", ctype_x};
    } else if (bits == 3) {
      return {"cutlass::uint3b_t", ctype_x};
    } else if (bits == 4) {
      return {"cutlass::uint4b_t", ctype_x};
    } else if (bits == 5) {
      return {"cutlass::uint5b_t", ctype_x};
    } else if (bits == 6) {
      return {"cutlass::uint6b_t", ctype_x};
    } else if (bits == 8) {
      return {"uint8_t", ctype_x};
    } else {
      throw std::invalid_argument(
          fmt::format(
              "[quantized_matmul] {}-bit quantization is not supported.",
              bits));
    }
  }
}

inline std::tuple<const char*, const char*, const char*> get_qmm_cutlass_types(
    const array& x,
    int bits,
    QuantizationMode mode = QuantizationMode::Affine) {
  auto ctype_x = get_weight_cutlass_type(x.dtype());
  auto [ctype_q, ctype_s] = get_quant_cutlass_types(ctype_x, bits, mode);
  return {ctype_x, ctype_q, ctype_s};
}

} // namespace mlx::core
