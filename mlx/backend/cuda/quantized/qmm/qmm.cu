// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"

#include <cute/tensor.hpp>

namespace mlx::core {

namespace {

inline bool is_last_2_dims_row_contiguous(const array& x) {
  return x.flags().contiguous && (x.ndim() >= 2) && (x.strides(-1) == 1) &&
      (x.strides(-2) == x.shape(-1));
}

} // namespace

#if defined(MLX_CUDA_SM90A_ENABLED)
// Defined in qmm_sm90.cu.
template <int TileN>
void qmm_sm90_impl(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);
#endif // defined(MLX_CUDA_SM90A_ENABLED)

bool supports_qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() != 9) {
    return false;
  }
  int k = x.shape(-1);
  if (k % 64 != 0) {
    return false;
  }
  if (!biases) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales) ||
      !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits % 2 != 0) {
    return false;
  }
  if (group_size < k) {
    return false;
  }
  if (mode != QuantizationMode::Affine) {
    return false;
  }
  return true;
}

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
#if defined(MLX_CUDA_SM90A_ENABLED)
  auto dispatch = [&]<int TileN>() {
    qmm_sm90_impl<TileN>(
        x, w, scales, biases, out, bits, group_size, encoder, s);
  };
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (m <= 16) {
    dispatch.template operator()<16>();
  } else if (m <= 32) {
    dispatch.template operator()<32>();
  } else if (m <= 64) {
    dispatch.template operator()<64>();
  } else if (m <= 128) {
    dispatch.template operator()<128>();
  } else {
    dispatch.template operator()<256>();
  }
#else
  throw std::runtime_error(
      "[quantized_matmul] Hopper-only kernel is not available.");
#endif // defined(MLX_CUDA_SM90A_ENABLED)
}

// Defined in qmm_sm80.cu.
template <int TileM>
void qmm_sm80_impl(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

bool supports_qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() < 8) {
    return false;
  }
  int n = out.shape(-1);
  int k = x.shape(-1);
  if ((n % 128 != 0) || (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits != 4 && bits != 8) {
    return false;
  }
  return true;
}

void qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto dispatch = [&]<int TileM>() {
    qmm_sm80_impl<TileM>(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        bits,
        group_size,
        mode,
        encoder);
  };
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (m <= 16) {
    dispatch.template operator()<16>();
  } else if (m <= 32) {
    dispatch.template operator()<32>();
  } else {
    dispatch.template operator()<64>();
  }
}

// Defined in qmm_naive.cu.
template <int TileM, bool KMajor, bool HasKResidue, bool SM80>
void qmm_naive_impl(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

bool supports_qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (transpose && (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  return true;
}

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto dispatch = [&]<int TileM, bool KMajor, bool HasKResidue, bool SM80>() {
    qmm_naive_impl<TileM, KMajor, HasKResidue, SM80>(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        bits,
        group_size,
        mode,
        encoder);
  };
  auto dispatch_k = [&](auto k_major, bool has_k_residue, auto&& f) {
    if constexpr (k_major.value) {
      if (has_k_residue) {
        throw std::invalid_argument(
            "[quantized_matmul] K must be multiples of group_size.");
      }
      f.template operator()<false>();
    } else {
      dispatch_bool(has_k_residue, [&](auto has_k_residue) {
        f.template operator()<has_k_residue.value>();
      });
    }
  };
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int k = x.shape(-1);
  bool has_k_residue = k % group_size != 0;
  bool sm80 = encoder.device().compute_capability_major() >= 8;
  dispatch_bool(transpose, [&](auto k_major) {
    dispatch_k(k_major, has_k_residue, [&]<bool HasKResidue>() {
      dispatch_bool(sm80, [&](auto sm80) {
        constexpr bool KMajor = k_major.value;
        constexpr bool SM80 = sm80.value;
        if (m <= 16) {
          dispatch.template operator()<16, KMajor, HasKResidue, SM80>();
        } else if (m <= 32) {
          dispatch.template operator()<32, KMajor, HasKResidue, SM80>();
        } else {
          dispatch.template operator()<64, KMajor, HasKResidue, SM80>();
        }
      });
    });
  });
}

bool supports_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  // The fp_qmv kernel uses less registers and is faster for sm120. For sm80/90
  // the qmv kernel is faster. We didn't test sm89/100.
  if (device.compute_capability_major() <= 9) {
    return false;
  }
  bool non_batched = w.ndim() == 2;
  int k = x.shape(-1);
  int n = out.shape(-1);
  int vec_batch = non_batched ? x.size() / k : x.shape(-2);
  if (vec_batch > 8) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (mode == QuantizationMode::Affine) {
    return false;
  }
  return true;
}

bool supports_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (k % 8 != 0) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  return true;
}

} // namespace mlx::core
