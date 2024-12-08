// Copyright © 2024 Apple Inc.

#include <cstring>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename mask_t>
inline void mask_matrix(
    T* data,
    const mask_t* mask,
    int block_size,
    const int X,
    const int Y,
    const int64_t X_data_str,
    const int64_t Y_data_str,
    const int64_t X_mask_str,
    const int64_t Y_mask_str,
    const size_t mask_offset) {
  int tX = (X + block_size - 1) / block_size;
  int tY = (Y + block_size - 1) / block_size;

  for (int i = 0; i < tX; i++) {
    for (int j = 0; j < tY; j++) {
      mask_t do_mask = mask[mask_offset + i * X_mask_str + j * Y_mask_str];
      if (do_mask != 1) {
        int loc_x = i * block_size;
        int loc_y = j * block_size;
        T* data_block = data + loc_x * X_data_str + loc_y * Y_data_str;

        int size_x = std::min(block_size, X - loc_x);
        int size_y = std::min(block_size, Y - loc_y);
        for (int ii = 0; ii < size_x; ii++) {
          for (int jj = 0; jj < size_y; jj++) {
            if constexpr (std::is_same_v<mask_t, bool>) {
              data_block[ii * X_data_str + jj * Y_data_str] = T(0.);
            } else {
              data_block[ii * X_data_str + jj * Y_data_str] *= do_mask;
            }
          }
        }
      }
    }
  }
}

} // namespace

void BlockMaskedMM::eval(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[BlockMaskedMM::eval] Currently only supports float32.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  auto check_transpose =
      [](const array& arr, bool do_copy, bool expand_all = false) {
        auto stx = arr.strides()[arr.ndim() - 2];
        auto sty = arr.strides()[arr.ndim() - 1];
        if (!expand_all && stx == arr.shape(-1) && sty == 1) {
          if (do_copy) {
            array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
            copy(arr, arr_copy, CopyType::Vector);
            return std::make_tuple(false, stx, arr_copy);
          }
          return std::make_tuple(false, stx, arr);
        } else if (!expand_all && stx == 1 && sty == arr.shape(-2)) {
          if (do_copy) {
            array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
            copy(arr, arr_copy, CopyType::Vector);
            return std::make_tuple(true, sty, arr_copy);
          }
          return std::make_tuple(true, sty, arr);
        } else {
          array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
          copy(arr, arr_copy, CopyType::General);
          int64_t stx = arr.shape(-1);
          return std::make_tuple(false, stx, arr_copy);
        }
      };

  bool has_op_mask = inputs.size() > 3;
  bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;
  auto [a_transposed, lda, a] =
      check_transpose(a_pre, has_op_mask, inputs.back().dtype() != bool_);
  auto [b_transposed, ldb, b] =
      check_transpose(b_pre, has_op_mask, inputs.back().dtype() != bool_);

  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0) {
    std::memset(static_cast<void*>(out.data<float>()), 0, out.nbytes());
    return;
  }

  auto mask_array = [](const array& mask,
                       float* data,
                       int block_size,
                       int batch_idx,
                       int X,
                       int Y,
                       size_t X_data_str,
                       size_t Y_data_str) {
    auto mask_offset = elem_to_loc(
        mask.shape(-1) * mask.shape(-2) * batch_idx,
        mask.shape(),
        mask.strides());

    auto X_mask_str = mask.strides()[mask.ndim() - 2];
    auto Y_mask_str = mask.strides()[mask.ndim() - 1];

    if (mask.dtype() == bool_) {
      return mask_matrix(
          data,
          mask.data<bool>(),
          block_size,
          X,
          Y,
          X_data_str,
          Y_data_str,
          X_mask_str,
          Y_mask_str,
          mask_offset);
    } else {
      return mask_matrix(
          data,
          mask.data<float>(),
          block_size,
          X,
          Y,
          X_data_str,
          Y_data_str,
          X_mask_str,
          Y_mask_str,
          mask_offset);
    }
  };

  for (int i = 0; i < (out.size() / (M * size_t(N))); ++i) {
    // Adjust pointer
    float* ai =
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides());
    float* bi =
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides());
    float* ci = out.data<float>() + M * N * i;

    // Zero out blocks in a and b if needed
    if (has_op_mask) {
      auto& a_mask = inputs[inputs.size() - 2];
      mask_array(
          a_mask,
          ai,
          block_size_,
          i,
          M,
          K,
          a_transposed ? 1 : lda,
          a_transposed ? lda : 1);

      auto& b_mask = inputs[inputs.size() - 1];
      mask_array(
          b_mask,
          bi,
          block_size_,
          i,
          K,
          N,
          b_transposed ? 1 : ldb,
          b_transposed ? ldb : 1);
    }

    // Do matmul
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        1.0, // alpha
        ai,
        lda,
        bi,
        ldb,
        0.0, // beta
        ci,
        out.shape(-1) // ldc
    );

    // Zero out blocks in out
    if (has_out_mask) {
      mask_array(inputs[2], ci, block_size_, i, M, N, N, 1);
    }
  }
}

void GatherMM::eval(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[GatherMM::eval] Currently only supports float32.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  auto check_transpose = [](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      int64_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);

  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0) {
    std::memset(static_cast<void*>(out.data<float>()), 0, out.nbytes());
    return;
  }

  // Get batch dims
  auto batch_size_out = out.size() / (M * N);
  size_t matrix_stride_out = M * N;

  auto get_batch_dims = [](const auto& v) {
    return decltype(v){v.begin(), v.end() - 2};
  };

  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  auto batch_shape = get_batch_dims(out.shape());
  int batch_ndim = batch_shape.size();

  auto batch_shape_A = get_batch_dims(a.shape());
  auto batch_strides_A = get_batch_dims(a.strides());
  auto batch_shape_B = get_batch_dims(b.shape());
  auto batch_strides_B = get_batch_dims(b.strides());

  const uint32_t* lhs_indices_ptr = lhs_indices.data<uint32_t>();
  const uint32_t* rhs_indices_ptr = rhs_indices.data<uint32_t>();

  for (int i = 0; i < batch_size_out; i++) {
    // Get index
    uint32_t indx_A = lhs_indices_ptr[elem_to_loc(i, lhs_indices)];
    uint32_t indx_B = rhs_indices_ptr[elem_to_loc(i, rhs_indices)];

    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        1.0f, // alpha
        a.data<float>() + elem_to_loc(indx_A, batch_shape_A, batch_strides_A),
        lda,
        b.data<float>() + elem_to_loc(indx_B, batch_shape_B, batch_strides_B),
        ldb,
        0.0f, // beta
        out.data<float>() + matrix_stride_out * i,
        out.shape(-1) // ldc
    );
  }
}

} // namespace mlx::core
