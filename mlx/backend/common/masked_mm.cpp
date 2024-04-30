// Copyright © 2024 Apple Inc.

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <cstring>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
inline void mask_matrix(
    T* data,
    const bool* mask,
    int block_size,
    const int X,
    const int Y,
    const size_t X_data_str,
    const size_t Y_data_str,
    const size_t X_mask_str,
    const size_t Y_mask_str) {
  int tX = (X + block_size - 1) / block_size;
  int tY = (Y + block_size - 1) / block_size;

  for (int i = 0; i < tX; i++) {
    for (int j = 0; j < tY; j++) {
      bool do_mask = mask[i * X_mask_str + j * Y_mask_str];
      if (!do_mask) {
        int loc_x = i * block_size;
        int loc_y = j * block_size;
        T* data_block = data + loc_x * X_data_str + loc_y * Y_data_str;

        int size_x = std::min(block_size, X - loc_x);
        int size_y = std::min(block_size, Y - loc_y);
        for (int ii = 0; ii < size_x; ii++) {
          for (int jj = 0; jj < size_y; jj++) {
            data_block[ii * X_data_str + jj * Y_data_str] = T(0.);
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
  auto& out_mask = inputs[2];

  auto check_transpose = [](const array& arr, bool do_copy) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      if (do_copy) {
        array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
        copy(arr, arr_copy, CopyType::Vector);
        return std::make_tuple(false, stx, arr_copy);
      }
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      if (do_copy) {
        array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
        copy(arr, arr_copy, CopyType::Vector);
        return std::make_tuple(true, sty, arr_copy);
      }
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  bool has_op_mask = inputs.size() > 3;
  auto [a_transposed, lda, a] = check_transpose(a_pre, has_op_mask);
  auto [b_transposed, ldb, b] = check_transpose(b_pre, has_op_mask);

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
    const bool* mask_ptr = mask.data<bool>() +
        elem_to_loc(mask.shape(-1) * mask.shape(-2) * batch_idx,
                    mask.shape(),
                    mask.strides());

    size_t X_mask_str = mask.strides()[mask.ndim() - 2];
    size_t Y_mask_str = mask.strides()[mask.ndim() - 1];

    return mask_matrix(
        data,
        mask_ptr,
        block_size,
        X,
        Y,
        X_data_str,
        Y_data_str,
        X_mask_str,
        Y_mask_str);
  };

  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    // Adjust pointer
    float* ai =
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides());
    float* bi =
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides());
    float* ci = out.data<float>() + M * N * i;

    // Zero out blocks in a and b if needed
    if (has_op_mask) {
      auto& a_mask = inputs[3];
      mask_array(
          a_mask,
          ai,
          block_size_,
          i,
          M,
          K,
          a_transposed ? 1 : lda,
          a_transposed ? lda : 1);

      auto& b_mask = inputs[4];
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
    mask_array(out_mask, ci, block_size_, i, M, N, N, 1);
  }
}

void BlockSparseMM::eval(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[BlockSparseMM::eval] Currently only supports float32.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
}

} // namespace mlx::core