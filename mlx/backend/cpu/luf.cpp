// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void luf_impl(const array& a, array& lu, array& pivots, array& row_indices) {
  int M = a.shape(-2);
  int N = a.shape(-1);

  // Copy a into lu and make it col contiguous
  auto ndim = lu.ndim();
  auto flags = lu.flags();
  flags.col_contiguous = ndim == 2;
  flags.row_contiguous = false;
  flags.contiguous = true;
  auto strides = lu.strides();
  strides[ndim - 1] = M;
  strides[ndim - 2] = 1;
  lu.set_data(
      allocator::malloc_or_wait(lu.nbytes()), lu.nbytes(), strides, flags);
  copy_inplace(
      a, lu, a.shape(), a.strides(), strides, 0, 0, CopyType::GeneralGeneral);

  auto a_ptr = lu.data<T>();

  pivots.set_data(allocator::malloc_or_wait(pivots.nbytes()));
  row_indices.set_data(allocator::malloc_or_wait(row_indices.nbytes()));
  auto pivots_ptr = pivots.data<uint32_t>();
  auto row_indices_ptr = row_indices.data<uint32_t>();

  int info;
  size_t num_matrices = a.size() / (M * N);
  for (size_t i = 0; i < num_matrices; ++i) {
    // Compute LU factorization of A
    getrf<T>(
        /* m */ &M,
        /* n */ &N,
        /* a */ a_ptr,
        /* lda */ &M,
        /* ipiv */ reinterpret_cast<int*>(pivots_ptr),
        /* info */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[LUF::eval_cpu] sgetrf_ failed with code " << info
         << ((info > 0) ? " because matrix is singular"
                        : " because argument had an illegal value");
      throw std::runtime_error(ss.str());
    }

    // Subtract 1 to get 0-based index
    int j = 0;
    for (; j < pivots.shape(-1); ++j) {
      pivots_ptr[j]--;
      row_indices_ptr[j] = j;
    }
    for (; j < row_indices.shape(-1); ++j) {
      row_indices_ptr[j] = j;
    }
    for (int j = pivots.shape(-1) - 1; j >= 0; --j) {
      auto piv = pivots_ptr[j];
      auto t1 = row_indices_ptr[piv];
      auto t2 = row_indices_ptr[j];
      row_indices_ptr[j] = t1;
      row_indices_ptr[piv] = t2;
    }

    // Advance pointers to the next matrix
    a_ptr += M * N;
    pivots_ptr += pivots.shape(-1);
    row_indices_ptr += pivots.shape(-1);
  }
}

void LUF::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  switch (inputs[0].dtype()) {
    case float32:
      luf_impl<float>(inputs[0], outputs[0], outputs[1], outputs[2]);
      break;
    case float64:
      luf_impl<double>(inputs[0], outputs[0], outputs[1], outputs[2]);
      break;
    default:
      throw std::runtime_error(
          "[LUF::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
