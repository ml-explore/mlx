// Copyright © 2023-2024 Apple Inc.

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

#define DEFAULT(primitive)                                                 \
  void primitive::eval_cpu(const std::vector<array>& inputs, array& out) { \
    primitive::eval(inputs, out);                                          \
  }

#define DEFAULT_MULTI(primitive)                                       \
  void primitive::eval_cpu(                                            \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    primitive::eval(inputs, outputs);                                  \
  }

namespace mlx::core {

DEFAULT(Abs)
DEFAULT(Add)
DEFAULT(Arange)
DEFAULT(ArcCos)
DEFAULT(ArcCosh)
DEFAULT(ArcSin)
DEFAULT(ArcSinh)
DEFAULT(ArcTan)
DEFAULT(ArcTanh)
DEFAULT(ArgPartition)
DEFAULT(ArgReduce)
DEFAULT(ArgSort)
DEFAULT(AsType)
DEFAULT(AsStrided)
DEFAULT(Broadcast)
DEFAULT_MULTI(DivMod)
DEFAULT(Ceil)
DEFAULT(Concatenate)
DEFAULT(Convolution)
DEFAULT(Copy)
DEFAULT(Cos)
DEFAULT(Cosh)
DEFAULT_MULTI(CustomVJP)
DEFAULT_MULTI(Depends)
DEFAULT(Divide)
DEFAULT(NumberOfElements)
DEFAULT(Remainder)
DEFAULT(Equal)
DEFAULT(Erf)
DEFAULT(ErfInv)
DEFAULT(Exp)
DEFAULT(Expm1)
DEFAULT(FFT)
DEFAULT(Floor)
DEFAULT(Full)
DEFAULT(Gather)
DEFAULT(Greater)
DEFAULT(GreaterEqual)
DEFAULT(Less)
DEFAULT(LessEqual)
DEFAULT(Load)
DEFAULT(Log)
DEFAULT(Log1p)
DEFAULT(LogicalNot)
DEFAULT(LogicalAnd)
DEFAULT(LogicalOr)
DEFAULT(LogAddExp)
DEFAULT(Maximum)
DEFAULT(Minimum)
DEFAULT(Multiply)
DEFAULT(Negative)
DEFAULT(NotEqual)
DEFAULT(Pad)
DEFAULT(Partition)
DEFAULT(Power)
DEFAULT_MULTI(QRF)
DEFAULT(QuantizedMatmul)
DEFAULT(RandomBits)
DEFAULT(Reduce)
DEFAULT(Reshape)
DEFAULT(Round)
DEFAULT(Scan)
DEFAULT(Scatter)
DEFAULT(Select)
DEFAULT(Sigmoid)
DEFAULT(Sign)
DEFAULT(Sin)
DEFAULT(Sinh)
DEFAULT(Slice)
DEFAULT(SliceUpdate)
DEFAULT(Softmax)
DEFAULT(Sort)
DEFAULT_MULTI(Split)
DEFAULT(Square)
DEFAULT(Sqrt)
DEFAULT(StopGradient)
DEFAULT(Subtract)
DEFAULT_MULTI(SVD)
DEFAULT(Tan)
DEFAULT(Tanh)
DEFAULT(Transpose)
DEFAULT(Inverse)

namespace {

inline void matmul_common_general(
    const array& a_pre,
    const array& b_pre,
    array& out,
    float alpha = 1.0f,
    float beta = 0.0f) {
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
      size_t stx = arr.shape(-1);
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

  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha, // alpha
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides()),
        lda,
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides()),
        ldb,
        beta, // beta
        out.data<float>() + M * N * i,
        out.shape(-1) // ldc
    );
  }
}

template <typename T>
inline void mask_matrix(
    T* data,
    const bool* mask,
    int tile_size,
    const int X,
    const int Y,
    const size_t X_data_str,
    const size_t Y_data_str,
    const size_t X_mask_str,
    const size_t Y_mask_str) {
  int tX = (X + tile_size - 1) / tile_size;
  int tY = (Y + tile_size - 1) / tile_size;

  for (int i = 0; i < tX; i++) {
    for (int j = 0; j < tY; j++) {
      bool do_mask = mask[i * X_mask_str + j * Y_mask_str];
      if (!do_mask) {
        int loc_x = i * tile_size;
        int loc_y = j * tile_size;
        T* data_block = data + loc_x * X_data_str + loc_y * Y_data_str;

        int size_x = std::min(tile_size, X - loc_x);
        int size_y = std::min(tile_size, Y - loc_y);
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

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[Matmul::eval_cpu] Currently only supports float32.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  return matmul_common_general(inputs[0], inputs[1], out);
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[AddMM::eval_cpu] Currently only supports float32.");
  }

  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1 ? CopyType::Scalar : CopyType::General;
  copy(c, out, ctype);

  return matmul_common_general(inputs[0], inputs[1], out, alpha_, beta_);
}

void TileMaskedMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[TileMaskedMM::eval_cpu] Currently only supports float32.");
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
                       int tile_size,
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
        tile_size,
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
          tile_size_,
          i,
          M,
          K,
          a_transposed ? 1 : lda,
          a_transposed ? lda : 1);

      auto& b_mask = inputs[4];
      mask_array(
          b_mask,
          bi,
          tile_size_,
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
    mask_array(out_mask, ci, tile_size_, i, M, N, N, 1);
  }
}
} // namespace mlx::core
