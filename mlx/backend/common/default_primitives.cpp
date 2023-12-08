// Copyright © 2023 Apple Inc.

#include <cblas.h>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

#define DEFAULT(primitive)                                                 \
  void primitive::eval_cpu(const std::vector<array>& inputs, array& out) { \
    primitive::eval(inputs, out);                                          \
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
DEFAULT(Concatenate)
DEFAULT(Convolution)
DEFAULT(Copy)
DEFAULT(Cos)
DEFAULT(Cosh)
DEFAULT(Divide)
DEFAULT(Remainder)
DEFAULT(Equal)
DEFAULT(Erf)
DEFAULT(ErfInv)
DEFAULT(Exp)
DEFAULT(FFT)
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
DEFAULT(LogAddExp)
DEFAULT(Maximum)
DEFAULT(Minimum)
DEFAULT(Multiply)
DEFAULT(Negative)
DEFAULT(NotEqual)
DEFAULT(Pad)
DEFAULT(Partition)
DEFAULT(Power)
DEFAULT(RandomBits)
DEFAULT(Reduce)
DEFAULT(Reshape)
DEFAULT(Scan)
DEFAULT(Scatter)
DEFAULT(Sigmoid)
DEFAULT(Sign)
DEFAULT(Sin)
DEFAULT(Sinh)
DEFAULT(Slice)
DEFAULT(Softmax)
DEFAULT(Sort)
DEFAULT(Square)
DEFAULT(Sqrt)
DEFAULT(StopGradient)
DEFAULT(Subtract)
DEFAULT(Tan)
DEFAULT(Tanh)
DEFAULT(Transpose)

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[Matmul::eval_cpu] Currently only supports float32.");
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
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);
  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        1.0f, // alpha
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides()),
        lda,
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides()),
        ldb,
        0.0f, // beta
        out.data<float>() + M * N * i,
        out.shape(-1) // ldc
    );
  }
}

} // namespace mlx::core
