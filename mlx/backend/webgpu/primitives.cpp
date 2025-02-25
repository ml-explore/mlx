// Copyright © 2023-2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#define UNARY_GPU(func, op)                                           \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    unary_op_gpu(inputs, out, op);                                    \
  }

#define BINARY_GPU(func, op)                                          \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    binary_op_gpu(inputs, out, op);                                   \
  }

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no GPU implementation.");     \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no GPU implementation.");    \
  }

namespace mlx::core {

namespace {

void reshape(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(webgpu::allocator().malloc_gpu(out));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op) {
  auto& in = inputs[0];
  if (in.size() == 0)
    return;
  if (in.flags().contiguous) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.move_shared_buffer(in);
    } else {
      out.set_data(
          webgpu::allocator().malloc_gpu(out, in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
    betann::UnaryOpContiguous(
        webgpu::device(out),
        op,
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        dtype_to_webgpu(in.dtype()),
        get_gpu_buffer(in),
        in.size());
  } else {
    out.set_data(webgpu::allocator().malloc_gpu(out));
    betann::UnaryOpGeneral(
        webgpu::device(out),
        op,
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        dtype_to_webgpu(in.dtype()),
        get_gpu_buffer(in),
        to_u32_vector(in.shape()),
        to_u32_vector(in.strides()));
  }
}

void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  bool b_donatable = is_donatable(b, out);
  bool a_donatable = is_donatable(a, out);
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      out.set_data(
          webgpu::allocator().malloc_gpu(out, out.itemsize()),
          1,
          a.strides(),
          a.flags());
      break;
    case BinaryOpType::ScalarVector:
      if (b_donatable) {
        out.move_shared_buffer(b);
      } else {
        out.set_data(
            webgpu::allocator().malloc_gpu(out, b.data_size() * out.itemsize()),
            b.data_size(),
            b.strides(),
            b.flags());
      }
      break;
    case BinaryOpType::VectorScalar:
      if (a_donatable) {
        out.move_shared_buffer(a);
      } else {
        out.set_data(
            webgpu::allocator().malloc_gpu(out, a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::VectorVector:
      if (a_donatable) {
        out.move_shared_buffer(a);
      } else if (b_donatable) {
        out.move_shared_buffer(b);
      } else {
        out.set_data(
            webgpu::allocator().malloc_gpu(out, a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::General:
      if (a_donatable && a.flags().row_contiguous && a.size() == out.size()) {
        out.move_shared_buffer(a);
      } else if (
          b_donatable && b.flags().row_contiguous && b.size() == out.size()) {
        out.move_shared_buffer(b);
      } else {
        out.set_data(webgpu::allocator().malloc_gpu(out));
      }
      break;
  }
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  if (bopt == BinaryOpType::General) {
    betann::BinaryOpGeneral(
        webgpu::device(out),
        op,
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        to_u32_vector(a.shape()),
        dtype_to_webgpu(a.dtype()),
        get_gpu_buffer(donate_a ? out : a),
        to_u32_vector(a.strides()),
        get_gpu_buffer(donate_b ? out : b),
        to_u32_vector(b.strides()));
  } else {
    betann::BinaryOpContiguous(
        webgpu::device(out),
        op,
        static_cast<betann::BinaryOpType>(bopt),
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        out.data_size(),
        dtype_to_webgpu(a.dtype()),
        get_gpu_buffer(donate_a ? out : a),
        get_gpu_buffer(donate_b ? out : b));
  }
}

void gpu_merge_sort(
    betann::Device& device,
    const array& in,
    array& out,
    int axis_,
    bool argsort) {
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  bool contiguous = in.flags().contiguous;
  auto check_strides = [](array x, int sort_stride) {
    int min_stride = *std::min_element(x.strides().begin(), x.strides().end());
    int max_stride = *std::max_element(x.strides().begin(), x.strides().end());
    return sort_stride == min_stride || sort_stride == max_stride;
  };
  contiguous &= check_strides(in, in.strides()[axis]);
  contiguous &= check_strides(out, out.strides()[axis]);

  if (in.shape(axis) > betann::SortBlockSize()) {
    throw std::runtime_error("Multi-blocks sort is not implemented.");
  } else {
    betann::SortBlock(
        device,
        axis,
        contiguous ? betann::SortInputType::Contiguous
                   : betann::SortInputType::General,
        argsort ? betann::SortResultType::Values
                : betann::SortResultType::Indices,
        get_gpu_buffer(out),
        to_u32_vector(out.strides()),
        dtype_to_webgpu(in.dtype()),
        get_gpu_buffer(in),
        to_u32_vector(in.shape()),
        to_u32_vector(in.strides()));
  }
}

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(webgpu::allocator().malloc_gpu(out));
  if (out.size() == 0)
    return;
  betann::ArrayRange(
      webgpu::device(out),
      static_cast<double>(start_),
      static_cast<double>(step_),
      dtype_to_webgpu(out.dtype()),
      get_gpu_buffer(out));
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  gpu_merge_sort(webgpu::device(out), inputs[0], out, axis_, true);
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  gpu_merge_sort(webgpu::device(out), inputs[0], out, axis_, true);
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  CopyType ctype =
      inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu(inputs, out, "bitwise_and");
      break;
    case BitwiseBinary::Or:
      binary_op_gpu(inputs, out, "bitwise_or");
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu(inputs, out, "bitwise_xor");
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu(inputs, out, "left_shift");
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu(inputs, out, "right_shift");
      break;
  }
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void BroadcastAxes::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Contiguous::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.flags().row_contiguous ||
      (allow_col_major_ && in.flags().col_contiguous)) {
    move_or_copy(in, out);
  } else {
    copy_gpu(in, out, CopyType::General);
  }
}

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void CustomTransforms::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Depends::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void ExpandDims::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto in = inputs[0];
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_gpu(in, out, ctype);
}

void Flatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (base_) {
    case Base::e:
      unary_op_gpu(inputs, out, "log");
      break;
    case Base::two:
      unary_op_gpu(inputs, out, "log2");
      break;
    case Base::ten:
      unary_op_gpu(inputs, out, "log10");
      break;
  }
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  betann::MatrixMultiply(
      webgpu::device(out),
      dtype_to_webgpu(out.dtype()),
      get_gpu_buffer(out),
      get_gpu_buffer(inputs[0]),
      to_u32_vector(inputs[0].shape()),
      to_u32_vector(inputs[0].strides()),
      get_gpu_buffer(inputs[1]),
      to_u32_vector(inputs[1].shape()),
      to_u32_vector(inputs[1].strides()));
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  gpu_merge_sort(webgpu::device(out), inputs[0], out, axis_, false);
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Align to 4 bytes.
  out.set_data(webgpu::allocator().malloc_gpu(out, (out.nbytes() + 3) & ~3));
  if (out.size() == 0)
    return;
  auto& keys = inputs[0];
  if (keys.flags().row_contiguous) {
    betann::RandomBitsContiguous(
        webgpu::device(out),
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        out.size(),
        get_gpu_buffer(keys),
        keys.size());
  } else {
    betann::RandomBitsGeneral(
        webgpu::device(out),
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        out.size(),
        get_gpu_buffer(keys),
        to_u32_vector(keys.shape()),
        to_u32_vector(keys.strides()));
  }
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  array in = inputs[0];
  ReductionPlan plan = get_reduction_plan(in, axes_);
  betann::ReductionPlanType plan_type;
  switch (plan.type) {
    case ContiguousAllReduce:
      plan_type = betann::ReductionPlanType::ReduceAll;
      break;
    case ContiguousReduce:
    case GeneralContiguousReduce:
      plan_type = betann::ReductionPlanType::ReduceRow;
      break;
    case ContiguousStridedReduce:
    case GeneralStridedReduce:
      plan_type = betann::ReductionPlanType::ReduceCol;
      break;
    default:
      plan_type = betann::ReductionPlanType::ReduceGeneral;
      break;
  }
  betann::Reduce(
      webgpu::device(out),
      {plan_type, to_u32_vector(plan.shape), to_u32_vector(plan.strides)},
      static_cast<betann::ReduceType>(reduce_type_),
      dtype_to_webgpu(out.dtype()),
      get_gpu_buffer(out),
      out.size(),
      dtype_to_webgpu(in.dtype()),
      get_gpu_buffer(in),
      in.size(),
      to_u32_vector(in.shape()),
      to_u32_vector(in.strides()),
      to_u32_vector(axes_));
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  const auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu(inputs, out, "round");
  } else {
    move_or_copy(in, out);
  }
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(webgpu::allocator().malloc_gpu(out));
  gpu_merge_sort(webgpu::device(out), inputs[0], out, axis_, false);
}

void Split::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Squeeze::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Unflatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out, stream());
}

void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto ibytes = size_of(in.dtype());
  auto obytes = size_of(out.dtype());
  if (ibytes == obytes || (obytes < ibytes && in.strides().back() == 1) ||
      in.flags().row_contiguous) {
    auto strides = in.strides();
    for (int i = 0; i < static_cast<int>(strides.size()) - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    move_or_copy(
        in, out, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(in.shape(), in.dtype(), nullptr, {});
    tmp.set_data(webgpu::allocator().malloc_gpu(tmp));
    copy_gpu_inplace(in, tmp, CopyType::General, stream());

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.move_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

UNARY_GPU(Abs, "abs")
UNARY_GPU(ArcCos, "acos")
UNARY_GPU(ArcCosh, "acosh")
UNARY_GPU(ArcSin, "asin")
UNARY_GPU(ArcSinh, "asinh")
UNARY_GPU(ArcTan, "atan")
UNARY_GPU(ArcTanh, "atanh")
UNARY_GPU(BitwiseInvert, "bitwise_invert")
UNARY_GPU(Cos, "cos")
UNARY_GPU(Cosh, "cosh")
UNARY_GPU(Erf, "erf")
UNARY_GPU(ErfInv, "erf_inv")
UNARY_GPU(Exp, "exp")
UNARY_GPU(Expm1, "expm1")
UNARY_GPU(Log1p, "log1p")
UNARY_GPU(LogicalNot, "logical_not")
UNARY_GPU(Floor, "floor")
UNARY_GPU(Ceil, "ceil")
UNARY_GPU(Negative, "negative")
UNARY_GPU(Sigmoid, "sigmoid")
UNARY_GPU(Sign, "sign")
UNARY_GPU(Sin, "sin")
UNARY_GPU(Sinh, "sinh")
UNARY_GPU(Square, "square")
UNARY_GPU(Sqrt, "sqrt")
UNARY_GPU(Tan, "tan")
UNARY_GPU(Tanh, "tanh")

BINARY_GPU(Add, "add")
BINARY_GPU(ArcTan2, "arc_tan2")
BINARY_GPU(Divide, "divide")
BINARY_GPU(Remainder, "remainder")
BINARY_GPU(Equal, "equal")
BINARY_GPU(Greater, "greater")
BINARY_GPU(GreaterEqual, "greater_equal")
BINARY_GPU(Less, "less")
BINARY_GPU(LessEqual, "less_equal")
BINARY_GPU(LogicalAnd, "logical_and")
BINARY_GPU(LogicalOr, "logical_or")
BINARY_GPU(LogAddExp, "log_add_exp")
BINARY_GPU(Maximum, "maximum")
BINARY_GPU(Minimum, "minimum")
BINARY_GPU(Multiply, "multiply")
BINARY_GPU(NotEqual, "not_equal")
BINARY_GPU(Power, "power")
BINARY_GPU(Subtract, "subtract")

NO_GPU(AddMM)
NO_GPU(ArgReduce)
NO_GPU(BlockMaskedMM)
NO_GPU_MULTI(Compiled)
NO_GPU(Concatenate)
NO_GPU(Conjugate)
NO_GPU(Convolution)
NO_GPU_MULTI(DivMod)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(NumberOfElements)
NO_GPU(FFT)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Load)
NO_GPU_MULTI(LUF)
NO_GPU(Pad)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(Real)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(Slice)
NO_GPU(SliceUpdate)
NO_GPU(Softmax)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)

namespace fast {
NO_GPU_MULTI(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_MULTI(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_MULTI(RoPE)
NO_GPU(ScaledDotProductAttention)
NO_GPU_MULTI(AffineQuantize)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace mlx::core
