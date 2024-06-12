// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    unary_op_gpu(inputs, out, op_name());                             \
  }

namespace mlx::core {

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s) {
  auto& in = inputs[0];
  bool contig = in.flags().contiguous;
  if (in.size() == 0) {
    return;
  }

  auto& d = metal::device(s.device);

  std::string kernel_name = (contig ? "v" : "g") + op + type_to_name(out);
  auto kernel = get_unary_kernel(d, kernel_name, out.dtype(), op);

  size_t nthreads = contig ? in.data_size() : in.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(
      in.data_shared_ptr() == nullptr ? out : in, 0);
  compute_encoder.set_output_array(out, 1);
  if (!contig) {
    compute_encoder->setBytes(in.shape().data(), in.ndim() * sizeof(int), 2);
    compute_encoder->setBytes(
        in.strides().data(), in.ndim() * sizeof(size_t), 3);
    int ndim = in.ndim();
    compute_encoder->setBytes(&ndim, sizeof(int), 4);
  }
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s) {
  auto& in = inputs[0];
  bool contig = in.flags().contiguous;
  if (contig) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.move_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc_or_wait(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }
  unary_op_gpu_inplace(inputs, out, op, s);
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  auto& s = out.primitive().stream();
  unary_op_gpu(inputs, out, op, s);
}

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Floor)
UNARY_GPU(Ceil)
UNARY_GPU(Negative)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Sqrt)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (base_) {
    case Base::e:
      unary_op_gpu(inputs, out, op_name());
      break;
    case Base::two:
      unary_op_gpu(inputs, out, op_name());
      break;
    case Base::ten:
      unary_op_gpu(inputs, out, op_name());
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu(inputs, out, op_name());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

} // namespace mlx::core
