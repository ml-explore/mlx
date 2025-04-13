// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    unary_op_gpu(inputs, out, get_primitive_string(this));            \
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

  auto maybe_collapse = [contig, &in, &out]() {
    if (!contig) {
      return collapse_contiguous_dims(in);
    } else {
      return std::make_pair(Shape{}, Strides{});
    }
  };
  auto [shape, strides] = maybe_collapse();
  int ndim = shape.size();
  bool large;
  if (!contig) {
    large = in.data_size() > INT32_MAX || out.size() > INT32_MAX;
  } else {
    large = in.data_size() > UINT32_MAX;
  }
  int work_per_thread;
  std::string kernel_name;
  if (contig) {
    work_per_thread = get_work_per_thread(in.dtype());
    kernel_name = (large ? "v2" : "v");
  } else {
    work_per_thread = large ? 4 : 1;
    kernel_name = "gn" + std::to_string(work_per_thread);
    if (large) {
      kernel_name += "large";
    }
  }
  concatenate(kernel_name, "_", op, type_to_name(in), type_to_name(out));
  auto kernel = get_unary_kernel(d, kernel_name, in.dtype(), out.dtype(), op);

  auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  if (!contig) {
    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    compute_encoder.set_vector_bytes(shape, 2);
    compute_encoder.set_vector_bytes(strides, 3);
    compute_encoder.set_bytes(ndim, 4);
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::unary] Must use 1024 sized block");
    }
    dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    size_t nthreads = ceildiv(in.data_size(), work_per_thread);
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }

    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    MTL::Size grid_dims;
    if (large) {
      compute_encoder.set_bytes<int64_t>(in.data_size(), 2);
      grid_dims = get_2d_grid_dims(out.shape(), out.strides(), work_per_thread);
    } else {
      compute_encoder.set_bytes<int>(in.data_size(), 2);
      grid_dims = MTL::Size(nthreads, 1, 1);
    }
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s) {
  set_unary_output_data(inputs[0], out);
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
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Floor)
UNARY_GPU(Ceil)
UNARY_GPU(Negative)
UNARY_GPU(Real)
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
      unary_op_gpu(inputs, out, get_primitive_string(this));
      break;
    case Base::two:
      unary_op_gpu(inputs, out, get_primitive_string(this));
      break;
    case Base::ten:
      unary_op_gpu(inputs, out, get_primitive_string(this));
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu(inputs, out, get_primitive_string(this));
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

} // namespace mlx::core
