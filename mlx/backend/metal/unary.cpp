// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <cassert>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr std::string_view unary_kernels = R"(
[[kernel]] void {0}_v(
    device const {1}* in,
    device {2}* out,
    uint index [[thread_position_in_grid]]) {{
  out[index] = {3}()(in[index]);
}}

[[kernel]] void {0}_g(
    device const {1}* in,
    device {2}* out,
    device const int* in_shape,
    device const size_t* in_strides,
    device const int& ndim,
    uint index [[thread_position_in_grid]]) {{
  auto idx = elem_to_loc(index, in_shape, in_strides, ndim);
  out[index] = {3}()(in[idx]);
}}
)";

void unary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
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
  if (in.size() == 0) {
    return;
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);
  std::string lib_name = op + type_to_name(in);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream op_t;
    out.primitive().print(op_t);
    std::ostringstream kernel_source;
    kernel_source << metal::get_kernel_preamble() << std::endl;
    kernel_source << fmt::format(
        unary_kernels,
        lib_name,
        get_type_string(in.dtype()),
        get_type_string(out.dtype()),
        op_t.str());
    lib = d.get_library(lib_name, kernel_source.str());
  }
  auto kernel = d.get_kernel(lib_name + (contig ? "_v" : "_g"), lib);

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
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "abs");
}

void ArcCos::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arccos");
}

void ArcCosh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arccosh");
}

void ArcSin::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arcsin");
}

void ArcSinh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arcsinh");
}

void ArcTan::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arctan");
}

void ArcTanh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "arctanh");
}

void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "cos");
}

void Cosh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "cosh");
}

void Erf::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "erf");
}

void ErfInv::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "erfinv");
}

void Exp::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "exp");
}

void Expm1::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "expm1");
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (base_) {
    case Base::e:
      unary_op(inputs, out, "log");
      break;
    case Base::two:
      unary_op(inputs, out, "log2");
      break;
    case Base::ten:
      unary_op(inputs, out, "log10");
      break;
  }
}

void Log1p::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "log1p");
}

void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "lnot");
}

void Floor::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "floor");
}

void Ceil::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "ceil");
}

void Negative::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "neg");
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_op(inputs, out, "round");
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sigmoid::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "sigmoid");
}

void Sign::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "sign");
}

void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "sin");
}

void Sinh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "sinh");
}

void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "square");
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (recip_) {
    unary_op(inputs, out, "rsqrt");
  } else {
    unary_op(inputs, out, "sqrt");
  }
}

void Tan::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "tan");
}

void Tanh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "tanh");
}

} // namespace mlx::core
