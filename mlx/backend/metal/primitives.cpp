// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

static constexpr int METAL_MAX_INDEX_ARRAYS = 10;

void binary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_out = strides[2];

  std::ostringstream kname;
  switch (bopt) {
    case ScalarScalar:
      kname << "ss";
      break;
    case ScalarVector:
      kname << "sv";
      break;
    case VectorScalar:
      kname << "vs";
      break;
    case VectorVector:
      kname << "vv";
      break;
    case General:
      kname << "g";
      break;
  }
  kname << op << type_to_name(a);
  if (bopt == General && out.ndim() <= MAX_BINARY_SPECIALIZED_DIMS) {
    kname << "_" << shape.size();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel(kname.str());
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, a, 0);
  set_array_buffer(compute_encoder, b, 1);
  set_array_buffer(compute_encoder, out, 2);

  if (bopt == General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 3);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 3);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 4);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 6);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D grid of threads
    size_t nthreads = bopt == General ? out.size() : out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

void unary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  auto& in = inputs[0];
  bool contig = in.flags().contiguous;
  if (contig) {
    out.set_data(
        allocator::malloc_or_wait(in.data_size() * out.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);
  std::string tname = type_to_name(in);
  std::string opt_name = contig ? "v" : "g";
  auto kernel = d.get_kernel(opt_name + op + tname);

  size_t nthreads = contig ? in.data_size() : in.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);

  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);
  if (!contig) {
    compute_encoder->setBytes(in.shape().data(), in.ndim() * sizeof(int), 2);
    compute_encoder->setBytes(
        in.strides().data(), in.ndim() * sizeof(size_t), 3);
    int ndim = in.ndim();
    compute_encoder->setBytes(&ndim, sizeof(int), 4);
  }
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace

void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "abs");
}

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "add");
}

template <typename T>
void arange_set_scalars(T start, T next, MTL::ComputeCommandEncoder* enc) {
  enc->setBytes(&start, sizeof(T), 0);
  T step = next - start;
  enc->setBytes(&step, sizeof(T), 1);
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel("arange" + type_to_name(out));
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(
      std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  switch (out.dtype()) {
    case bool_: // unsupported
      throw std::runtime_error("[Arange::eval_gpu] Does not support bool");
    case uint8:
      arange_set_scalars<uint8_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint16:
      arange_set_scalars<uint16_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint32:
      arange_set_scalars<uint32_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint64:
      arange_set_scalars<uint64_t>(start_, start_ + step_, compute_encoder);
      break;
    case int8:
      arange_set_scalars<int8_t>(start_, start_ + step_, compute_encoder);
      break;
    case int16:
      arange_set_scalars<int16_t>(start_, start_ + step_, compute_encoder);
      break;
    case int32:
      arange_set_scalars<int32_t>(start_, start_ + step_, compute_encoder);
      break;
    case int64:
      arange_set_scalars<int64_t>(start_, start_ + step_, compute_encoder);
      break;
    case float16:
      arange_set_scalars<float16_t>(start_, start_ + step_, compute_encoder);
      break;
    case float32:
      arange_set_scalars<float>(start_, start_ + step_, compute_encoder);
      break;
    case bfloat16:
      arange_set_scalars<bfloat16_t>(start_, start_ + step_, compute_encoder);
      break;
    case complex64:
      throw std::runtime_error("[Arange::eval_gpu] Does not support complex64");
  }

  set_array_buffer(compute_encoder, out, 2);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
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

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string op_name;
  switch (reduce_type_) {
    case ArgReduce::ArgMin:
      op_name = "argmin_";
      break;
    case ArgReduce::ArgMax:
      op_name = "argmax_";
      break;
  }

  // Prepare the shapes, strides and axis arguments.
  std::vector<size_t> in_strides = in.strides();
  std::vector<int> shape = in.shape();
  std::vector<size_t> out_strides = out.strides();
  size_t axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];
  if (out_strides.size() == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  // ArgReduce
  int simd_size = 32;
  int n_reads = 4;
  auto compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name + type_to_name(in));
    NS::UInteger thread_group_size = std::min(
        (axis_size + n_reads - 1) / n_reads,
        kernel->maxTotalThreadsPerThreadgroup());
    // round up to the closest number divisible by simd_size
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    size_t n_threads = out.size() * thread_group_size;
    MTL::Size grid_dims = MTL::Size(n_threads, 1, 1);
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 2);
    compute_encoder->setBytes(in_strides.data(), ndim * sizeof(size_t), 3);
    compute_encoder->setBytes(out_strides.data(), ndim * sizeof(size_t), 4);
    compute_encoder->setBytes(&ndim, sizeof(size_t), 5);
    compute_encoder->setBytes(&axis_stride, sizeof(size_t), 6);
    compute_encoder->setBytes(&axis_size, sizeof(size_t), 7);
    compute_encoder->setThreadgroupMemoryLength(
        simd_size * (sizeof(uint32_t) + in.itemsize()), 0);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  CopyType ctype =
      inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis_));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis_] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, stream());
  }
}

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "cos");
}

void Cosh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "cosh");
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "div");
}

void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "rem");
}

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, equal_nan_ ? "naneq" : "eq");
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

void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "ge");
}

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "geq");
}

void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "le");
}

void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "leq");
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
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

void LogAddExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "lae");
}

void Maximum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "max");
}

void Minimum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "min");
}

void Floor::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "floor");
}

void Ceil::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "ceil");
}

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "mul");
}

void Negative::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "neg");
}

void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "neq");
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Inputs must be base input array and scalar val array
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  // Padding value must be a scalar
  assert(val.size() == 1);

  // Padding value, input and output must be of the same type
  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  // Fill output with val
  copy_gpu(val, out, CopyType::Scalar, stream());

  // Find offset for start of input values
  size_t data_offset = 0;
  for (int i = 0; i < axes_.size(); i++) {
    auto ax = axes_[i] < 0 ? out.ndim() + axes_[i] : axes_[i];
    data_offset += out.strides()[ax] * low_pad_size_[i];
  }

  // Extract slice from output where input will be pasted
  array out_slice(in.shape(), out.dtype(), nullptr, {});
  out_slice.copy_shared_buffer(
      out, out.strides(), out.flags(), out_slice.size(), data_offset);

  // Copy input values into the slice
  copy_gpu_inplace(in, out_slice, CopyType::GeneralGeneral, stream());
}

void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "pow");
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  size_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_per_key / 2;
  bool odd = out_per_key % 2;

  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string kname = keys.flags().row_contiguous ? "rbitsc" : "rbits";
  auto kernel = d.get_kernel(kname);

  // organize into grid nkeys x elem_per_key
  MTL::Size grid_dims = MTL::Size(num_keys, half_size + odd, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  auto nthreads = std::min(num_keys * (half_size + odd), thread_group_size);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, keys, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(&odd, sizeof(bool), 2);
  compute_encoder->setBytes(&bytes_per_key, sizeof(size_t), 3);

  if (!keys.flags().row_contiguous) {
    int ndim = keys.ndim();
    compute_encoder->setBytes(&ndim, sizeof(int), 4);
    compute_encoder->setBytes(
        keys.shape().data(), keys.ndim() * sizeof(int), 5);
    compute_encoder->setBytes(
        keys.strides().data(), keys.ndim() * sizeof(size_t), 6);
  }

  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (in.flags().row_contiguous) {
    auto flags = in.flags();
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.copy_shared_buffer(in, out.strides(), flags, in.data_size());
  } else {
    copy_gpu(in, out, CopyType::General);
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (not is_integral(in.dtype())) {
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

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "sub");
}

void Tan::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "tan");
}

void Tanh::eval_gpu(const std::vector<array>& inputs, array& out) {
  unary_op(inputs, out, "tanh");
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

} // namespace mlx::core
