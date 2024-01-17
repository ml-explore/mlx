// Copyright © 2023 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "axpby/axpby.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
array axpby(
    const array& x, // Input array x
    const array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  // Promote dtypes between x and y as needed
  auto promoted_dtype = promote_types(x.dtype(), y.dtype());

  // Upcast to float32 for non-floating point inputs x and y
  auto out_dtype = is_floating_point(promoted_dtype)
      ? promoted_dtype
      : promote_types(promoted_dtype, float32);

  // Cast x and y up to the determined dtype (on the same stream s)
  auto x_casted = astype(x, out_dtype, s);
  auto y_casted = astype(y, out_dtype, s);

  // Broadcast the shapes of x and y (on the same stream s)
  auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
  auto out_shape = broadcasted_inputs[0].shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return array(
      /* const std::vector<int>& shape = */ out_shape,
      /* Dtype dtype = */ out_dtype,
      /* std::unique_ptr<Primitive> primitive = */
      std::make_unique<Axpby>(to_stream(s), alpha, beta),
      /* const std::vector<array>& inputs = */ broadcasted_inputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void axpby_impl(
    const array& x,
    const array& y,
    array& out,
    float alpha_,
    float beta_) {
  // We only allocate memory when we are ready to fill the output
  // malloc_or_wait synchronously allocates available memory
  // There may be a wait executed here if the allocation is requested
  // under memory-pressured conditions
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Collect input and output data pointers
  const T* x_ptr = x.data<T>();
  const T* y_ptr = y.data<T>();
  T* out_ptr = out.data<T>();

  // Cast alpha and beta to the relevant types
  T alpha = static_cast<T>(alpha_);
  T beta = static_cast<T>(beta_);

  // Do the element-wise operation for each output
  for (size_t out_idx = 0; out_idx < out.size(); out_idx++) {
    // Map linear indices to offsets in x and y
    auto x_offset = elem_to_loc(out_idx, x.shape(), x.strides());
    auto y_offset = elem_to_loc(out_idx, y.shape(), y.strides());

    // We allocate the output to be contiguous and regularly strided
    // (defaults to row major) and hence it doesn't need additional mapping
    out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
  }
}

/** Fall back implementation for evaluation on CPU */
void Axpby::eval(
    const std::vector<array>& inputs,
    std::vector<array>& out_arr) {
  auto out = out_arr[0];
  // Check the inputs (registered in the op while constructing the out array)
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];

  // Dispatch to the correct dtype
  if (out.dtype() == float32) {
    return axpby_impl<float>(x, y, out, alpha_, beta_);
  } else if (out.dtype() == float16) {
    return axpby_impl<float16_t>(x, y, out, alpha_, beta_);
  } else if (out.dtype() == bfloat16) {
    return axpby_impl<bfloat16_t>(x, y, out, alpha_, beta_);
  } else if (out.dtype() == complex64) {
    return axpby_impl<complex64_t>(x, y, out, alpha_, beta_);
  } else {
    throw std::runtime_error(
        "Axpby is only supported for floating point types.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Accelerate Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef ACCELERATE_NEW_LAPACK

template <typename T>
void axpby_impl_accelerate(
    const array& x,
    const array& y,
    array& out,
    float alpha_,
    float beta_) {
  // Accelerate library provides catlas_saxpby which does
  // Y = (alpha * X) + (beta * Y) in place
  // To use it, we first copy the data in y over to the output array

  // This specialization requires both x and y be contiguous in the same mode
  // i.e: corresponding linear indices in both point to corresponding elements
  // The data in the output array is allocated to match the strides in y
  // such that x, y, and out are contiguous in the same mode and
  // no transposition is needed
  out.set_data(
      allocator::malloc_or_wait(y.data_size() * out.itemsize()),
      y.data_size(),
      y.strides(),
      y.flags());

  // We then copy over the elements using the contiguous vector specialization
  copy_inplace(y, out, CopyType::Vector);

  // Get x and y pointers for catlas_saxpby
  const T* x_ptr = x.data<T>();
  T* y_ptr = out.data<T>();

  T alpha = static_cast<T>(alpha_);
  T beta = static_cast<T>(beta_);

  // Call the inplace accelerate operator
  catlas_saxpby(
      /* N = */ out.size(),
      /* ALPHA = */ alpha,
      /* X = */ x_ptr,
      /* INCX = */ 1,
      /* BETA = */ beta,
      /* Y = */ y_ptr,
      /* INCY = */ 1);
}

/** Evaluate primitive on CPU using accelerate specializations */
void Axpby::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outarr) {
  auto out = outarr[0];
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];

  // Accelerate specialization for contiguous single precision float arrays
  if (out.dtype() == float32 &&
      ((x.flags().row_contiguous && y.flags().row_contiguous) ||
       (x.flags().col_contiguous && y.flags().col_contiguous))) {
    axpby_impl_accelerate<float>(x, y, out, alpha_, beta_);
    return;
  }

  // Fall back to common backend if specializations are not available
  eval(inputs, outarr);
}

#else // Accelerate not available

/** Evaluate primitive on CPU falling back to common backend */
void Axpby::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& out) {
  eval(inputs, out);
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

/** Evaluate primitive on GPU */
void Axpby::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outarr) {
  // Prepare inputs
  auto out = outarr[0];
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];

  // Each primitive carries the stream it should execute on
  // and each stream carries its device identifiers
  auto& s = stream();
  // We get the needed metal device using the stream
  auto& d = metal::device(s.device);

  // Prepare to specialize based on contiguity
  bool contiguous_kernel =
      (x.flags().row_contiguous && y.flags().row_contiguous) ||
      (x.flags().col_contiguous && y.flags().col_contiguous);

  // Allocate output memory with strides based on specialization
  if (contiguous_kernel) {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * out.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  // Resolve name of kernel (corresponds to axpby.metal)
  std::ostringstream kname;
  kname << "axpby_";
  kname << (contiguous_kernel ? "contiguous_" : "general_");
  kname << type_to_name(out);

  // Make sure the metal library is available and look for it
  // in the same folder as this executable if needed
  d.register_library("mlx_ext", metal::get_colocated_mtllib_path);

  // Make a kernel from this metal library
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");

  // Prepare to encode kernel
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  // Kernel parameters are registered with buffer indices corresponding to
  // those in the kernel declaration at axpby.metal
  int ndim = out.ndim();
  size_t nelem = out.size();

  // Encode input arrays to kernel
  set_array_buffer(compute_encoder, x, 0);
  set_array_buffer(compute_encoder, y, 1);

  // Encode output arrays to kernel
  set_array_buffer(compute_encoder, out, 2);

  // Encode alpha and beta
  compute_encoder->setBytes(&alpha_, sizeof(float), 3);
  compute_encoder->setBytes(&beta_, sizeof(float), 4);

  // Encode shape, strides and ndim if needed
  if (!contiguous_kernel) {
    compute_encoder->setBytes(x.shape().data(), ndim * sizeof(int), 5);
    compute_encoder->setBytes(x.strides().data(), ndim * sizeof(size_t), 6);
    compute_encoder->setBytes(y.strides().data(), ndim * sizeof(size_t), 7);
    compute_encoder->setBytes(&ndim, sizeof(int), 8);
  }

  // We launch 1 thread for each input and make sure that the number of
  // threads in any given threadgroup is not higher than the max allowed
  size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());

  // Fix the 3D size of each threadgroup (in terms of threads)
  MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);

  // Fix the 3D size of the launch grid (in terms of threads)
  MTL::Size grid_dims = MTL::Size(nelem, 1, 1);

  // Launch the grid with the given number of threads divided among
  // the given threadgroups
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

#else // Metal is not available

/** Fail evaluation on GPU */
void Axpby::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& out) {
  throw std::runtime_error("Axpby has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<array> Axpby::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // Forward mode diff that pushes along the tangents
  // The jvp transform on the primitive can built with ops
  // that are scheduled on the same stream as the primitive

  // If argnums = {0}, we only push along x in which case the
  // jvp is just the tangent scaled by alpha
  // Similarly, if argnums = {1}, the jvp is just the tangent
  // scaled by beta
  if (argnums.size() > 1) {
    auto scale = argnums[0] == 0 ? alpha_ : beta_;
    auto scale_arr = array(scale, tangents[0].dtype());
    return {multiply(scale_arr, tangents[0], stream())};
  }
  // If, argnums = {0, 1}, we take contributions from both
  // which gives us jvp = tangent_x * alpha + tangent_y * beta
  else {
    return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
  }
}

/** The vector-Jacobian product. */
std::vector<array> Axpby::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  // Reverse mode diff
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto scale = arg == 0 ? alpha_ : beta_;
    auto scale_arr = array(scale, cotangents[0].dtype());
    vjps.push_back(multiply(scale_arr, cotangents[0], stream()));
  }
  return vjps;
}

/** Vectorize primitive along given axis */
std::pair<std::vector<array>, std::vector<int>> Axpby::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("Axpby has no vmap implementation.");
}

/** Equivalence check **/
bool Axpby::is_equivalent(const Primitive& other) const {
  const Axpby& r_other = static_cast<const Axpby&>(other);
  return alpha_ == r_other.alpha_ && beta_ == r_other.beta_;
}

} // namespace mlx::core