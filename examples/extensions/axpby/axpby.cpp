// Copyright © 2023-2025 Apple Inc.

#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "axpby/axpby.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace my_ext {

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
mx::array axpby(
    const mx::array& x, // Input mx::array x
    const mx::array& y, // Input mx::array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  // Promote dtypes between x and y as needed
  auto promoted_dtype = promote_types(x.dtype(), y.dtype());

  // Upcast to float32 for non-floating point inputs x and y
  auto out_dtype = mx::issubdtype(promoted_dtype, mx::float32)
      ? promoted_dtype
      : promote_types(promoted_dtype, mx::float32);

  // Cast x and y up to the determined dtype (on the same stream s)
  auto x_casted = mx::astype(x, out_dtype, s);
  auto y_casted = mx::astype(y, out_dtype, s);

  // Broadcast the shapes of x and y (on the same stream s)
  auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
  auto out_shape = broadcasted_inputs[0].shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return mx::array(
      /* const mx::Shape& shape = */ out_shape,
      /* mx::Dtype dtype = */ out_dtype,
      /* std::shared_ptr<mx::Primitive> primitive = */
      std::make_shared<Axpby>(to_stream(s), alpha, beta),
      /* const std::vector<mx::array>& inputs = */ broadcasted_inputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void axpby_impl(
    const mx::array& x,
    const mx::array& y,
    mx::array& out,
    float alpha_,
    float beta_,
    mx::Stream stream) {
  // Allocate the output with `malloc_or_wait` which synchronously allocates
  // memory, potentially waiting if the system is under memory pressure
  out.set_data(mx::allocator::malloc_or_wait(out.nbytes()));

  // Get the CPU command encoder and register input and output arrays
  auto& encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(y);
  encoder.set_output_array(out);

  // Launch the CPU kernel
  encoder.dispatch([x_ptr = x.data<T>(),
                    y_ptr = y.data<T>(),
                    out_ptr = out.data<T>(),
                    size = out.size(),
                    shape = out.shape(),
                    x_strides = x.strides(),
                    y_strides = y.strides(),
                    alpha_,
                    beta_]() {
    // Cast alpha and beta to the relevant types
    T alpha = static_cast<T>(alpha_);
    T beta = static_cast<T>(beta_);

    // Do the element-wise operation for each output
    for (size_t out_idx = 0; out_idx < size; out_idx++) {
      // Map linear indices to offsets in x and y
      auto x_offset = mx::elem_to_loc(out_idx, shape, x_strides);
      auto y_offset = mx::elem_to_loc(out_idx, shape, y_strides);

      // We allocate the output to be contiguous and regularly strided
      // (defaults to row major) and hence it doesn't need additional mapping
      out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
    }
  });
}

void Axpby::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Dispatch to the correct dtype
  if (out.dtype() == mx::float32) {
    return axpby_impl<float>(x, y, out, alpha_, beta_, stream());
  } else if (out.dtype() == mx::float16) {
    return axpby_impl<mx::float16_t>(x, y, out, alpha_, beta_, stream());
  } else if (out.dtype() == mx::bfloat16) {
    return axpby_impl<mx::bfloat16_t>(x, y, out, alpha_, beta_, stream());
  } else if (out.dtype() == mx::complex64) {
    return axpby_impl<mx::complex64_t>(x, y, out, alpha_, beta_, stream());
  } else {
    throw std::runtime_error(
        "Axpby is only supported for floating point types.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

/** Evaluate primitive on GPU */
void Axpby::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // Prepare inputs
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Each primitive carries the stream it should execute on
  // and each stream carries its device identifiers
  auto& s = stream();
  // We get the needed metal device using the stream
  auto& d = mx::metal::device(s.device);

  // Prepare to specialize based on contiguity
  bool contiguous_kernel =
      (x.flags().row_contiguous && y.flags().row_contiguous) ||
      (x.flags().col_contiguous && y.flags().col_contiguous);

  // Allocate output memory with strides based on specialization
  if (contiguous_kernel) {
    out.set_data(
        mx::allocator::malloc_or_wait(x.data_size() * out.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  } else {
    out.set_data(mx::allocator::malloc_or_wait(out.nbytes()));
  }

  // Resolve name of kernel (corresponds to axpby.metal)
  std::ostringstream kname;
  kname << "axpby_";
  kname << (contiguous_kernel ? "contiguous_" : "general_");
  kname << type_to_name(out);

  // Make sure the metal library is available
  d.register_library("mlx_ext");

  // Make a kernel from this metal library
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");

  // Prepare to encode kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Kernel parameters are registered with buffer indices corresponding to
  // those in the kernel declaration at axpby.metal
  int ndim = out.ndim();
  size_t nelem = out.size();

  // Encode input arrays to kernel
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(y, 1);

  // Encode output arrays to kernel
  compute_encoder.set_output_array(out, 2);

  // Encode alpha and beta
  compute_encoder.set_bytes(alpha_, 3);
  compute_encoder.set_bytes(beta_, 4);

  // Encode shape, strides and ndim if needed
  if (!contiguous_kernel) {
    compute_encoder.set_vector_bytes(x.shape(), 5);
    compute_encoder.set_vector_bytes(x.strides(), 6);
    compute_encoder.set_vector_bytes(y.strides(), 7);
    compute_encoder.set_bytes(ndim, 8);
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
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else // Metal is not available

/** Fail evaluation on GPU */
void Axpby::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& out) {
  throw std::runtime_error("Axpby has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<mx::array> Axpby::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
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
    auto scale_arr = mx::array(scale, tangents[0].dtype());
    return {mx::multiply(scale_arr, tangents[0], stream())};
  }
  // If, argnums = {0, 1}, we take contributions from both
  // which gives us jvp = tangent_x * alpha + tangent_y * beta
  else {
    return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
  }
}

/** The vector-Jacobian product. */
std::vector<mx::array> Axpby::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  // Reverse mode diff
  std::vector<mx::array> vjps;
  for (auto arg : argnums) {
    auto scale = arg == 0 ? alpha_ : beta_;
    auto scale_arr = mx::array(scale, cotangents[0].dtype());
    vjps.push_back(mx::multiply(scale_arr, cotangents[0], stream()));
  }
  return vjps;
}

/** Vectorize primitive along given axis */
std::pair<std::vector<mx::array>, std::vector<int>> Axpby::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("Axpby has no vmap implementation.");
}

/** Equivalence check **/
bool Axpby::is_equivalent(const Primitive& other) const {
  const Axpby& r_other = static_cast<const Axpby&>(other);
  return alpha_ == r_other.alpha_ && beta_ == r_other.beta_;
}

} // namespace my_ext
