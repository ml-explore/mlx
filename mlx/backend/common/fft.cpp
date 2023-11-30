// Copyright © 2023 Apple Inc.

#include <numeric>

#include "mlx/3rdparty/pocketfft.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"

namespace mlx::core {

void FFT::eval(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  std::vector<std::ptrdiff_t> strides_in(
      in.strides().begin(), in.strides().end());
  for (auto& s : strides_in) {
    s *= in.itemsize();
  }
  std::vector<std::ptrdiff_t> strides_out(
      out.strides().begin(), out.strides().end());
  for (auto& s : strides_out) {
    s *= out.itemsize();
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  std::vector<size_t> shape;
  if (out.dtype() == float32) {
    shape.insert(shape.end(), out.shape().begin(), out.shape().end());
  } else {
    shape.insert(shape.end(), in.shape().begin(), in.shape().end());
  }

  float scale = 1.0f;
  if (inverse_) {
    size_t nelem = std::accumulate(
        axes_.begin(), axes_.end(), 1, [&shape](auto x, auto y) {
          return x * shape[y];
        });
    scale /= nelem;
  }
  if (in.dtype() == complex64 && out.dtype() == complex64) {
    auto in_ptr =
        reinterpret_cast<const std::complex<float>*>(in.data<complex64_t>());
    auto out_ptr =
        reinterpret_cast<std::complex<float>*>(out.data<complex64_t>());
    pocketfft::c2c(
        shape,
        strides_in,
        strides_out,
        axes_,
        !inverse_,
        in_ptr,
        out_ptr,
        scale);
  } else if (in.dtype() == float32 && out.dtype() == complex64) {
    auto in_ptr = in.data<float>();
    auto out_ptr =
        reinterpret_cast<std::complex<float>*>(out.data<complex64_t>());
    pocketfft::r2c(
        shape,
        strides_in,
        strides_out,
        axes_,
        !inverse_,
        in_ptr,
        out_ptr,
        scale);
  } else if (in.dtype() == complex64 && out.dtype() == float32) {
    auto in_ptr =
        reinterpret_cast<const std::complex<float>*>(in.data<complex64_t>());
    auto out_ptr = out.data<float>();
    pocketfft::c2r(
        shape,
        strides_in,
        strides_out,
        axes_,
        !inverse_,
        in_ptr,
        out_ptr,
        scale);
  } else {
    throw std::runtime_error(
        "[FFT] Received unexpected input and output type combination.");
  }
}

} // namespace mlx::core
