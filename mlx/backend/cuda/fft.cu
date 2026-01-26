// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cufft.h>
#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <numeric>
#include <sstream>

namespace mlx::core {

namespace {

void check_cufft_error(cufftResult result, const char* msg) {
  if (result != CUFFT_SUCCESS) {
    std::ostringstream oss;
    oss << "[cuFFT] " << msg << " failed with error code " << result;
    throw std::runtime_error(oss.str());
  }
}

#define CHECK_CUFFT_ERROR(x) check_cufft_error(x, #x)

// Functor for scaling complex values
struct ScaleComplex {
  float scale;
  __host__ __device__ cufftComplex operator()(const cufftComplex& x) const {
    return cufftComplex{x.x * scale, x.y * scale};
  }
};

// Functor for scaling real values
struct ScaleReal {
  float scale;
  __host__ __device__ float operator()(float x) const {
    return x * scale;
  }
};

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("FFT::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto& in = inputs[0];

  // Make input contiguous if needed
  array in_contiguous = in;
  if (!in.flags().row_contiguous) {
    in_contiguous = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in_contiguous);
  }

  // Allocate output
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  if (out.size() == 0) {
    return;
  }

  encoder.set_input_array(in_contiguous);
  encoder.set_output_array(out);

  // cuFFT requires committing any pending work before running
  // since it uses its own stream management internally
  encoder.commit();

  // Determine FFT type
  bool is_real_to_complex =
      (in_contiguous.dtype() == float32 && out.dtype() == complex64);
  bool is_complex_to_real =
      (in_contiguous.dtype() == complex64 && out.dtype() == float32);
  bool is_complex_to_complex =
      (in_contiguous.dtype() == complex64 && out.dtype() == complex64);

  if (!is_real_to_complex && !is_complex_to_real && !is_complex_to_complex) {
    std::ostringstream oss;
    oss << "[FFT] Unsupported dtype combination: input "
        << dtype_to_string(in_contiguous.dtype()) << ", output "
        << dtype_to_string(out.dtype());
    throw std::runtime_error(oss.str());
  }

  // Determine the shape for FFT
  std::vector<int64_t> fft_shape;
  if (out.dtype() == float32) {
    // irfft: output shape determines FFT size
    fft_shape.insert(fft_shape.end(), out.shape().begin(), out.shape().end());
  } else {
    // fft/rfft: input shape determines FFT size
    fft_shape.insert(
        fft_shape.end(),
        in_contiguous.shape().begin(),
        in_contiguous.shape().end());
  }

  // Compute batch size and FFT dimensions
  int rank = static_cast<int>(axes_.size());
  std::vector<int> n(rank);
  int ndim = in_contiguous.ndim();

  // Get FFT sizes along each axis
  for (int i = 0; i < rank; ++i) {
    n[i] = static_cast<int>(fft_shape[axes_[i]]);
  }

  // Create cuFFT plan
  cufftHandle plan;
  cudaStream_t cuda_stream = encoder.stream();

  // Check if axes are the last dimensions in order
  bool standard_layout = true;
  for (int i = 0; i < rank; ++i) {
    if (static_cast<int>(axes_[i]) != ndim - rank + i) {
      standard_layout = false;
      break;
    }
  }

  if (rank == 1) {
    // 1D FFT case - can handle arbitrary axis using strides
    int fft_size = n[0];
    int axis = static_cast<int>(axes_[0]);

    // For cuFFT with strided data:
    // - istride/ostride: distance between consecutive elements along FFT axis
    // - idist/odist: distance between first elements of consecutive batches
    // - batch: number of FFTs to compute

    // For row-major storage, stride along axis i is:
    // product of dimensions after axis i
    int istride = 1;
    for (int i = axis + 1; i < ndim; ++i) {
      istride *= in_contiguous.shape(i);
    }
    int ostride = 1;
    for (int i = axis + 1; i < ndim; ++i) {
      ostride *= out.shape(i);
    }

    // Batch count is product of dimensions not on FFT axis
    int batch = 1;
    for (int i = 0; i < ndim; ++i) {
      if (i != axis) {
        batch *= in_contiguous.shape(i);
      }
    }

    // For strided FFT, we need to loop over the batches manually if
    // they're not contiguous. cuFFT's idist/odist assume contiguous batches.
    // For arbitrary axis, the batches are at distance 1 (adjacent elements
    // along the innermost dimension) or at larger distances.

    // If axis is the last dimension, batches are at stride = fft_size
    // If axis is not the last dimension, we have strided access

    // The simplest case: treat all dimensions after axis as inner batches
    // and all dimensions before axis as outer batches

    // For cuFFT, idist is the distance between consecutive FFT sequences
    // For axis being the last dimension: idist = fft_size (for C2C)
    // For axis NOT being the last dimension: idist = 1 (consecutive batches)

    int idist, odist;
    int inembed_val, onembed_val;

    if (standard_layout) {
      // Axis is the last dimension - standard case
      if (is_real_to_complex) {
        idist = fft_size;
        odist = fft_size / 2 + 1;
      } else if (is_complex_to_real) {
        idist = fft_size / 2 + 1;
        odist = fft_size;
      } else {
        idist = fft_size;
        odist = fft_size;
      }
      inembed_val = fft_size;
      onembed_val = fft_size;
      if (is_real_to_complex) {
        onembed_val = fft_size / 2 + 1;
      } else if (is_complex_to_real) {
        inembed_val = fft_size / 2 + 1;
      }
    } else {
      // Axis is NOT the last dimension
      // For non-last axis FFT, the data layout is more complex.
      // We use cuFFT's advanced data layout feature.

      // For example, if we have shape [2, 4] and want FFT along axis 0:
      // - fft_size = 2
      // - The 4 FFTs operate on elements at positions:
      //   batch 0: [0, 4] (elements [0,0] and [1,0])
      //   batch 1: [1, 5] (elements [0,1] and [1,1])
      //   batch 2: [2, 6] (elements [0,2] and [1,2])
      //   batch 3: [3, 7] (elements [0,3] and [1,3])
      // - istride = 4 (elements along axis 0 are 4 apart)
      // - idist = 1 (consecutive batches are 1 apart)
      // - batch = 4

      // idist = distance from one batch to the next
      // For non-last axis, consecutive batch elements are 1 apart
      idist = 1;
      odist = 1;

      // Adjust batch count: we only batch along the last axis/axes
      // For shape [M, N] with axis=0: batch = N, fft_size = M
      // istride = N (distance between consecutive elements along axis 0)
      // idist = 1 (consecutive FFTs start 1 element apart)

      if (is_real_to_complex) {
        inembed_val = fft_size;
        onembed_val = fft_size / 2 + 1;
      } else if (is_complex_to_real) {
        inembed_val = fft_size / 2 + 1;
        onembed_val = fft_size;
      } else {
        inembed_val = fft_size;
        onembed_val = fft_size;
      }
    }

    if (is_real_to_complex) {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          1, // rank
          &fft_size,
          &inembed_val,
          istride,
          idist,
          &onembed_val,
          ostride,
          odist,
          CUFFT_R2C,
          batch));
    } else if (is_complex_to_real) {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          1, // rank
          &fft_size,
          &inembed_val,
          istride,
          idist,
          &onembed_val,
          ostride,
          odist,
          CUFFT_C2R,
          batch));
    } else {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          1, // rank
          &fft_size,
          &inembed_val,
          istride,
          idist,
          &onembed_val,
          ostride,
          odist,
          CUFFT_C2C,
          batch));
    }
  } else {
    // Multi-dimensional FFT
    if (!standard_layout) {
      throw std::runtime_error(
          "[FFT] cuFFT backend currently only supports multi-dimensional FFT "
          "on the last "
          "contiguous dimensions. "
          "Please transpose your input so FFT axes are the last dimensions.");
    }

    int batch = 1;
    for (int i = 0; i < ndim - rank; ++i) {
      batch *= in_contiguous.shape(i);
    }

    // Compute embed/stride/dist parameters for batched multi-dimensional FFT
    std::vector<int> inembed(n.begin(), n.end());
    std::vector<int> onembed(n.begin(), n.end());

    int istride = 1, ostride = 1;
    int idist = 1, odist = 1;

    // Compute distance between batches
    for (int i = 0; i < rank; ++i) {
      if (is_real_to_complex && i == rank - 1) {
        idist *= n[i];
        onembed[i] = n[i] / 2 + 1;
        odist *= (n[i] / 2 + 1);
      } else if (is_complex_to_real && i == rank - 1) {
        inembed[i] = n[i] / 2 + 1;
        idist *= (n[i] / 2 + 1);
        odist *= n[i];
      } else {
        idist *= n[i];
        odist *= n[i];
      }
    }

    if (is_real_to_complex) {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          rank,
          n.data(),
          inembed.data(),
          istride,
          idist,
          onembed.data(),
          ostride,
          odist,
          CUFFT_R2C,
          batch));
    } else if (is_complex_to_real) {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          rank,
          n.data(),
          inembed.data(),
          istride,
          idist,
          onembed.data(),
          ostride,
          odist,
          CUFFT_C2R,
          batch));
    } else {
      CHECK_CUFFT_ERROR(cufftPlanMany(
          &plan,
          rank,
          n.data(),
          inembed.data(),
          istride,
          idist,
          onembed.data(),
          ostride,
          odist,
          CUFFT_C2C,
          batch));
    }
  }

  // Set the stream
  CHECK_CUFFT_ERROR(cufftSetStream(plan, cuda_stream));

  // Execute the FFT
  int direction = inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD;

  // Must use gpu_ptr, not data<void>() which may copy to managed/host memory
  if (is_real_to_complex) {
    CHECK_CUFFT_ERROR(cufftExecR2C(
        plan,
        reinterpret_cast<cufftReal*>(
            const_cast<void*>(gpu_ptr<void>(in_contiguous))),
        reinterpret_cast<cufftComplex*>(gpu_ptr<void>(out))));
  } else if (is_complex_to_real) {
    CHECK_CUFFT_ERROR(cufftExecC2R(
        plan,
        reinterpret_cast<cufftComplex*>(
            const_cast<void*>(gpu_ptr<void>(in_contiguous))),
        reinterpret_cast<cufftReal*>(gpu_ptr<void>(out))));
  } else {
    CHECK_CUFFT_ERROR(cufftExecC2C(
        plan,
        reinterpret_cast<cufftComplex*>(
            const_cast<void*>(gpu_ptr<void>(in_contiguous))),
        reinterpret_cast<cufftComplex*>(gpu_ptr<void>(out)),
        direction));
  }

  // Destroy the plan
  CHECK_CUFFT_ERROR(cufftDestroy(plan));

  // Apply normalization for inverse FFT
  // cuFFT does not normalize, so we need to divide by N for inverse transforms
  if (inverse_) {
    size_t nelem = std::accumulate(
        axes_.begin(), axes_.end(), size_t{1}, [&fft_shape](auto x, auto y) {
          return x * fft_shape[y];
        });
    float scale = 1.0f / static_cast<float>(nelem);

    // Use thrust to scale the output in-place
    // Must use gpu_ptr, not data<>() which may copy to managed/host memory
    if (is_complex_to_real) {
      // Scale real output
      thrust::device_ptr<float> data_ptr(gpu_ptr<float>(out));
      thrust::transform(
          cu::thrust_policy(cuda_stream),
          data_ptr,
          data_ptr + out.size(),
          data_ptr,
          ScaleReal{scale});
    } else {
      // Scale complex output
      thrust::device_ptr<cufftComplex> data_ptr(
          reinterpret_cast<cufftComplex*>(gpu_ptr<void>(out)));
      thrust::transform(
          cu::thrust_policy(cuda_stream),
          data_ptr,
          data_ptr + out.size(),
          data_ptr,
          ScaleComplex{scale});
    }
  }
}

} // namespace mlx::core
