// Copyright © 2025 Apple Inc.

#include <cufftXt.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/complex.cuh"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T>
__global__ void scale_fft_output(T* out, T scale, size_t size) {
  auto index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] *= scale;
  }
}

} // namespace cu

namespace {

void check_cufft_error(const char* name, cufftResult err) {
  if (err != CUFFT_SUCCESS) {
    throw std::runtime_error(
        std::string(name) +
        " failed with code: " + std::to_string(static_cast<int>(err)) + ".");
  }
}

#define CHECK_CUFFT_ERROR(cmd) check_cufft_error(#cmd, (cmd))

enum class FFTTransformType : uint8_t {
  C2C = 0,
  R2C = 1,
  C2R = 2,
};

struct FFTPlanKey {
  int device_id;
  FFTTransformType transform_type;
  int64_t n;
  int64_t batch;
};

struct CuFFTPlan {
  explicit CuFFTPlan(int device_id, cufftHandle handle, size_t workspace_size)
      : device_id(device_id), handle(handle), workspace_size(workspace_size) {}

  ~CuFFTPlan() {
    if (handle != 0) {
      try {
        cu::device(device_id).make_current();
        cufftDestroy(handle);
      } catch (...) {
      }
    }
  }

  int device_id;
  cufftHandle handle;
  size_t workspace_size;
};

struct OrderedArray {
  array arr;
  std::vector<int> order;
};

auto& fft_plan_cache() {
  static LRUBytesKeyCache<FFTPlanKey, std::shared_ptr<CuFFTPlan>> cache(
      "MLX_CUDA_FFT_CACHE_SIZE",
      /* default_capacity */ 128);
  return cache;
}

FFTPlanKey make_plan_key(
    int device_id,
    FFTTransformType transform_type,
    int64_t n,
    int64_t batch) {
  FFTPlanKey key{};
  key.device_id = device_id;
  key.transform_type = transform_type;
  key.n = n;
  key.batch = batch;
  return key;
}

cudaDataType_t input_type(FFTTransformType transform_type) {
  switch (transform_type) {
    case FFTTransformType::C2C:
    case FFTTransformType::C2R:
      return CUDA_C_32F;
    case FFTTransformType::R2C:
      return CUDA_R_32F;
  }
  throw std::runtime_error("[FFT] Unsupported cuFFT input transform type.");
}

cudaDataType_t output_type(FFTTransformType transform_type) {
  switch (transform_type) {
    case FFTTransformType::C2C:
    case FFTTransformType::R2C:
      return CUDA_C_32F;
    case FFTTransformType::C2R:
      return CUDA_R_32F;
  }
  throw std::runtime_error("[FFT] Unsupported cuFFT output transform type.");
}

cudaDataType_t execution_type(FFTTransformType transform_type) {
  switch (transform_type) {
    case FFTTransformType::C2C:
      return CUDA_C_32F;
    case FFTTransformType::R2C:
      return CUDA_R_32F;
    case FFTTransformType::C2R:
      return CUDA_C_32F;
  }
  throw std::runtime_error("[FFT] Unsupported cuFFT execution transform type.");
}

int64_t input_embed(FFTTransformType transform_type, int64_t n) {
  return transform_type == FFTTransformType::C2R ? (n / 2 + 1) : n;
}

int64_t output_embed(FFTTransformType transform_type, int64_t n) {
  return transform_type == FFTTransformType::R2C ? (n / 2 + 1) : n;
}

int exec_direction(FFTTransformType transform_type, bool inverse) {
  switch (transform_type) {
    case FFTTransformType::C2C:
      return inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    case FFTTransformType::R2C:
      return CUFFT_FORWARD;
    case FFTTransformType::C2R:
      return CUFFT_INVERSE;
  }
  throw std::runtime_error("[FFT] Unsupported cuFFT execution direction.");
}

std::shared_ptr<CuFFTPlan> get_fft_plan(
    cu::CommandEncoder& encoder,
    FFTTransformType transform_type,
    int64_t n,
    int64_t batch) {
  auto key = BytesKey<FFTPlanKey>{};
  key.pod =
      make_plan_key(encoder.device().cuda_device(), transform_type, n, batch);

  auto& cache = fft_plan_cache();
  if (auto entry = cache.find(key); entry != cache.end()) {
    return entry->second;
  }

  encoder.device().make_current();

  cufftHandle handle = 0;
  size_t workspace_size = 0;
  try {
    CHECK_CUFFT_ERROR(cufftCreate(&handle));
    CHECK_CUFFT_ERROR(cufftSetAutoAllocation(handle, 0));
    CHECK_CUFFT_ERROR(cufftSetStream(handle, encoder.stream()));

    long long plan_n[1] = {n};
    long long inembed[1] = {input_embed(transform_type, n)};
    long long onembed[1] = {output_embed(transform_type, n)};
    CHECK_CUFFT_ERROR(cufftXtMakePlanMany(
        handle,
        /* rank= */ 1,
        plan_n,
        inembed,
        /* istride= */ 1,
        /* idist= */ input_embed(transform_type, n),
        input_type(transform_type),
        onembed,
        /* ostride= */ 1,
        /* odist= */ output_embed(transform_type, n),
        output_type(transform_type),
        batch,
        &workspace_size,
        execution_type(transform_type)));
  } catch (...) {
    if (handle != 0) {
      encoder.device().make_current();
      cufftDestroy(handle);
    }
    throw;
  }

  auto plan = std::make_shared<CuFFTPlan>(
      encoder.device().cuda_device(), handle, workspace_size);
  return cache.emplace(key, plan).first->second;
}

std::vector<int> make_identity_order(int ndim) {
  std::vector<int> order(ndim);
  std::iota(order.begin(), order.end(), 0);
  return order;
}

std::vector<int> move_axis_to_back_permutation(int ndim, int axis_pos) {
  std::vector<int> perm;
  perm.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (i != axis_pos) {
      perm.push_back(i);
    }
  }
  perm.push_back(axis_pos);
  return perm;
}

std::vector<int> apply_permutation(
    const std::vector<int>& values,
    const std::vector<int>& perm) {
  std::vector<int> out(perm.size());
  for (int i = 0; i < perm.size(); ++i) {
    out[i] = values[perm[i]];
  }
  return out;
}

int find_axis_position(const std::vector<int>& order, int axis) {
  auto it = std::find(order.begin(), order.end(), axis);
  if (it == order.end()) {
    throw std::runtime_error("[FFT] Internal axis tracking mismatch.");
  }
  return static_cast<int>(it - order.begin());
}

OrderedArray prepare_input(
    const OrderedArray& current,
    int axis,
    bool allow_direct,
    cu::CommandEncoder& encoder,
    Stream s) {
  int axis_pos = find_axis_position(current.order, axis);
  bool axis_last = axis_pos == static_cast<int>(current.order.size()) - 1;
  bool direct = allow_direct && axis_last && current.arr.flags().row_contiguous;

  if (direct) {
    return current;
  }

  array view = current.arr;
  std::vector<int> order = current.order;
  if (!axis_last) {
    auto perm = move_axis_to_back_permutation(current.arr.ndim(), axis_pos);
    view = transpose_in_eval(current.arr, perm);
    order = apply_permutation(current.order, perm);
  }

  array packed = contiguous_copy_gpu(view, s);
  encoder.add_temporary(packed);
  return {std::move(packed), std::move(order)};
}

void execute_fft(
    const array& in,
    array& out,
    FFTTransformType transform_type,
    bool inverse,
    cu::CommandEncoder& encoder) {
  if (!in.flags().row_contiguous || in.strides(-1) != 1) {
    throw std::runtime_error("[FFT] Expected packed row-contiguous FFT input.");
  }

  int64_t n =
      transform_type == FFTTransformType::C2R ? out.shape(-1) : in.shape(-1);
  int64_t batch = in.shape().empty() ? 1 : in.size() / in.shape(-1);
  auto plan = get_fft_plan(encoder, transform_type, n, batch);

  encoder.set_input_array(in);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_output_array(out);
  encoder.add_completed_handler([plan]() {});

  encoder.device().make_current();
  CHECK_CUFFT_ERROR(cufftSetStream(plan->handle, encoder.stream()));
  auto* workspace = allocate_workspace(encoder, plan->workspace_size);
  CHECK_CUFFT_ERROR(cufftSetWorkArea(plan->handle, workspace));

  auto capture = encoder.capture_context();
  CHECK_CUFFT_ERROR(cufftXtExec(
      plan->handle,
      gpu_ptr<void>(in),
      gpu_ptr<void>(out),
      exec_direction(transform_type, inverse)));
}

void restore_output_layout(const OrderedArray& current, array& out) {
  Strides out_strides(out.ndim());
  for (int i = 0; i < current.order.size(); ++i) {
    out_strides[current.order[i]] = current.arr.strides(i);
  }

  auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(out.shape(), out_strides);
  bool contiguous =
      current.arr.flags().contiguous && data_size == current.arr.data_size();

  out.copy_shared_buffer(
      current.arr,
      out_strides,
      {contiguous, row_contiguous, col_contiguous},
      current.arr.data_size());
}

void apply_inverse_scale(
    array& arr,
    const std::vector<size_t>& axes,
    const array& out,
    cu::CommandEncoder& encoder) {
  if (axes.empty()) {
    return;
  }

  double scale = 1.0;
  for (auto axis : axes) {
    scale /= out.shape(axis);
  }

  size_t size = arr.data_size();
  dim3 block_dims(256);
  dim3 grid_dims((size + block_dims.x - 1) / block_dims.x);

  encoder.set_input_array(arr);
  encoder.set_output_array(arr);

  if (arr.dtype() == float32) {
    float scale_f = static_cast<float>(scale);
    encoder.add_kernel_node(
        cu::scale_fft_output<float>,
        grid_dims,
        block_dims,
        gpu_ptr<float>(arr),
        scale_f,
        size);
  } else if (arr.dtype() == complex64) {
    cu::complex64_t scale_f(static_cast<float>(scale), 0.0f);
    encoder.add_kernel_node(
        cu::scale_fft_output<cu::complex64_t>,
        grid_dims,
        block_dims,
        gpu_ptr<cu::complex64_t>(arr),
        scale_f,
        size);
  } else {
    throw std::runtime_error("[FFT] Unsupported dtype for inverse scaling.");
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("FFT::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  auto& in = inputs[0];

  if (out.size() == 0) {
    return;
  }

  auto order = make_identity_order(in.ndim());
  OrderedArray current{in, std::move(order)};

  std::vector<int> axis_sequence;
  axis_sequence.reserve(axes_.size());
  if (inverse_) {
    for (auto axis : axes_) {
      axis_sequence.push_back(static_cast<int>(axis));
    }
  } else {
    for (int i = static_cast<int>(axes_.size()) - 1; i >= 0; --i) {
      axis_sequence.push_back(static_cast<int>(axes_[i]));
    }
  }

  int real_axis = axes_.empty() ? -1 : static_cast<int>(axes_.back());

  for (int i = 0; i < axis_sequence.size(); ++i) {
    int axis = axis_sequence[i];
    bool step_real = real_ && axis == real_axis;
    auto transform_type = step_real
        ? (inverse_ ? FFTTransformType::C2R : FFTTransformType::R2C)
        : FFTTransformType::C2C;

    // cuFFT may overwrite the input buffer for C2R, so only use the direct
    // input when the transform is out-of-place from the library's perspective
    // or when the original input may be donated to the output.
    auto prepared = prepare_input(
        current,
        axis,
        /* allow_direct= */ transform_type != FFTTransformType::C2R ||
            is_donatable(in, out),
        encoder,
        s);

    Shape step_shape = prepared.arr.shape();
    if (step_real) {
      step_shape.back() = out.shape(axis);
    }

    Dtype step_dtype =
        transform_type == FFTTransformType::C2R ? float32 : complex64;
    array step_out(std::move(step_shape), step_dtype, nullptr, {});
    execute_fft(prepared.arr, step_out, transform_type, inverse_, encoder);
    encoder.add_temporary(step_out);

    current = {std::move(step_out), std::move(prepared.order)};
  }

  if (inverse_) {
    apply_inverse_scale(current.arr, axes_, out, encoder);
  }

  restore_output_layout(current, out);
}

} // namespace mlx::core
