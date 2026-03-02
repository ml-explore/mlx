// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/numeric_conversion.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// Fused vectorized dequantize and multiply-add:
// w_dq = w * scale + bias
// out = fma(x, w_dq, out)
template <int N, typename T, typename Q>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, T scale, T bias, float* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::AlignedArray<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::AlignedArray<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  w_dq = w_dq * scale + bias;

  // Promote x/w to float.
  static_assert(!cuda::std::is_same_v<T, float>);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);

  // Multiply and add.
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// Specialized for float which does not need promotions.
template <int N, typename Q>
__device__ __forceinline__ void
dequant_fma(const float* x, const Q* w, float scale, float bias, float* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::AlignedArray<float, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::AlignedArray<Q, N>*>(w));
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  cutlass::NumericArrayConverter<float, Q, N> converter;
  cutlass::Array<float, N> w_dq = converter(w_vec);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    w_dq[i] = w_dq[i] * scale + bias;
  }

  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q>
__global__ void qmv_kernel(
    const T* x,
    const Q* w,
    const T* scales,
    const T* biases,
    T* out,
    int n,
    int k) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // The row that this warp handles.
  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  // Advance pointers of x/out.
  x += block.group_index().y * k;
  out += block.group_index().y * n;

  // For sub-byte Q, pointer moves by 8bits for each advance, e.g. w += 1 would
  // move past 2 elements for 4-bit Q.
  constexpr int w_step = 8 / cuda::std::min(8, cute::sizeof_bits_v<Q>);

  // How many groups (and scales/biases) in a row.
  int groups_per_row = k / group_size;

  // Advance w/scales/biases to current row.
  w += static_cast<int64_t>(row) * k / w_step;
  scales += static_cast<int64_t>(row) * groups_per_row;
  if constexpr (has_bias) {
    biases += static_cast<int64_t>(row) * groups_per_row;
  }

  // Accumulations of current row.
  float sums[elems_per_thread] = {};

  auto dequant_fma_tile = [&](int idx) {
    T scale = scales[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases[idx / group_size];
    }
    dequant_fma<elems_per_thread>(x + idx, w + idx / w_step, scale, bias, sums);
  };

  // Loop over k dimension.
  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int r = 0; r < k / elems_per_warp; ++r) {
    int idx = warp.thread_rank() * elems_per_thread + r * elems_per_warp;
    dequant_fma_tile(idx);
  }

  // Handle remaining elements in k dimension.
  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      dequant_fma_tile(idx);
    }
  }

  // Result for current row.
  float sum{0};
#pragma unroll
  for (int i = 0; i < elems_per_thread; ++i) {
    sum += sums[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  // Write result for current warp, which maps to rows 1-to-1.
  if (warp.thread_rank() == 0) {
    out[row] = static_cast<T>(sum);
  }
}

template <int group_size, bool has_bias, typename T, typename Q, typename F>
void qmv(
    const T* x,
    const Q* w,
    const T* scales,
    const T* biases,
    T* out,
    int m,
    int n,
    int k,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread = 8;

  dim3 num_blocks{uint32_t(cuda::ceil_div(n, rows_per_block)), uint32_t(m)};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {&x, &w, &scales, &biases, &out, &n, &k};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        cuda_type_t<T>,
        cuda_type_t<Q>>;
    launch_kernel(
        reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
  });
}

} // namespace cu

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void
dispatch_quant_types(int bits, QuantizationMode mode, const char* tag, F&& f) {
  if (mode == QuantizationMode::Mxfp4) {
    f.template operator()<cutlass::float_e2m1_t>();
  } else if (mode == QuantizationMode::Mxfp8) {
    f.template operator()<cutlass::float_e4m3_t>();
  } else if (mode == QuantizationMode::Nvfp4) {
    f.template operator()<cutlass::float_e2m1_t>();
  } else {
    if (bits == 2) {
      f.template operator()<cutlass::uint2b_t>();
    } else if (bits == 4) {
      f.template operator()<cutlass::uint4b_t>();
    } else if (bits == 8) {
      f.template operator()<uint8_t>();
    } else {
      throw std::invalid_argument(
          fmt::format("{} {}-bit quantization is not supported.", tag, bits));
    }
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 16) {
    f.template operator()<16>();
  } else if (group_size == 32) {
    f.template operator()<32>();
  } else if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[quantized_matmul]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_bool(biases.has_value(), [&](auto has_bias) {
      dispatch_quant_types(bits, mode, tag, [&]<typename Q>() {
        dispatch_groups(group_size, tag, [&]<int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_output_array(out);
          cu::qmv<group_size, has_bias.value>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<T>(scales),
              biases ? gpu_ptr<T>(*biases) : nullptr,
              gpu_ptr<T>(out),
              m,
              n,
              k,
              [&](auto* kernel, dim3 num_blocks, dim3 block_dims, void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        });
      });
    });
  });
}

} // namespace mlx::core
