// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/cute_dequant.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// Fused vectorized dequantize and multiply-add:
// w_dq = w * scale + bias
// out = fma(x, w_dq, out)
template <int N, bool has_bias, typename T, typename Q, typename S>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, T* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<T, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    if constexpr (cuda::std::is_same_v<T, float>) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        w_dq[i] = w_dq[i] * T(scale) + bias;
      }
    } else {
      w_dq = w_dq * T(scale) + bias;
    }
  } else {
    w_dq = w_dq * T(scale);
  }

  // Multiply and add.
  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

// Specialization for doing float32 accumulations on narrow types.
template <
    int N,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, float>>>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, float* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    w_dq = w_dq * T(scale) + bias;
  } else {
    w_dq = w_dq * T(scale);
  }

  // Promote x/w to float.
  static_assert(!cuda::std::is_same_v<T, float>);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);

  // Multiply and add.
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

template <
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__device__ __forceinline__ void qmv_kernel_impl(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    int row,
    int w_batch,
    int n,
    int k) {
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  // For sub-byte Q, pointer moves by 8bits for each advance, e.g. w += 1 would
  // move past 2 elements for 4-bit Q.
  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [&](int idx) { return idx * cuda::std::min(8, bits) / 8; };

  // How many groups (and scales/biases) in a row.
  int groups_per_row = k / group_size;

  // Advance w/scales/biases to current row.
  w += (static_cast<int64_t>(row) + n * w_batch) * w_step(k);
  scales += (static_cast<int64_t>(row) + n * w_batch) * groups_per_row;
  if constexpr (has_bias) {
    biases += (static_cast<int64_t>(row) + n * w_batch) * groups_per_row;
  }

  // Accumulations of current row.
  cuda::std::conditional_t<(bits >= 8), float, T> sums[elems_per_thread] = {};

  auto dequant_fma_tile = [&](int idx) {
    S scale = scales[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases[idx / group_size];
    }
    dequant_fma<elems_per_thread, has_bias>(
        x + idx, w + w_step(idx), scale, bias, sums);
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

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ void qmv_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    int n,
    int k,
    bool broadcast_w) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // The row that this warp handles.
  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  // Advance pointers of x/out for M and batch dimensions.
  int m = grid.dim_blocks().y;
  int l = block.group_index().z;
  x += block.group_index().y * k + m * k * l;
  out += block.group_index().y * n + m * n * l;
  int w_batch = broadcast_w ? 0 : l;

  qmv_kernel_impl<elems_per_thread, group_size, has_bias, has_residue_k>(
      x, w, scales, biases, out, row, w_batch, n, k);
}

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ void gather_qmv_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int n,
    int k) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  int m = grid.dim_blocks().y;
  int l = block.group_index().z;
  uint32_t x_idx = lhs_indices[l];
  uint32_t w_idx = rhs_indices[l];

  x += block.group_index().y * k + m * k * x_idx;
  out += block.group_index().y * n + m * n * l;

  qmv_kernel_impl<elems_per_thread, group_size, has_bias, has_residue_k>(
      x, w, scales, biases, out, row, w_idx, n, k);
}

template <
    int group_size,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename F>
void qmv(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    int m,
    int n,
    int k,
    int l,
    bool broadcast_w,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread =
      (cute::sizeof_bits_v<T> <= 16 && cute::sizeof_bits_v<Q> <= 4) ? 16 : 8;

  dim3 num_blocks{
      uint32_t(cuda::ceil_div(n, rows_per_block)), uint32_t(m), uint32_t(l)};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {&x, &w, &scales, &biases, &out, &n, &k, &broadcast_w};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        T,
        Q,
        S>;
    launch_kernel(
        reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
  });
}

template <
    int group_size,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename F>
void gather_qmv(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int m,
    int n,
    int k,
    int l,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread =
      (cute::sizeof_bits_v<T> <= 16 && cute::sizeof_bits_v<Q> <= 4) ? 16 : 8;

  dim3 num_blocks{
      uint32_t(cuda::ceil_div(n, rows_per_block)), uint32_t(m), uint32_t(l)};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {
      &x, &w, &scales, &biases, &out, &lhs_indices, &rhs_indices, &n, &k};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &gather_qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        T,
        Q,
        S>;
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
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 32) {
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

template <typename T, typename F>
inline void dispatch_quant_types(
    int bits,
    int group_size,
    QuantizationMode mode,
    const char* tag,
    F&& f) {
  if (mode == QuantizationMode::Mxfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Mxfp8) {
    f.template operator()<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Nvfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_e4m3_t, 16>();
  } else {
    dispatch_groups(group_size, tag, [&]<int group_size>() {
      if (bits == 2) {
        f.template operator()<cutlass::uint2b_t, T, group_size>();
      } else if (bits == 3) {
        f.template operator()<cutlass::uint3b_t, T, group_size>();
      } else if (bits == 4) {
        f.template operator()<cutlass::uint4b_t, T, group_size>();
      } else if (bits == 5) {
        f.template operator()<cutlass::uint5b_t, T, group_size>();
      } else if (bits == 6) {
        f.template operator()<cutlass::uint6b_t, T, group_size>();
      } else if (bits == 8) {
        f.template operator()<uint8_t, T, group_size>();
      } else {
        throw std::invalid_argument(
            fmt::format("{} {}-bit quantization is not supported.", tag, bits));
      }
    });
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
  int l = out.size() / (m * n);
  bool broadcast_w = w.ndim() == 2;

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_quant_types<T>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Q, typename S, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_output_array(out);
          constexpr bool has_bias = !cutlass::has_negative_zero_v<Q>;
          cu::qmv<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              biases ? gpu_ptr<T>(*biases) : nullptr,
              gpu_ptr<T>(out),
              m,
              n,
              k,
              l,
              broadcast_w,
              [&](auto* kernel, dim3 num_blocks, dim3 block_dims, void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        });
  });
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[gather_qmm]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_quant_types<T>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Q, typename S, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_input_array(lhs_indices);
          encoder.set_input_array(rhs_indices);
          encoder.set_output_array(out);
          constexpr bool has_bias = !cutlass::has_negative_zero_v<Q>;
          cu::gather_qmv<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              biases ? gpu_ptr<T>(*biases) : nullptr,
              gpu_ptr<T>(out),
              gpu_ptr<uint32_t>(lhs_indices),
              gpu_ptr<uint32_t>(rhs_indices),
              m,
              n,
              k,
              l,
              [&](auto* kernel, dim3 num_blocks, dim3 block_dims, void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        });
  });
}

} // namespace mlx::core
