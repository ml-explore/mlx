// Copyright © 2023 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <
  typename T, 
  const int BM, /* Threadgroup rows (in threads) */
  const int BN, /* Threadgroup cols (in threads) */
  const int TM, /* Thread rows (in elements) */
  const int TN > /* Thread cols (in elements) */ 
struct GEMVKernel {

  static_assert(BN == SIMD_SIZE, "gemv block must have a width of SIMD_SIZE");

  // - The matrix of size (M = out_vec_size, N = in_vec_size) is divided up 
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows 
  //      and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across the rows
  //      These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid will have blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted inwards 
  //     such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN * TN * 2;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec, 
      const constant int& in_vec_size [[buffer(3)]],
      const constant int& out_vec_size [[buffer(4)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {

    // Appease compiler 
    (void)lid;

    // Threadgroup in_vec cache
    threadgroup T* in_vec_block = tgp_memory + simd_lid * TN * 2;

    // Thread local accumulation results 
    thread T result[TM] = {0};
    thread T inter[TN];
    thread T v_coeff[TN];

    // Block position
    int out_row = (tid.x * BM + simd_gid) * TM;

    // Exit simdgroup if rows out of bound
    if(out_row >= out_vec_size) 
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * in_vec_size;

    // Loop over in_vec in blocks of BN * TN
    for(int bn = simd_lid * TN; bn < in_vec_size; bn += BN * TN) {

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Prefetch in_vector for threadgroup use
      if(simd_gid == 0) {
        // Main load loop
        if(bn + TN <= in_vec_size) {

          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            in_vec_block[tn] = in_vec[bn + tn];
          }

        } else { // Edgecase

          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            in_vec_block[tn] = bn + tn < in_vec_size ? in_vec[bn + tn] : 0;
          }

        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load for all rows
      #pragma clang loop unroll(full)
      for(int tn = 0; tn < TN; tn++) {
        v_coeff[tn] = in_vec_block[tn];
      }

      // Per thread work loop
      #pragma clang loop unroll(full)
      for(int tm = 0; tm < TM; tm++) {

        // Load for the row 
        if(bn + TN <= in_vec_size) {
          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[tm * in_vec_size + bn + tn];
          }

        } else { // Edgecase
          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            int col_idx = (bn + tn) < in_vec_size ? (bn + tn) : (in_vec_size - 1);
            inter[tn] = mat[tm * in_vec_size + col_idx];
          }
        }

        // Accumulate results
        for(int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }

      }
    }

    // Simdgroup accumulations
    #pragma clang loop unroll(full)
    for(int tm = 0; tm < TM; tm++) {
      result[tm] = simd_sum(result[tm]);
    }

    // Write outputs
    if(simd_lid == 0) {

      #pragma clang loop unroll(full)
      for(int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = result[tm];
      }

    }

  }

};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
  typename T, 
  const int BM, /* Threadgroup rows (in threads) */
  const int BN, /* Threadgroup cols (in threads) */
  const int TM, /* Thread rows (in elements) */
  const int TN > /* Thread cols (in elements) */ 
struct GEMVTKernel {

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up 
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows 
  //      and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across the rows
  //      These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid will have blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted inwards 
  //     such that the thread block fits exactly in the matrix


  MLX_MTL_CONST short tgp_mem_size = BN * BM * TN;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec, 
      const constant int& in_vec_size [[buffer(3)]],
      const constant int& out_vec_size [[buffer(4)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {

    // Appease compiler 
    (void)simd_gid;
    (void)simd_lid;

    // Thread local accumulation results
    T result[TN] = {0};
    T inter[TN];
    T v_coeff[TM];

    // Threadgroup accumulation results
    threadgroup T* tgp_results = tgp_memory + lid.x * BM * TN;

    int out_col = (tid.x * BN + lid.x) * TN;
    int in_row = lid.y * TM;

    // Edgecase handling
    if (out_col < out_vec_size) {

      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      int bm = in_row;
      for(; bm < in_vec_size; bm += BM * TM) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        if(bm + TM <= in_vec_size) {

          #pragma clang loop unroll(full)
          for(int tm = 0; tm < TM; tm++) {
            v_coeff[tm] = in_vec[bm + tm];
          }

          #pragma clang loop unroll(full)
          for(int tm = 0; tm < TM; tm++) {
            for(int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * out_vec_size + out_col + tn];
            }
            for(int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }
          }
        
        } else { // Edgecase handling
          for(int tm = 0; bm + tm < in_vec_size; tm++) {
            v_coeff[tm] = in_vec[bm + tm];

            for(int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * out_vec_size + out_col + tn];
            }
            for(int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }

          }
        }
      }

    }

    // Threadgroup collection

    #pragma clang loop unroll(full)
    for(int i = 0; i < TN; i++) {
      tgp_results[lid.y * TN + i] = result[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup accumulation and writing out results
    if(lid.y == 0 && out_col < out_vec_size) {
      
      #pragma clang loop unroll(full)
      for(int i = 1; i < BM; i++) {

        #pragma clang loop unroll(full)
        for(int j = 0; j < TN; j++) {
          result[j] += tgp_results[i * TN + j];
        }
      }

      #pragma clang loop unroll(full)
      for(int j = 0; j < TN; j++) {
        out_vec[out_col + j] = result[j];
      }
    }

  }


};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T, 
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]], 
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& vector_batch_stride [[buffer(5)]],
    const constant int& matrix_batch_stride [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run( 
    mat, 
    in_vec, 
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}

template <
    typename T, 
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_nc(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]], 
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& nc_dim [[buffer(5)]],
    const device int* nc_shape [[buffer(6)]],
    const device size_t* nc_strides_vec [[buffer(7)]],
    const device size_t* nc_strides_mat [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += elem_to_loc(tid.z, nc_shape, nc_strides_vec, nc_dim);
  mat += elem_to_loc(tid.z, nc_shape, nc_strides_mat, nc_dim);
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run( 
    mat, 
    in_vec, 
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}


#define instantiate_gemv_c(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn)]] \
  [[kernel]] void gemv<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& vector_batch_stride [[buffer(5)]], \
    const constant int& matrix_batch_stride [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_nc(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn "_nc")]] \
  [[kernel]] void gemv_nc<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& nc_dim [[buffer(5)]], \
    const device int* nc_shape [[buffer(6)]], \
    const device size_t* nc_strides_vec [[buffer(7)]], \
    const device size_t* nc_strides_mat [[buffer(8)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_c(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_nc(name, itype, bm, bn, tm, tn)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 4, 32, 1, 4) \
  instantiate_gemv(name, itype, 4, 32, 4, 4) \
  instantiate_gemv(name, itype, 8, 32, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T, 
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]], 
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& vector_batch_stride [[buffer(5)]],
    const constant int& matrix_batch_stride [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run( 
    mat, 
    in_vec, 
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );
}

template <
    typename T, 
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_t_nc(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]], 
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& nc_dim [[buffer(5)]],
    const device int* nc_shape [[buffer(6)]],
    const device size_t* nc_strides_vec [[buffer(7)]],
    const device size_t* nc_strides_mat [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += elem_to_loc(tid.z, nc_shape, nc_strides_vec, nc_dim);
  mat += elem_to_loc(tid.z, nc_shape, nc_strides_mat, nc_dim);
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run( 
    mat, 
    in_vec, 
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}

#define instantiate_gemv_t_c(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_t_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn)]] \
  [[kernel]] void gemv_t<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& vector_batch_stride [[buffer(5)]], \
    const constant int& matrix_batch_stride [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_t_nc(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_t_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn "_nc")]] \
  [[kernel]] void gemv_t_nc<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& nc_dim [[buffer(5)]], \
    const device int* nc_shape [[buffer(6)]], \
    const device size_t* nc_strides_vec [[buffer(7)]], \
    const device size_t* nc_strides_mat [[buffer(8)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_t(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_t_c(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_t_nc(name, itype, bm, bn, tm, tn)

#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 8, 8, 4, 1) \
  instantiate_gemv_t(name, itype, 8, 8, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 16, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 32, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 64, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 128, 4, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);
