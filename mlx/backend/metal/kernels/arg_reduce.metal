// Copyright © 2023 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr constant U init = Limits<U>::max;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val > current.val || (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U> reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i=0; i<N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset+i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr constant U init = Limits<U>::min;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val < current.val || (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U> reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i=0; i<N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset+i;
      }
    }
    return best;
  }
};

template <typename U>
IndexValPair<U> simd_shuffle_down(IndexValPair<U> data, uint16_t delta) {
  return IndexValPair<U>{
    simd_shuffle_down(data.index, delta),
    simd_shuffle_down(data.val, delta)
  };
}


template <typename T, typename Op, int N_READS>
[[kernel]] void arg_reduce_general(
    const device T *in [[buffer(0)]],
    device uint32_t *out [[buffer(1)]],
    const device int *shape [[buffer(2)]],
    const device size_t *in_strides [[buffer(3)]],
    const device size_t *out_strides [[buffer(4)]],
    const device size_t& ndim [[buffer(5)]],
    const device size_t& axis_stride [[buffer(6)]],
    const device size_t& axis_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  //
  // The sketch of the kernel is as follows.
  //    1. Launch prod(shape) * thread_group_size threads.
  //    2. Loop ceildiv(axis_size / lsize) times
  //    3. Read input values
  //    4. Reduce among them and go to 3
  //    4. Reduce in each simd_group
  //    6. Write in the thread local memory
  //    6. Reduce them across thread group
  //    7. Write the output without need for atomic
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole threadgroup.
  auto in_idx = elem_to_loc(gid / lsize, shape, in_strides, ndim);
  auto out_idx = elem_to_loc(gid / lsize, shape, out_strides, ndim);

  IndexValPair<T> best{0, Op::init};

  threadgroup IndexValPair<T> local_data[32];

  // Loop over the reduction axis in lsize*N_READS buckets
  for (uint r=0; r < ceildiv(axis_size, N_READS*lsize); r++) {
    // Read the current value
    uint32_t current_index = r*lsize*N_READS + lid*N_READS;
    uint32_t offset = current_index;
    const device T * current_in = in + in_idx + current_index * axis_stride;
    T vals[N_READS];
    for (int i=0; i<N_READS; i++) {
      vals[i] = (current_index < axis_size) ? *current_in : T(Op::init);
      current_index++;
      current_in += axis_stride;
    }
    best = op.template reduce_many<N_READS>(best, vals, offset);
  }
  // At this point we have reduced the axis into thread group best values so we
  // need to reduce across the thread group.

  // First per simd reduction.
  for (uint offset=simd_size/2; offset>0; offset/=2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Write to the threadgroup memory
  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }

  // Read the appropriate value from local data and perform one simd reduction
  uint simd_groups = ceildiv(lsize, simd_size);
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  }
  for (uint offset=simd_size/2; offset>0; offset/=2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Finally write the output
  if (lid == 0) {
    out[out_idx] = best.index;
  }
}

#define instantiate_arg_reduce_helper(name, itype, op) \
  template [[host_name(name)]] \
  [[kernel]] void arg_reduce_general<itype, op<itype>, 4>( \
      const device itype *in [[buffer(0)]], \
      device uint32_t * out [[buffer(1)]], \
      const device int *shape [[buffer(2)]], \
      const device size_t *in_strides [[buffer(3)]], \
      const device size_t *out_strides [[buffer(4)]], \
      const device size_t& ndim [[buffer(5)]], \
      const device size_t& axis_stride [[buffer(6)]], \
      const device size_t& axis_size [[buffer(7)]], \
      uint gid [[thread_position_in_grid]], \
      uint lid [[thread_position_in_threadgroup]], \
      uint lsize [[threads_per_threadgroup]], \
      uint simd_size [[threads_per_simdgroup]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_arg_reduce(name, itype) \
  instantiate_arg_reduce_helper("argmin_" #name , itype, ArgMin) \
  instantiate_arg_reduce_helper("argmax_" #name , itype, ArgMax)

instantiate_arg_reduce(bool_, bool)
instantiate_arg_reduce(uint8, uint8_t)
instantiate_arg_reduce(uint16, uint16_t)
instantiate_arg_reduce(uint32, uint32_t)
instantiate_arg_reduce(uint64, uint64_t)
instantiate_arg_reduce(int8, int8_t)
instantiate_arg_reduce(int16, int16_t)
instantiate_arg_reduce(int32, int32_t)
instantiate_arg_reduce(int64, int64_t)
instantiate_arg_reduce(float16, half)
instantiate_arg_reduce(float32, float)
instantiate_arg_reduce(bfloat16, bfloat16_t)