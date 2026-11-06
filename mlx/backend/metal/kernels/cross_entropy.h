// Copyright © 2026 Apple Inc.

template <
    typename T,
    typename AccT = float,
    int N_READS = CROSS_ENTROPY_N_READS>
[[kernel]] void cross_entropy(
    const device T* in,
    const device int32_t* targets,
    device float* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  in += gid * size_t(axis_size);

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  int y_n = targets[gid];
  y_n = (y_n < 0) ? y_n + axis_size : y_n;
  AccT x_t = AccT(in[y_n]);

  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min;
  AccT normalizer = 0;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = AccT(in[offset + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] =
            (offset + i < axis_size) ? AccT(in[offset + i]) : Limits<AccT>::min;
      }
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += fast::exp(vals[i] - maxval);
    }
  }

  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);

  uint n_simdgroups = ceildiv(lsize, uint(SIMD_SIZE));
  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = (simd_lane_id < n_simdgroups) ? local_max[simd_lane_id]
                                         : Limits<AccT>::finite_min;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer =
      (simd_lane_id < n_simdgroups) ? local_normalizer[simd_lane_id] : AccT(0);
  normalizer = simd_sum(normalizer);

  if (lid == 0) {
    // Subtract the two logits first; they are the same magnitude, so less is
    // lost than in logsumexp - x_t.
    AccT gap = maxval - x_t;
    out[gid] = isinf(maxval) ? float(gap) : float(log(normalizer) + gap);
  }
}

template <
    typename T,
    typename AccT = float,
    int N_READS = CROSS_ENTROPY_N_READS>
[[kernel]] void cross_entropy_vjp(
    const device T* in,
    const device int32_t* targets,
    const device float* loss,
    const device float* cotan,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  size_t row_offset = gid * size_t(axis_size);
  in += row_offset;
  out += row_offset;

  int y_n = targets[gid];
  y_n = (y_n < 0) ? y_n + axis_size : y_n;
  AccT x_t = AccT(in[y_n]);
  AccT g = AccT(cotan[gid]);
  AccT l = AccT(loss[gid]);

  // out aliases in when the input is donated, so every thread must read the
  // target column before any thread writes.
  threadgroup_barrier(mem_flags::mem_device);

  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        int col = offset + i;
        AccT p = fast::exp((AccT(in[col]) - x_t) - l);
        out[col] = T(g * (p - ((col == y_n) ? AccT(1) : AccT(0))));
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        int col = offset + i;
        if (col < axis_size) {
          AccT p = fast::exp((AccT(in[col]) - x_t) - l);
          out[col] = T(g * (p - ((col == y_n) ? AccT(1) : AccT(0))));
        }
      }
    }
  }
}
