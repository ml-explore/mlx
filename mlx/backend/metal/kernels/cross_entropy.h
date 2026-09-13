// Copyright © 2026 Apple Inc.

// Fused logsumexp + gather.
//
// The accumulation is done in float32 regardless of the input type so that
// callers do not need to promote the logits with logits.astype(mx.float32).
//
// For each row: loss = logsumexp(x) - x[target]
//
// One threadgroup handles one row. The logsumexp is accumulated first, then a
// single thread does the gather and writes the loss.

template <typename T, typename AccT = float, int N_READS = 4>
[[kernel]] void cross_entropy(
    const device T* x,
    const device int* y,
    device float* loss,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  AccT ld[N_READS];

  const device T* row = x + gid * size_t(axis_size);
  const device T* in = row + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = AccT(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] =
          ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : Limits<AccT>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max
  AccT maxval = Limits<AccT>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];

  // Compute exp(x_i - maxval) and store the partial sums in local_normalizer
  AccT normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    normalizer += fast::exp(ld[i] - maxval);
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      // Gather the score of the target class and subtract it from the lse.
      AccT gap = maxval - AccT(row[y[gid]]);
      loss[gid] =
          static_cast<float>(isinf(maxval) ? gap : log(normalizer) + gap);
    }
  }
}

template <typename T, typename AccT = float, int N_READS = 4>
[[kernel]] void cross_entropy_looped(
    const device T* x,
    const device int* y,
    device float* loss,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  const device T* row = x + gid * size_t(axis_size);

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  // The threadgroup may hold fewer than SIMD_SIZE simdgroups, so initialize
  // every slot: the cross-simdgroup reduction below reads all of them.
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::finite_min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max and the normalizer in one go
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min;
  AccT normalizer = 0;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = AccT(row[offset + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = (offset + i < axis_size) ? AccT(row[offset + i])
                                           : Limits<AccT>::min;
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

  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);

  if (lid == 0) {
    AccT gap = maxval - AccT(row[y[gid]]);
    loss[gid] = static_cast<float>(isinf(maxval) ? gap : log(normalizer) + gap);
  }
}

// Gradient of the fused loss. The forward pass already folded the target score
// into the loss, so lse = loss + x[target] and the softmax probabilities come
// back as exp((x_i - x[target]) - loss).
template <typename T, typename AccT = float, int N_READS = 4>
[[kernel]] void cross_entropy_vjp(
    const device T* x,
    const device int* y,
    const device float* loss,
    const device float* cotan,
    device T* grads,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  const device T* row = x + gid * size_t(axis_size);
  device T* grads_row = grads + gid * size_t(axis_size);

  int y_n = y[gid];
  AccT g = AccT(cotan[gid]);
  AccT loss_n = AccT(loss[gid]);
  AccT x_t = AccT(row[y_n]);

  // grads aliases x when the logits buffer is donated, so hold every thread
  // here until all of them have read the target score.
  threadgroup_barrier(mem_flags::mem_device);

  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = AccT(row[offset + i]);
      }
      for (int i = 0; i < N_READS; i++) {
        AccT p = fast::exp((vals[i] - x_t) - loss_n);
        vals[i] = g * (p - ((offset + i) == y_n ? AccT(1) : AccT(0)));
      }
      for (int i = 0; i < N_READS; i++) {
        grads_row[offset + i] = T(vals[i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (offset + i < axis_size) {
          AccT v = AccT(row[offset + i]);
          AccT p = fast::exp((v - x_t) - loss_n);
          grads_row[offset + i] =
              T(g * (p - ((offset + i) == y_n ? AccT(1) : AccT(0))));
        }
      }
    }
  }
}
