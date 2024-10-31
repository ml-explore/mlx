// Copyright © 2023-2024 Apple Inc.

#pragma once

#define DEFINE_SIMD_SCAN()                                               \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_scan(T val) {                                                   \
    return simd_scan_impl(val);                                          \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_scan(T val) {                                                   \
    for (int i = 1; i <= 16; i *= 2) {                                   \
      val = operator()(val, simd_shuffle_and_fill_up(val, init, i));     \
    }                                                                    \
    return val;                                                          \
  }

#define DEFINE_SIMD_EXCLUSIVE_SCAN()                                     \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_exclusive_scan(T val) {                                         \
    return simd_exclusive_scan_impl(val);                                \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_exclusive_scan(T val) {                                         \
    val = simd_scan(val);                                                \
    return simd_shuffle_and_fill_up(val, init, 1);                       \
  }

template <typename U>
struct CumSum {
  DEFINE_SIMD_SCAN()
  DEFINE_SIMD_EXCLUSIVE_SCAN()

  static constexpr constant U init = static_cast<U>(0);

  template <typename T>
  U operator()(U a, T b) {
    return a + b;
  }

  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_sum(x);
  }

  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_sum(x);
  }
};

template <typename U>
struct CumProd {
  DEFINE_SIMD_SCAN()
  DEFINE_SIMD_EXCLUSIVE_SCAN()

  static constexpr constant U init = static_cast<U>(1.0f);

  template <typename T>
  U operator()(U a, T b) {
    return a * b;
  }

  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_product(x);
  }

  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_product(x);
  }
};

template <>
struct CumProd<bool> {
  static constexpr constant bool init = true;

  template <typename T>
  bool operator()(bool a, T b) {
    return a & static_cast<bool>(b);
  }

  bool simd_scan(bool x) {
    for (int i = 1; i <= 16; i *= 2) {
      bool other = simd_shuffle_and_fill_up(x, init, i);
      x &= other;
    }
    return x;
  }

  bool simd_exclusive_scan(bool x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename U>
struct CumMax {
  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  U operator()(U a, T b) {
    return (a >= b) ? a : b;
  }

  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x >= other) ? x : other;
    }
    return x;
  }

  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename U>
struct CumMin {
  static constexpr constant U init = Limits<U>::max;

  template <typename T>
  U operator()(U a, T b) {
    return (a <= b) ? a : b;
  }

  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x <= other) ? x : other;
    }
    return x;
  }

  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename T, typename U, int N_READS, bool reverse>
inline void load_unsafe(U values[N_READS], const device T* input) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] = input[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = input[i];
    }
  }
}

template <typename T, typename U, int N_READS, bool reverse>
inline void load_safe(
    U values[N_READS],
    const device T* input,
    int start,
    int total,
    U init) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] =
          (start + N_READS - i - 1 < total) ? input[i] : init;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = (start + i < total) ? input[i] : init;
    }
  }
}

template <typename U, int N_READS, bool reverse>
inline void write_unsafe(U values[N_READS], device U* out) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[N_READS - i - 1];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[i];
    }
  }
}

template <typename U, int N_READS, bool reverse>
inline void write_safe(U values[N_READS], device U* out, int start, int total) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      if (start + N_READS - i - 1 < total) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (start + i < total) {
        out[i] = values[i];
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void contiguous_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  Op op;

  // Position the pointers
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out += offset;

  // Compute the number of simd_groups
  uint simd_groups = lsize.x / simd_size;

  // Allocate memory
  U prefix = Op::init;
  U values[N_READS];
  threadgroup U simdgroup_sums[32];

  // Loop over the reduced axis in blocks of size ceildiv(axis_size,
  // N_READS*lsize)
  //    Read block
  //    Compute inclusive scan of the block
  //      Compute inclusive scan per thread
  //      Compute exclusive scan of thread sums in simdgroup
  //      Write simdgroup sums in SM
  //      Compute exclusive scan of simdgroup sums
  //      Compute the output by scanning prefix, prev_simdgroup, prev_thread,
  //      value
  //    Write block

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Compute the block offset
    uint offset = r * lsize.x * N_READS + lid.x * N_READS;

    // Read the values
    if (reverse) {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(
            values, in + axis_size - offset - N_READS);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values,
            in + axis_size - offset - N_READS,
            offset,
            axis_size,
            Op::init);
      }
    } else {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(values, in + offset);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values, in + offset, offset, axis_size, Op::init);
      }
    }

    // Compute an inclusive scan per thread
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums
    U prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);

    // Write simdgroup_sums to SM
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      U prev_simdgroup = op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
      simdgroup_sums[simd_lane_id] = prev_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute the output
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }

    // Write the values
    if (reverse) {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[axis_size - 1] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - 1 - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values,
              out + axis_size - offset - 1 - N_READS,
              offset + 1,
              axis_size);
        }
      }
    } else {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[0] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset + 1);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset + 1, offset + 1, axis_size);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Share the prefix
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void strided_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(U);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup U read_buffer[BM * BN_pad];
  U values[n_scans];
  U prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = Op::init;
  }

  // Compute offsets
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  threadgroup U* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup U* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }

    // Read in SM
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = in[index_y * stride + i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = in[index_y * stride + i];
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read strided into registers
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Perform the scan
    for (int i = 0; i < n_scans; i++) {
      values[i] = op.simd_scan(values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }

    // Write to SM
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to device memory
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
          for (int i = 0; i < N_READS; i++) {
            out[index_y * stride + i] = Op::init;
          }
        } else {
          for (int i = 0; i < N_READS; i++) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = Op::init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}
