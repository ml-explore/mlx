// Copyright © 2026 Apple Inc.

// Radix select partition: one threadgroup per row finds the k-th key with a
// histogram pass per digit, then writes the row as below | equal | above it.

using namespace metal;

static constant constexpr int radix_n_reads = 8;
// Keys a thread loads together in a select pass, so the loads are in flight
// at the same time
static constant constexpr int radix_n_loads = 16;
static constant constexpr int radix_simd_size = 32;
static constant constexpr int radix_threshold_sides = 3;

template <int BLOCK_THREADS>
struct RadixLayout {
  static constant constexpr int simd_groups = BLOCK_THREADS / radix_simd_size;
  // Wide digits need fewer passes and leave fewer matching keys per pass,
  // 256 threads spread the 2048 bins at no extra cost per thread
  static constant constexpr int digit_bits = BLOCK_THREADS >= 256 ? 11 : 8;
  static constant constexpr int bins = 1 << digit_bits;
  static constant constexpr int bins_per_thread = bins / BLOCK_THREADS;
  // Scan scratch per value: one sum per simdgroup and the total after them
  static constant constexpr int scan_size = simd_groups + 1;
  // Candidates cache, 8 keys per thread so small groups keep more rows per core
  static constant constexpr int candidates_cache_size = 8 * BLOCK_THREADS;

  // Digit passes over a key of `key_bits`, the last digit may be narrower
  static METAL_FUNC constexpr int passes(int key_bits) {
    return (key_bits + digit_bits - 1) / digit_bits;
  }
  static METAL_FUNC constexpr int shift(int key_bits, int pass) {
    int shift = key_bits - digit_bits * (pass + 1);
    return shift > 0 ? shift : 0;
  }

  static_assert(
      BLOCK_THREADS % radix_simd_size == 0,
      "BLOCK_THREADS must be a multiple of the simd size");
  static_assert(
      bins % BLOCK_THREADS == 0,
      "BLOCK_THREADS must divide the number of bins");
};

///////////////////////////////////////////////////////////////////////////////
// Radix keys
///////////////////////////////////////////////////////////////////////////////

// Unsigned key of the same width as T, so the digits cover the whole value.
// Signed ints and positive floats flip the sign bit, negative floats flip every
// bit, NaN gets the largest key so it sorts last.
template <typename T>
struct RadixKeyTraits {
  using KeyT = metal::conditional_t<
      sizeof(T) == 1,
      uint8_t,
      metal::conditional_t<
          sizeof(T) == 2,
          uint16_t,
          metal::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
  static METAL_FUNC KeyT to_key(T v) {
    constexpr KeyT sign_bit = KeyT(1) << (8 * sizeof(T) - 1);
    if constexpr (metal::is_floating_point_v<T>) {
      if (metal::isnan(v)) {
        return KeyT(~KeyT(0));
      }
      KeyT bits = as_type<KeyT>(v);
      return bits ^ ((bits & sign_bit) ? KeyT(~KeyT(0)) : sign_bit);
    } else if constexpr (metal::is_signed_v<T>) {
      return as_type<KeyT>(v) ^ sign_bit;
    } else {
      return v;
    }
  }
};

// Complex keys order by the real part, then the imaginary part, and a NaN in
// either part sorts last
template <>
struct RadixKeyTraits<complex64_t> {
  using KeyT = uint64_t;
  static METAL_FUNC KeyT to_key(complex64_t v) {
    if (metal::isnan(v.real) || metal::isnan(v.imag)) {
      return ~KeyT(0);
    }
    return (KeyT(RadixKeyTraits<float>::to_key(v.real))
            << (8 * sizeof(float))) |
        RadixKeyTraits<float>::to_key(v.imag);
  }
};

// Side of the threshold a key goes to: 0 below, 1 equal, 2 above
template <typename KeyT>
METAL_FUNC int radix_threshold_side(KeyT key, KeyT threshold) {
  return key < threshold ? 0 : (key == threshold ? 1 : 2);
}

///////////////////////////////////////////////////////////////////////////////
// Threadgroup scan
///////////////////////////////////////////////////////////////////////////////

// Exclusive scan of N values per thread over the threadgroup. `values` become
// the sums before this thread, `totals` the sums over all threads.
template <int BLOCK_THREADS, int N>
METAL_FUNC void threadgroup_exclusive_sum(
    thread uint* values,
    thread uint* totals,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* sums) {
  constexpr int simd_groups = RadixLayout<BLOCK_THREADS>::simd_groups;
  constexpr int scan_size = RadixLayout<BLOCK_THREADS>::scan_size;
  constexpr int last_lane = radix_simd_size - 1;

  // Scan inside each simdgroup, the last lane stores the simdgroup sum
  uint prev[N];
  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < N; c++) {
    prev[c] = simd_prefix_exclusive_sum(values[c]);
    if (simd_lane_id == last_lane) {
      sums[c * scan_size + simd_group_id] = prev[c] + values[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Scan the simdgroup sums, the total lands right after them
  if (simd_group_id == 0) {
    bool valid = simd_lane_id < simd_groups;
    MLX_MTL_PRAGMA_UNROLL
    for (int c = 0; c < N; c++) {
      threadgroup uint* group_sums = sums + c * scan_size;
      uint sum = valid ? group_sums[simd_lane_id] : 0;
      uint scanned = simd_prefix_exclusive_sum(sum);
      if (valid) {
        group_sums[simd_lane_id] = scanned;
      }
      if (simd_lane_id == last_lane) {
        group_sums[simd_groups] = scanned + sum;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < N; c++) {
    threadgroup uint* group_sums = sums + c * scan_size;
    values[c] = prev[c] + group_sums[simd_group_id];
    totals[c] = group_sums[simd_groups];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Radix select
///////////////////////////////////////////////////////////////////////////////

struct RadixBin {
  uint digit;
  uint below;
  uint count;
};

// Prefix of the k-th key fixed so far with the counts around it, k counts
// from the bottom of the keys that still match the prefix
template <typename KeyT>
struct RadixSelectState {
  KeyT prefix;
  KeyT fixed;
  uint k;
  uint n_below;
  uint n_equal;
};

template <typename KeyT>
METAL_FUNC RadixSelectState<KeyT> radix_select_start(uint k) {
  RadixSelectState<KeyT> state;
  state.prefix = 0;
  state.fixed = 0;
  state.k = k;
  state.n_below = 0;
  state.n_equal = 0;
  return state;
}

template <int DIGIT_BITS, typename KeyT>
METAL_FUNC void radix_select_advance(
    thread RadixSelectState<KeyT>& state,
    RadixBin bin,
    int shift) {
  state.n_below += bin.below;
  state.k -= bin.below;
  state.prefix |= KeyT(bin.digit) << shift;
  state.fixed |= KeyT((1 << DIGIT_BITS) - 1) << shift;
  state.n_equal = bin.count;
}

// Counts the key if its fixed digits match the prefix. Returns whether it did.
template <int DIGIT_BITS, typename KeyT>
METAL_FUNC bool radix_count_key(
    KeyT key,
    const thread RadixSelectState<KeyT>& state,
    int shift,
    threadgroup atomic_uint* hist) {
  if ((key & state.fixed) != state.prefix) {
    return false;
  }
  uint digit = uint((key >> shift) & KeyT((1 << DIGIT_BITS) - 1));
  atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
  return true;
}

METAL_FUNC uint radix_bin_count(threadgroup atomic_uint* hist, uint bin) {
  return atomic_load_explicit(&hist[bin], memory_order_relaxed);
}

template <int BINS_PER_THREAD>
METAL_FUNC uint radix_bins_sum(threadgroup atomic_uint* hist, uint first) {
  uint sum = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < BINS_PER_THREAD; j++) {
    sum += radix_bin_count(hist, first + j);
  }
  return sum;
}

// Walks the bins of one thread with `below` keys before them. Returns whether
// the sum from the bottom reaches k inside them, then `bin` is that bin.
template <int BINS_PER_THREAD>
METAL_FUNC bool radix_bins_walk(
    threadgroup atomic_uint* hist,
    uint first,
    uint below,
    uint k,
    thread RadixBin& bin) {
  bool found = false;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < BINS_PER_THREAD; j++) {
    uint count = radix_bin_count(hist, first + j);
    if (!found && below < k && below + count >= k) {
      found = true;
      bin.digit = first + j;
      bin.below = below;
      bin.count = count;
    }
    below += count;
  }
  return found;
}

// Scans the histogram from the bottom with the whole threadgroup,
// `bins_per_thread` bins per thread. Returns the bin of the k-th smallest key.
template <int BLOCK_THREADS>
METAL_FUNC RadixBin radix_find_bin(
    threadgroup atomic_uint* hist,
    uint k,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* bin_count_simdgroup_sums,
    threadgroup RadixBin& kth_bin) {
  constexpr int bins_per_thread = RadixLayout<BLOCK_THREADS>::bins_per_thread;

  uint first = lid_x * bins_per_thread;
  uint below = radix_bins_sum<bins_per_thread>(hist, first);
  uint total;
  threadgroup_exclusive_sum<BLOCK_THREADS, 1>(
      &below, &total, simd_lane_id, simd_group_id, bin_count_simdgroup_sums);

  // The sum from the bottom reaches k inside exactly one bin
  RadixBin bin;
  if (radix_bins_walk<bins_per_thread>(hist, first, below, k, bin)) {
    kth_bin = bin;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return kth_bin;
}

// Source of a select pass: the row, the row while it fills the candidates
// cache, or the cache only
enum class RadixPass { Row, FillCache, Cache };

template <typename T, int BLOCK_THREADS>
METAL_FUNC void radix_load_keys(
    const device T* in,
    int i,
    thread typename RadixKeyTraits<T>::KeyT* keys) {
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < radix_n_loads; r++) {
    keys[r] = RadixKeyTraits<T>::to_key(in[i + r * BLOCK_THREADS]);
  }
}

template <int DIGIT_BITS, typename KeyT>
METAL_FUNC void radix_count_row_key(
    KeyT key,
    const thread RadixSelectState<KeyT>& state,
    int shift,
    RadixPass pass,
    threadgroup atomic_uint* hist,
    threadgroup KeyT* candidates_cache,
    threadgroup atomic_uint& candidates_cache_size) {
  if (radix_count_key<DIGIT_BITS>(key, state, shift, hist) &&
      pass == RadixPass::FillCache) {
    uint slot = atomic_fetch_add_explicit(
        &candidates_cache_size, 1u, memory_order_relaxed);
    candidates_cache[slot] = key;
  }
}

// Finds the k-th smallest element of the row, one histogram pass per digit
// from the top. Once the keys that still match the prefix fit in the
// candidates cache, the later passes read the cache instead of the row.
template <typename T, int BLOCK_THREADS>
METAL_FUNC RadixSelectState<typename RadixKeyTraits<T>::KeyT> radix_select_row(
    const device T* in,
    int n,
    uint k,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup atomic_uint* hist,
    threadgroup typename RadixKeyTraits<T>::KeyT* candidates_cache,
    threadgroup atomic_uint& candidates_cache_size,
    threadgroup uint* bin_count_simdgroup_sums,
    threadgroup RadixBin& kth_bin) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  using Layout = RadixLayout<BLOCK_THREADS>;
  constexpr int key_bits = 8 * sizeof(KeyT);

  auto state = radix_select_start<KeyT>(k);
  uint n_candidates = 0;
  RadixPass pass = RadixPass::Row;

  for (int p = 0; p < Layout::passes(key_bits); p++) {
    int shift = Layout::shift(key_bits, p);
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < Layout::bins_per_thread; j++) {
      atomic_store_explicit(
          &hist[lid_x + j * BLOCK_THREADS], 0u, memory_order_relaxed);
    }
    if (lid_x == 0) {
      atomic_store_explicit(&candidates_cache_size, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count the keys that still match the fixed prefix
    if (pass == RadixPass::Cache) {
      for (uint i = lid_x; i < n_candidates; i += BLOCK_THREADS) {
        radix_count_key<Layout::digit_bits>(
            candidates_cache[i], state, shift, hist);
      }
    } else {
      int i = lid_x;
      for (; i + (radix_n_loads - 1) * BLOCK_THREADS < n;
           i += radix_n_loads * BLOCK_THREADS) {
        KeyT keys[radix_n_loads];
        radix_load_keys<T, BLOCK_THREADS>(in, i, keys);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < radix_n_loads; r++) {
          radix_count_row_key<Layout::digit_bits>(
              keys[r],
              state,
              shift,
              pass,
              hist,
              candidates_cache,
              candidates_cache_size);
        }
      }
      for (; i < n; i += BLOCK_THREADS) {
        radix_count_row_key<Layout::digit_bits>(
            RadixKeyTraits<T>::to_key(in[i]),
            state,
            shift,
            pass,
            hist,
            candidates_cache,
            candidates_cache_size);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pass == RadixPass::FillCache) {
      n_candidates =
          atomic_load_explicit(&candidates_cache_size, memory_order_relaxed);
      pass = RadixPass::Cache;
    }

    RadixBin bin = radix_find_bin<BLOCK_THREADS>(
        hist,
        state.k,
        lid_x,
        simd_lane_id,
        simd_group_id,
        bin_count_simdgroup_sums,
        kth_bin);
    radix_select_advance<Layout::digit_bits>(state, bin, shift);

    // Once the candidates fit, the next pass copies them to the cache
    if (pass == RadixPass::Row && bin.count <= Layout::candidates_cache_size) {
      pass = RadixPass::FillCache;
    }
  }
  return state;
}

///////////////////////////////////////////////////////////////////////////////
// Partition write
///////////////////////////////////////////////////////////////////////////////

// Writes one chunk of BLOCK_THREADS * radix_n_reads elements from `j`, one
// threadgroup scan gives each thread its output positions. A FULL chunk
// ends before `end`, so it needs no bounds checks.
template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int BLOCK_THREADS,
    bool FULL>
METAL_FUNC void radix_partition_write_chunk(
    const device T* in,
    device U* out,
    int j,
    int end,
    typename RadixKeyTraits<T>::KeyT threshold,
    thread uint* side_offset,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* threshold_side_count_simdgroup_sums) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;

  T vals[radix_n_reads];
  int threshold_side[radix_n_reads];
  uint pos[radix_threshold_sides] = {0, 0, 0};

  // Side of the threshold for radix_n_reads consecutive elements
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < radix_n_reads; r++) {
    int i = j + lid_x * radix_n_reads + r;
    if (FULL || i < end) {
      vals[r] = in[i];
      KeyT key = RadixKeyTraits<T>::to_key(vals[r]);
      threshold_side[r] = radix_threshold_side(key, threshold);
      pos[threshold_side[r]]++;
    }
  }

  // Offsets of this thread inside the chunk and the chunk totals
  uint totals[radix_threshold_sides];
  threadgroup_exclusive_sum<BLOCK_THREADS, radix_threshold_sides>(
      pos,
      totals,
      simd_lane_id,
      simd_group_id,
      threshold_side_count_simdgroup_sums);

  // Add the offsets of each side, then write
  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < radix_threshold_sides; c++) {
    pos[c] += side_offset[c];
    side_offset[c] += totals[c];
  }
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < radix_n_reads; r++) {
    int i = j + lid_x * radix_n_reads + r;
    if (FULL || i < end) {
      if constexpr (ARG_PARTITION) {
        out[pos[threshold_side[r]]] = U(i);
      } else {
        out[pos[threshold_side[r]]] = U(vals[r]);
      }
      pos[threshold_side[r]]++;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Writes the elements in [start, end) of the row below, equal and above the
// threshold, keeping the input order inside each side. `side_offset` holds
// where each side continues in the output and moves along.
template <typename T, typename U, bool ARG_PARTITION, int BLOCK_THREADS>
METAL_FUNC void radix_partition_write(
    const device T* in,
    device U* out,
    int start,
    int end,
    typename RadixKeyTraits<T>::KeyT threshold,
    thread uint* side_offset,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* threshold_side_count_simdgroup_sums) {
  constexpr int chunk_size = BLOCK_THREADS * radix_n_reads;

  for (int j = start; j < end; j += chunk_size) {
    if (j + chunk_size <= end) {
      radix_partition_write_chunk<T, U, ARG_PARTITION, BLOCK_THREADS, true>(
          in,
          out,
          j,
          end,
          threshold,
          side_offset,
          lid_x,
          simd_lane_id,
          simd_group_id,
          threshold_side_count_simdgroup_sums);
    } else {
      radix_partition_write_chunk<T, U, ARG_PARTITION, BLOCK_THREADS, false>(
          in,
          out,
          j,
          end,
          threshold,
          side_offset,
          lid_x,
          simd_lane_id,
          simd_group_id,
          threshold_side_count_simdgroup_sums);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Partition kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, bool ARG_PARTITION, int BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_partition(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& axis_size [[buffer(2)]],
    const constant int& kth [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  using Layout = RadixLayout<BLOCK_THREADS>;

  threadgroup atomic_uint hist[Layout::bins];
  threadgroup KeyT candidates_cache[Layout::candidates_cache_size];
  threadgroup atomic_uint candidates_cache_size;
  threadgroup uint bin_count_simdgroup_sums[Layout::scan_size];
  threadgroup uint threshold_side_count_simdgroup_sums
      [radix_threshold_sides * Layout::scan_size];
  threadgroup RadixBin kth_bin;

  int64_t offset = int64_t(gid.y) * axis_size;
  in += offset;
  out += offset;

  // kth is zero based, k is a count
  uint k = kth + 1;
  auto state = radix_select_row<T, BLOCK_THREADS>(
      in,
      axis_size,
      k,
      lid.x,
      simd_lane_id,
      simd_group_id,
      hist,
      candidates_cache,
      candidates_cache_size,
      bin_count_simdgroup_sums,
      kth_bin);
  uint side_offset[radix_threshold_sides] = {
      0, state.n_below, state.n_below + state.n_equal};
  radix_partition_write<T, U, ARG_PARTITION, BLOCK_THREADS>(
      in,
      out,
      0,
      axis_size,
      state.prefix,
      side_offset,
      lid.x,
      simd_lane_id,
      simd_group_id,
      threshold_side_count_simdgroup_sums);
}
