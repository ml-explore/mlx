// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

enum class QuantMode { Affine, Mxfp4, Mxfp8, Nvfp4 };

template <typename OutT, typename EncodedT>
struct DecodeValue {
  [[clang::always_inline]] OutT operator()(uint8_t v) const {
    return OutT(*(thread EncodedT*)(&v));
  }
};

// Specialization for Affine (plain integer cast)
template <typename OutT>
struct DecodeValue<OutT, void> {
  [[clang::always_inline]] OutT operator()(uint8_t v) const {
    return OutT(v);
  }
};

template <QuantMode mode>
struct QuantConfig;

template <>
struct QuantConfig<QuantMode::Affine> {
  static constant constexpr bool has_bias = true;

  using value_type = void;
  using scale_type = void;

  template <typename T>
  using scale_storage_t = T;
};

template <>
struct QuantConfig<QuantMode::Mxfp4> {
  static constant constexpr bool has_bias = false;

  using value_type = fp4_e2m1;
  using scale_type = fp8_e8m0;

  template <typename T>
  using scale_storage_t = uint8_t;
};

template <>
struct QuantConfig<QuantMode::Nvfp4> {
  static constant constexpr bool has_bias = false;

  using value_type = fp4_e2m1;
  using scale_type = fp8_e4m3;

  template <typename T>
  using scale_storage_t = uint8_t;
};

template <>
struct QuantConfig<QuantMode::Mxfp8> {
  static constant constexpr bool has_bias = false;

  using value_type = fp8_e4m3;
  using scale_type = fp8_e8m0;

  template <typename T>
  using scale_storage_t = uint8_t;
};

template <QuantMode mode, typename T>
struct Dequant {
  using Cfg = QuantConfig<mode>;

  [[clang::always_inline]] T raw(uint8_t v) const {
    return DecodeValue<T, typename Cfg::value_type>{}(v);
  }

  [[clang::always_inline]] T scale(
      typename Cfg::template scale_storage_t<T> s) const {
    if constexpr (metal::is_same_v<typename Cfg::scale_type, void>) {
      return s;
    } else {
      return DecodeValue<T, typename Cfg::scale_type>{}(s);
    }
  }

  [[clang::always_inline]] T operator()(uint8_t v, T s, T bias) const {
    if constexpr (Cfg::has_bias) {
      return fma(s, raw(v), bias);
    } else {
      return s * raw(v);
    }
  }
};

// Pack metadata and unpackers for arbitrary bit-widths (wsize fixed at 32 bits)
template <int bits>
struct PackInfo {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "PackInfo only supports bits in {2,3,4,5,6,8}");

  static constant constexpr int pack_factor =
      (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
  static constant constexpr int bytes_per_pack =
      ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
};

template <int bits>
struct PackReader {
  static constant constexpr int pack_factor = PackInfo<bits>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<bits>::bytes_per_pack;
  static constant constexpr uint64_t mask = (uint64_t(1) << bits) - 1;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p,
      thread uint8_t (&out)[pack_factor]) {
    uint64_t packed = load_packed(p);
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; ++i) {
      out[i] = static_cast<uint8_t>((packed >> (bits * i)) & mask);
    }
  }

 private:
  [[gnu::always_inline]] static uint64_t load_packed(const device uint8_t* p) {
    if constexpr (bytes_per_pack == 4) {
      return static_cast<uint64_t>(
          *(reinterpret_cast<const device uint32_t*>(p)));
    } else {
      uint64_t packed = 0;
#pragma clang loop unroll(full)
      for (int i = 0; i < bytes_per_pack; ++i) {
        packed |= static_cast<uint64_t>(p[i]) << (8 * i);
      }
      return packed;
    }
  }
};

// Pointer wrapper for quantized data that handles byte-level addressing
// correctly for all bit widths. For non-4-byte-aligned packs (3, 5, 6-bit),
template <int bits>
class QuantDataPtr {
  const device uint8_t* byte_ptr_;

 public:
  static constant constexpr int pack_factor = PackInfo<bits>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<bits>::bytes_per_pack;

  // Initialize from base pointer, head stride (in uint32 units), head index,
  // and element index
  [[clang::always_inline]] QuantDataPtr(
      const device uint32_t* base,
      size_t head_stride,
      int head_idx,
      int elem_idx) {
    int packed_idx = elem_idx / pack_factor;
    byte_ptr_ = reinterpret_cast<const device uint8_t*>(base) +
        head_idx * head_stride * 4 + // head_stride is in uint32 units
        packed_idx * bytes_per_pack;
  }

  // Advance by number of elements
  [[clang::always_inline]] void advance(int num_elements) {
    byte_ptr_ += num_elements * bits / 8;
  }

  // Get raw pointer for passing to dot/accumulate functions
  [[clang::always_inline]] const device uint32_t* ptr() const {
    return reinterpret_cast<const device uint32_t*>(byte_ptr_);
  }
};

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_aligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <
    bool rows_aligned,
    bool cols_aligned,
    bool transpose,
    typename T,
    typename mma_t,
    typename loader_a_t,
    typename loader_b_t>
METAL_FUNC void gemm_loop_unaligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations,
    const short tgp_bm,
    const short tgp_bn,
    const short tgp_bk) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    if (rows_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(short2(tgp_bk, tgp_bm));
    }
    if (cols_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(
          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_finalize(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const short2 tile_a,
    const short2 tile_b) {
  loader_a.load_safe(tile_a);
  loader_b.load_safe(tile_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
}
