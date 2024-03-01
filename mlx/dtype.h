// Copyright © 2023 Apple Inc.

#pragma once

#include <complex>
#include <cstdint>
#include <ostream>
#include <string>

#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"

namespace mlx::core {

struct Dtype {
  enum class Val {
    bool_,
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    bfloat16,
    complex64,
  };

  enum class Kind {
    b, /* bool */
    u, /* unsigned int */
    i, /* signed int */
    f, /* float */
    c, /* complex */
    V, /* void - used for brain float */
  };

  Val val;
  const uint8_t size;
  constexpr explicit Dtype(Val val, uint8_t size) : val(val), size(size){};
  constexpr operator Val() const {
    return val;
  };
};

static constexpr Dtype bool_{Dtype::Val::bool_, sizeof(bool)};

static constexpr Dtype uint8{Dtype::Val::uint8, sizeof(uint8_t)};
static constexpr Dtype uint16{Dtype::Val::uint16, sizeof(uint16_t)};
static constexpr Dtype uint32{Dtype::Val::uint32, sizeof(uint32_t)};
static constexpr Dtype uint64{Dtype::Val::uint64, sizeof(uint64_t)};

static constexpr Dtype int8{Dtype::Val::int8, sizeof(int8_t)};
static constexpr Dtype int16{Dtype::Val::int16, sizeof(int16_t)};
static constexpr Dtype int32{Dtype::Val::int32, sizeof(int32_t)};
static constexpr Dtype int64{Dtype::Val::int64, sizeof(int64_t)};

static constexpr Dtype float16{Dtype::Val::float16, sizeof(uint16_t)};
static constexpr Dtype float32{Dtype::Val::float32, sizeof(float)};
static constexpr Dtype bfloat16{Dtype::Val::bfloat16, sizeof(uint16_t)};
static constexpr Dtype complex64{Dtype::Val::complex64, sizeof(complex64_t)};

Dtype promote_types(const Dtype& t1, const Dtype& t2);

inline uint8_t size_of(const Dtype& t) {
  return t.size;
}

Dtype::Kind kindof(const Dtype& t);

inline bool is_unsigned(const Dtype& t) {
  return kindof(t) == Dtype::Kind::u || kindof(t) == Dtype::Kind::b;
}

inline bool is_floating_point(const Dtype& t) {
  return kindof(t) == Dtype::Kind::f || kindof(t) == Dtype::Kind::V ||
      kindof(t) == Dtype::Kind::c;
}

inline bool is_complex(const Dtype& t) {
  return kindof(t) == Dtype::Kind::c;
}

inline bool is_integral(const Dtype& t) {
  return !(is_floating_point(t));
}

template <typename T>
struct TypeToDtype {
  operator Dtype();
};

// Array protocol typestring for Dtype
std::string dtype_to_array_protocol(const Dtype& t);
// Dtype from array protocol type string
Dtype dtype_from_array_protocol(const std::string& t);

} // namespace mlx::core
