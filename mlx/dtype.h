// Copyright © 2023-2024 Apple Inc.

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

  enum class Category {
    complexfloating,
    floating,
    inexact,
    signedinteger,
    unsignedinteger,
    integer,
    number,
    generic
  };

  Val val;
  const uint8_t size;
  constexpr explicit Dtype(Val val, uint8_t size) : val(val), size(size){};
  constexpr operator Val() const {
    return val;
  };
};

inline constexpr Dtype bool_{Dtype::Val::bool_, sizeof(bool)};

inline constexpr Dtype uint8{Dtype::Val::uint8, sizeof(uint8_t)};
inline constexpr Dtype uint16{Dtype::Val::uint16, sizeof(uint16_t)};
inline constexpr Dtype uint32{Dtype::Val::uint32, sizeof(uint32_t)};
inline constexpr Dtype uint64{Dtype::Val::uint64, sizeof(uint64_t)};

inline constexpr Dtype int8{Dtype::Val::int8, sizeof(int8_t)};
inline constexpr Dtype int16{Dtype::Val::int16, sizeof(int16_t)};
inline constexpr Dtype int32{Dtype::Val::int32, sizeof(int32_t)};
inline constexpr Dtype int64{Dtype::Val::int64, sizeof(int64_t)};

inline constexpr Dtype float16{Dtype::Val::float16, sizeof(uint16_t)};
inline constexpr Dtype float32{Dtype::Val::float32, sizeof(float)};
inline constexpr Dtype bfloat16{Dtype::Val::bfloat16, sizeof(uint16_t)};
inline constexpr Dtype complex64{Dtype::Val::complex64, sizeof(complex64_t)};

inline constexpr Dtype::Category complexfloating =
    Dtype::Category::complexfloating;
inline constexpr Dtype::Category floating = Dtype::Category::floating;
inline constexpr Dtype::Category inexact = Dtype::Category::inexact;
inline constexpr Dtype::Category signedinteger = Dtype::Category::signedinteger;
inline constexpr Dtype::Category unsignedinteger =
    Dtype::Category::unsignedinteger;
inline constexpr Dtype::Category integer = Dtype::Category::integer;
inline constexpr Dtype::Category number = Dtype::Category::number;
inline constexpr Dtype::Category generic = Dtype::Category::generic;

bool issubdtype(const Dtype& a, const Dtype& b);
bool issubdtype(const Dtype::Category& a, const Dtype& b);
bool issubdtype(const Dtype& a, const Dtype::Category& b);
bool issubdtype(const Dtype::Category& a, const Dtype::Category& b);

Dtype promote_types(const Dtype& t1, const Dtype& t2);

inline uint8_t size_of(const Dtype& t) {
  return t.size;
}

Dtype::Kind kindof(const Dtype& t);

template <typename T>
struct TypeToDtype {
  operator Dtype();
};

// Array protocol typestring for Dtype
std::string dtype_to_array_protocol(const Dtype& t);
// Dtype from array protocol type string
Dtype dtype_from_array_protocol(std::string_view t);

} // namespace mlx::core
