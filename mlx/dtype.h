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

  struct Category {
    enum class Val {
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
    const std::vector<const Kind> kinds;

    const std::vector<const Category> subcategories;

    Category(
        const Val val,
        const std::vector<const Kind>& kinds,
        const std::vector<const Category>& subcategories)
        : val(val), kinds(kinds), subcategories(subcategories) {}

    constexpr operator Val() const {
      return val;
    }
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

static const Dtype::Category complexfloating = Dtype::Category(
    Dtype::Category::Val::complexfloating,
    {Dtype::Kind::c},
    {});
static const Dtype::Category floating = Dtype::Category(
    Dtype::Category::Val::floating,
    {Dtype::Kind::f, Dtype::Kind::V},
    {});

static const Dtype::Category inexact = Dtype::Category(
    Dtype::Category::Val::inexact,
    {},
    {floating, complexfloating});

static const Dtype::Category signedinteger =
    Dtype::Category(Dtype::Category::Val::signedinteger, {Dtype::Kind::i}, {});
static const Dtype::Category unsignedinteger = Dtype::Category(
    Dtype::Category::Val::unsignedinteger,
    {Dtype::Kind::u},
    {});
static const Dtype::Category integer = Dtype::Category(
    Dtype::Category::Val::integer,
    {},
    {signedinteger, unsignedinteger});

static const Dtype::Category number =
    Dtype::Category(Dtype::Category::Val::number, {}, {integer, inexact});

static const Dtype::Category generic =
    Dtype::Category(Dtype::Category::Val::generic, {Dtype::Kind::b}, {number});

bool issubdtype(const Dtype& a, const Dtype& b);
bool issubdtype(const Dtype::Category& a, const Dtype& b);
bool issubdtype(const Dtype& a, const Dtype::Category& b);
bool issubdtype(const Dtype::Category& a, const Dtype::Category& b);

Dtype promote_types(const Dtype& t1, const Dtype& t2);

inline uint8_t size_of(const Dtype& t) {
  return t.size;
}

Dtype::Kind kindof(const Dtype& t);

/**
 * equivalent to `issubdtype(t, unsignedinteger) || issubdtype(t, bool_)`
 */
inline bool is_unsigned(const Dtype& t) {
  return kindof(t) == Dtype::Kind::u || kindof(t) == Dtype::Kind::b;
}

/**
 * equivalent to `issubdtype(t, floating) && !issubdtype(t, complexfloating)`
 */
inline bool is_floating_point(const Dtype& t) {
  return kindof(t) == Dtype::Kind::f || kindof(t) == Dtype::Kind::V ||
      kindof(t) == Dtype::Kind::c;
}

/**
 * equivalent to `issubdtype(t, complexfloating)`
 */
inline bool is_complex(const Dtype& t) {
  return kindof(t) == Dtype::Kind::c;
}

/**
 * equivalent to `issubdtype(t, integer) || issubdtype(t, bool_)`
 */
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
