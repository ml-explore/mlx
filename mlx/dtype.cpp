// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <sstream>
#include <vector>

#include "mlx/dtype.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int num_types = 13;
constexpr int num_cats = 8;

constexpr Dtype::Kind type_kinds[num_types] = {
    Dtype::Kind::b, // bool_,
    Dtype::Kind::u, // uint8,
    Dtype::Kind::u, // uint16,
    Dtype::Kind::u, // uint32,
    Dtype::Kind::u, // uint64,
    Dtype::Kind::i, // int8,
    Dtype::Kind::i, // int16,
    Dtype::Kind::i, // int32,
    Dtype::Kind::i, // int64,
    Dtype::Kind::f, // float16,
    Dtype::Kind::f, // float32,
    Dtype::Kind::V, // bfloat16,
    Dtype::Kind::c // complex64,
};

// Following Jax type promotion rules:
// https://jax.readthedocs.io/en/latest/type_promotion.html
// clang-format off
constexpr Dtype type_rules[num_types][num_types] = {
// bool       uint8      uint16     uint32     uint64     int8       int16      int32      int64      float16    float32    bfloat16   complex64
  {bool_,     uint8,     uint16,    uint32,    uint64,    int8,      int16,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // bool
  {uint8,     uint8,     uint16,    uint32,    uint64,    int16,     int16,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // uint8
  {uint16,    uint16,    uint16,    uint32,    uint64,    int32,     int32,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // uint16
  {uint32,    uint32,    uint32,    uint32,    uint64,    int64,     int64,     int64,     int64,     float16,   float32,   bfloat16,  complex64}, // uint32
  {uint64,    uint64,    uint64,    uint64,    uint64,    float32,   float32,   float32,   float32,   float16,   float32,   bfloat16,  complex64}, // uint64
  {int8,      int16,     int32,     int64,     float32,   int8,      int16,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // int8
  {int16,     int16,     int32,     int64,     float32,   int16,     int16,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // int16
  {int32,     int32,     int32,     int64,     float32,   int32,     int32,     int32,     int64,     float16,   float32,   bfloat16,  complex64}, // int32
  {int64,     int64,     int64,     int64,     float32,   int64,     int64,     int64,     int64,     float16,   float32,   bfloat16,  complex64}, // int64
  {float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float32,   float32,   complex64}, // float16
  {float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   complex64}, // float32
  {bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  float32,   float32,   bfloat16,  complex64}, // bfloat16
  {complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64}, // complex64
};


constexpr bool subcategory_to_category[num_cats][num_cats] = {
// complexfloating floating inexact signedinteger unsignedinteger integer number generic
  {true,           false,   true,   false,        false,          false,  true,  true}, // complexfloating
  {false,          true,    true,   false,        false,          false,  true,  true}, // floating
  {false,          false,   true,   false,        false,          false,  true,  true}, // inexact
  {false,          false,   false,  true,         false,          true,   true,  true}, // signedinteger
  {false,          false,   false,  false,        true,           true,   true,  true}, // unsignedinteger
  {false,          false,   false,  false,        false,          true,   true,  true}, // integer
  {false,          false,   false,  false,        false,          false,  true,  true}, // number
  {false,          false,   false,  false,        false,          false,  false, true}, // generic
};

constexpr Dtype::Category type_to_category[num_types] = {
    Dtype::Category::generic, // bool_,
    Dtype::Category::unsignedinteger, // uint8,
    Dtype::Category::unsignedinteger, // uint16,
    Dtype::Category::unsignedinteger, // uint32,
    Dtype::Category::unsignedinteger, // uint64,
    Dtype::Category::signedinteger, // int8,
    Dtype::Category::signedinteger, // int16,
    Dtype::Category::signedinteger, // int32,
    Dtype::Category::signedinteger, // int64,
    Dtype::Category::floating, // float16,
    Dtype::Category::floating, // float32,
    Dtype::Category::floating, // bfloat16,
    Dtype::Category::complexfloating, // complex64,
};

// clang-format on

} // namespace

Dtype promote_types(const Dtype& t1, const Dtype& t2) {
  return Dtype(type_rules[static_cast<int>(t1.val)][static_cast<int>(t2.val)]);
}

Dtype::Kind kindof(const Dtype& t) {
  return type_kinds[static_cast<int>(t.val)];
}

template <>
TypeToDtype<bool>::operator Dtype() {
  return bool_;
}

template <>
TypeToDtype<uint8_t>::operator Dtype() {
  return uint8;
}

template <>
TypeToDtype<uint16_t>::operator Dtype() {
  return uint16;
}

template <>
TypeToDtype<uint32_t>::operator Dtype() {
  return uint32;
}

template <>
TypeToDtype<uint64_t>::operator Dtype() {
  return uint64;
}

template <>
TypeToDtype<int8_t>::operator Dtype() {
  return int8;
}

template <>
TypeToDtype<int16_t>::operator Dtype() {
  return int16;
}

template <>
TypeToDtype<int32_t>::operator Dtype() {
  return int32;
}

template <>
TypeToDtype<int64_t>::operator Dtype() {
  return int64;
}

template <>
TypeToDtype<float16_t>::operator Dtype() {
  return float16;
}

template <>
TypeToDtype<float>::operator Dtype() {
  return float32;
}

template <>
TypeToDtype<double>::operator Dtype() {
  return float32;
}

template <>
TypeToDtype<bfloat16_t>::operator Dtype() {
  return bfloat16;
}

template <>
TypeToDtype<complex64_t>::operator Dtype() {
  return complex64;
}

bool issubdtype(const Dtype& a, const Dtype& b) {
  return a == b;
}

bool issubdtype(const Dtype::Category& cat, const Dtype& type) {
  return false;
}

bool issubdtype(const Dtype& type, const Dtype::Category& cat) {
  return issubdtype(type_to_category[static_cast<uint32_t>(type.val)], cat);
}

bool issubdtype(const Dtype::Category& a, const Dtype::Category& b) {
  return subcategory_to_category[static_cast<uint32_t>(a)]
                                [static_cast<uint32_t>(b)];
}

// Array protocol typestring for Dtype
std::string dtype_to_array_protocol(const Dtype& t) {
  std::ostringstream r;
  if (size_of(t) > 1)
    r << (is_big_endian() ? ">" : "<");
  else
    r << "|";
  r << kindof(t) << (int)size_of(t);
  return r.str();
}

// Dtype from array protocol type string
Dtype dtype_from_array_protocol(const std::string& t) {
  if (t.length() == 2 || t.length() == 3) {
    std::string r = t.length() == 3 ? t.substr(1, 2) : t;

    if (r == "V2") {
      return bfloat16;
    }

    uint8_t size = r[1] - '0';

    switch (r[0]) {
      case 'b': {
        if (size == 1)
          return bool_;
      }
      case 'i': {
        if (size == 1)
          return int8;
        else if (size == 2)
          return int16;
        else if (size == 4)
          return int32;
        else if (size == 8)
          return int64;
      }
      case 'u': {
        if (size == 1)
          return uint8;
        else if (size == 2)
          return uint16;
        else if (size == 4)
          return uint32;
        else if (size == 8)
          return uint64;
      }
      case 'f': {
        if (size == 2)
          return float16;
        else if (size == 4)
          return float32;
      }
      case 'c': {
        return complex64;
      }
    }
  }

  throw std::invalid_argument(
      "[from_str] Invalid array protocol type-string: " + t);
}

} // namespace mlx::core
