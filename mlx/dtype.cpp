// Copyright © 2023 Apple Inc.

#include <cstdint>
#include <sstream>
#include <vector>

#include "mlx/dtype.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int num_types = 13;

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

// clang-format on

inline bool is_big_endian() {
  union ByteOrder {
    int32_t i;
    uint8_t c[4];
  };
  ByteOrder b = {0x01234567};

  return b.c[0] == 0x01;
}

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
  if (std::find(cat.kinds.begin(), cat.kinds.end(), kindof(type)) !=
      cat.kinds.end()) {
    return true;
  }
  return std::any_of(
      cat.subcategories.begin(),
      cat.subcategories.end(),
      [&type](const auto& subcategory) {
        return issubdtype(type, subcategory);
      });
}

bool issubdtype(const Dtype::Category& a, const Dtype::Category& b) {
  if (a == b) {
    return true;
  }
  return std::any_of(
      b.subcategories.begin(),
      b.subcategories.end(),
      [&a](const auto& subcategory) { return issubdtype(a, subcategory); });
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
