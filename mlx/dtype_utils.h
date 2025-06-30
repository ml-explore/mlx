// Copyright © 2025 Apple Inc.

#pragma once

#include <sstream>

#include "mlx/dtype.h"
#include "mlx/utils.h"

namespace mlx::core {

// Return string representation of dtype.
const char* dtype_to_string(Dtype arg);

#define MLX_INTERNAL_DTYPE_SWITCH_CASE(DTYPE, TYPE) \
  case DTYPE:                                       \
    f(type_identity<TYPE>{});                       \
    break

#define MLX_INTERNAL_DTYPE_SWITCH_INTS()            \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(int8, int8_t);     \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(int16, int16_t);   \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(int32, int32_t);   \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(int64, int64_t);   \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(uint8, uint8_t);   \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(uint16, uint16_t); \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(uint32, uint32_t); \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(uint64, uint64_t)

#define MLX_INTERNAL_DTYPE_SWITCH_FLOATS()              \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(float16, float16_t);   \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(bfloat16, bfloat16_t); \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(float32, float);       \
  MLX_INTERNAL_DTYPE_SWITCH_CASE(float64, double)

// This already exists in C++20 but in C++20 we can also just use templated
// lambdas which will make this so much nicer.
template <typename T>
struct type_identity {
  using type = T;
};

#define MLX_GET_TYPE(x) typename decltype(x)::type
#define MLX_GET_VALUE(x) decltype(x)::value

template <typename F>
void dispatch_all_types(Dtype dt, F&& f) {
  switch (dt) {
    MLX_INTERNAL_DTYPE_SWITCH_CASE(bool_, bool);
    MLX_INTERNAL_DTYPE_SWITCH_INTS();
    MLX_INTERNAL_DTYPE_SWITCH_FLOATS();
    MLX_INTERNAL_DTYPE_SWITCH_CASE(complex64, complex64_t);
  }
}

template <typename F>
void dispatch_int_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    MLX_INTERNAL_DTYPE_SWITCH_INTS();
    default:
      std::ostringstream msg;
      msg << tag << " Only integer types supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_float_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    MLX_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only float types supported but " << dt << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_int_float_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    MLX_INTERNAL_DTYPE_SWITCH_INTS();
    MLX_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only integer and float types supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_real_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    MLX_INTERNAL_DTYPE_SWITCH_CASE(bool_, bool);
    MLX_INTERNAL_DTYPE_SWITCH_INTS();
    MLX_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only real numbers supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

} // namespace mlx::core
