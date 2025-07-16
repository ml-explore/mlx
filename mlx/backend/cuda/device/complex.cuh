// Copyright © 2025 Apple Inc.

#pragma once

// Make multiplication and division faster.
#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS

#include <cuda/std/complex>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

// TODO: Consider using a faster implementation as cuda::std::complex has to
// conform to C++ standard.
template <typename T>
using complex_t = cuda::std::complex<T>;

using complex64_t = complex_t<float>;
using complex128_t = complex_t<double>;

template <typename T>
struct is_complex : cuda::std::false_type {};

template <typename T>
struct is_complex<cuda::std::complex<T>> : cuda::std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// cuda::std::complex is missing some operators.
template <typename T>
inline __host__ __device__ complex_t<T> operator%(
    complex_t<T> a,
    complex_t<T> b) {
  T r = a.real() - floor(a.real() / b.real()) * b.real();
  T i = a.imag() - floor(a.imag() / b.imag()) * b.imag();
  return complex_t<T>{r, i};
}

template <typename T>
inline __host__ __device__ bool operator>(complex_t<T> a, complex_t<T> b) {
  return (a.real() > b.real()) || (a.real() == b.real() && a.imag() > b.imag());
}

template <typename T>
inline __host__ __device__ bool operator<(complex_t<T> a, complex_t<T> b) {
  return operator>(b, a);
}

template <typename T>
inline __host__ __device__ bool operator<=(complex_t<T> a, complex_t<T> b) {
  return !(a > b);
}

template <typename T>
inline __host__ __device__ bool operator>=(complex_t<T> a, complex_t<T> b) {
  return !(a < b);
}

} // namespace mlx::core::cu
