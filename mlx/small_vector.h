// Copyright © 2024 Apple Inc.

#pragma once

#include <array>
#include <sstream>
#include <variant>
#include <vector>

namespace mlx::core {

template <typename T, size_t stack_size = 8>
class small_vector {
 public:
  using array_type = std::array<T, stack_size>;
  using vector_type = std::vector<T>;

  small_vector() : head_(nullptr), size_(0), storage_(nullptr) {}

  small_vector(vector_type vec)
      : head_(vec.empty() ? nullptr : &vec[0]),
        size_(vec.size()),
        storage_(std::move(vec)) {}

  template <
      typename... Ts,
      typename = std::enable_if_t<(std::is_same_v<Ts, T> && ...)>>
  small_vector(Ts... args)
      : size_(sizeof...(args)), storage_(array_type{std::move(args)...}) {
    static_assert(
        sizeof...(args) > 0, "Empty args should go to default constructor.");
    static_assert(
        sizeof...(args) < stack_size,
        "Lots of args not supported in aggregate constructor.");
    head_ = &std::get<array_type>(storage_)[0];
  }

  small_vector(small_vector&& moved)
      : head_(nullptr),
        size_(moved.size_),
        storage_(std::move(moved.storage_)) {
    if (size_ > 0) {
      if (std::holds_alternative<array_type>(storage_))
        head_ = &std::get<array_type>(storage_)[0];
      else if (std::holds_alternative<vector_type>(storage_))
        head_ = &std::get<vector_type>(storage_)[0];
    }
    moved.head_ = nullptr;
    moved.size_ = 0;
  }

  small_vector(const small_vector& copy) = delete;
  small_vector& operator=(const small_vector& assign) = delete;

  void clear() {
    head_ = nullptr;
    size_ = 0;
    storage_ = nullptr;
  }

  bool empty() const {
    return size_ == 0;
  }

  size_t size() const {
    return size_;
  }

  T* begin() {
    return head_;
  }
  const T* begin() const {
    return head_;
  }
  T* end() {
    return head_ + size_;
  }
  const T* end() const {
    return head_ + size_;
  }

  T& back() {
    return head_[size_ - 1];
  }
  const T& back() const {
    return head_[size_ - 1];
  }

  T& operator[](size_t index) {
    return head_[index];
  }
  const T& operator[](size_t index) const {
    return head_[index];
  }

  T& at(size_t index) {
    if (index >= size_) {
      std::ostringstream msg;
      msg << "index " << index << " is larger than vector size " << size_;
      throw std::out_of_range(msg.str());
    }
    return head_[index];
  }
  const T& at(size_t index) const {
    return const_cast<small_vector*>(this)->at(index);
  }

  vector_type& as_vector() {
    convert_to_vector();
    return std::get<vector_type>(storage_);
  }

 private:
  friend class array;

  void convert_to_vector() {
    if (std::holds_alternative<vector_type>(storage_)) {
      return;
    }
    if (std::holds_alternative<array_type>(storage_) && size_ > 0) {
      // Copy the values to vector. While this looks expensive, it is actually
      // the same cost when you initialize vector with initializer_list.
      // We will be able to remove most of such calls after migrating all code
      // to use small_vector.
      storage_ = vector_type(begin(), end());
      head_ = &std::get<vector_type>(storage_)[0];
      return;
    }
    storage_ = vector_type();
  }

  T* head_;
  size_t size_;
  std::variant<nullptr_t, array_type, vector_type> storage_;
};

} // namespace mlx::core
