// Copyright © 2025 Apple Inc.
// Copyright © 2018 the V8 project authors.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

namespace mlx::core {

#if defined(__has_builtin)
#define MLX_HAS_BUILTIN(x) __has_builtin(x)
#else
#define MLX_HAS_BUILTIN(x) 0
#endif

#if defined(__has_attribute)
#define MLX_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define MLX_HAS_ATTRIBUTE(x) 0
#endif

#if MLX_HAS_BUILTIN(__builtin_expect)
#define MLX_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#define MLX_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
#define MLX_LIKELY(condition) (condition)
#define MLX_UNLIKELY(condition) (condition)
#endif

#if MLX_HAS_ATTRIBUTE(noinline)
#define MLX_NOINLINE __attribute__((noinline))
#else
#define MLX_NOINLINE
#endif

template <typename T, typename = void>
struct is_iterator : std::false_type {};

template <typename T>
struct is_iterator<
    T,
    std::void_t<
        typename std::iterator_traits<T>::difference_type,
        typename std::iterator_traits<T>::iterator_category,
        typename std::iterator_traits<T>::pointer,
        typename std::iterator_traits<T>::reference,
        typename std::iterator_traits<T>::value_type>> : std::true_type {};

template <typename T>
constexpr bool is_iterator_v = is_iterator<T>::value;

// Minimal SmallVector implementation. Uses inline storage first, switches to
// dynamic storage when it overflows.
//
// Notes:
// * The default inline storage size is MAX_NDIM, as it is mainly used for
//   shapes and strides, users should choose a better size for other cases.
// * The data() returns real address even for empty vector.
// * The pointer returned by data() will change after moving the vector as it
//   points to the inline storage.
// * For trivial elements the storage will not be default constructed,
//   i.e. SmallVector<int>(10) will not be filled with 0 by default.
template <typename T, size_t kSize = 10, typename Allocator = std::allocator<T>>
class SmallVector {
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  SmallVector() = default;

  explicit SmallVector(const Allocator& allocator) : allocator_(allocator) {}

  explicit SmallVector(size_t size, const Allocator& allocator = Allocator())
      : allocator_(allocator) {
    resize(size);
  }

  SmallVector(
      size_t size,
      const T& initial_value,
      const Allocator& allocator = Allocator())
      : allocator_(allocator) {
    resize(size, initial_value);
  }

  SmallVector(
      std::initializer_list<T> init,
      const Allocator& allocator = Allocator())
      : allocator_(allocator) {
    if (init.size() > capacity()) {
      grow(init.size());
    }
    assert(capacity() >= init.size()); // sanity check
    std::uninitialized_move(init.begin(), init.end(), begin_);
    end_ = begin_ + init.size();
  }

  template <typename Iter, typename = std::enable_if_t<is_iterator_v<Iter>>>
  SmallVector(Iter begin, Iter end, const Allocator& allocator = Allocator())
      : allocator_(allocator) {
    size_t size = std::distance(begin, end);
    if (size > capacity()) {
      grow(size);
    }
    assert(capacity() >= size); // sanity check
    std::uninitialized_copy(begin, end, begin_);
    end_ = begin_ + size;
  }

  SmallVector(const SmallVector& other) : allocator_(other.allocator_) {
    *this = other;
  }
  SmallVector(const SmallVector& other, const Allocator& allocator)
      : allocator_(allocator) {
    *this = other;
  }
  SmallVector(SmallVector&& other) : allocator_(std::move(other.allocator_)) {
    *this = std::move(other);
  }
  SmallVector(SmallVector&& other, const Allocator& allocator)
      : allocator_(allocator) {
    *this = std::move(other);
  }

  ~SmallVector() {
    free_storage();
  }

  SmallVector& operator=(const SmallVector& other) {
    if (this == &other) {
      return *this;
    }
    size_t other_size = other.size();
    if (capacity() < other_size) {
      // Create large-enough heap-allocated storage.
      free_storage();
      begin_ = allocator_.allocate(other_size);
      end_of_storage_ = begin_ + other_size;
      std::uninitialized_copy(other.begin_, other.end_, begin_);
    } else if constexpr (kHasTrivialElement) {
      std::copy(other.begin_, other.end_, begin_);
    } else {
      ptrdiff_t to_copy =
          std::min(static_cast<ptrdiff_t>(other_size), end_ - begin_);
      std::copy(other.begin_, other.begin_ + to_copy, begin_);
      if (other.begin_ + to_copy < other.end_) {
        std::uninitialized_copy(
            other.begin_ + to_copy, other.end_, begin_ + to_copy);
      } else {
        std::destroy_n(begin_ + to_copy, size() - to_copy);
      }
    }
    end_ = begin_ + other_size;
    return *this;
  }

  SmallVector& operator=(SmallVector&& other) {
    if (this == &other) {
      return *this;
    }
    if (other.is_big()) {
      free_storage();
      begin_ = other.begin_;
      end_ = other.end_;
      end_of_storage_ = other.end_of_storage_;
    } else {
      assert(capacity() >= other.size()); // sanity check
      size_t other_size = other.size();
      if constexpr (kHasTrivialElement) {
        std::move(other.begin_, other.end_, begin_);
      } else {
        ptrdiff_t to_move =
            std::min(static_cast<ptrdiff_t>(other_size), end_ - begin_);
        std::move(other.begin_, other.begin_ + to_move, begin_);
        if (other.begin_ + to_move < other.end_) {
          std::uninitialized_move(
              other.begin_ + to_move, other.end_, begin_ + to_move);
        } else {
          std::destroy_n(begin_ + to_move, size() - to_move);
        }
      }
      end_ = begin_ + other_size;
    }
    other.reset_to_inline_storage();
    return *this;
  }

  bool operator==(const SmallVector& other) const {
    if (size() != other.size()) {
      return false;
    }
    return std::equal(begin_, end_, other.begin_);
  }

  bool operator!=(const SmallVector& other) const {
    return !(*this == other);
  }

  T* data() {
    return begin_;
  }
  const T* data() const {
    return begin_;
  }

  iterator begin() {
    return begin_;
  }
  const_iterator begin() const {
    return begin_;
  }

  iterator end() {
    return end_;
  }
  const_iterator end() const {
    return end_;
  }

  const_iterator cbegin() const {
    return begin_;
  }

  const_iterator cend() const {
    return end_;
  }

  auto rbegin() {
    return std::make_reverse_iterator(end_);
  }
  auto rbegin() const {
    return std::make_reverse_iterator(end_);
  }

  auto rend() {
    return std::make_reverse_iterator(begin_);
  }
  auto rend() const {
    return std::make_reverse_iterator(begin_);
  }

  size_t size() const {
    return end_ - begin_;
  }
  bool empty() const {
    return end_ == begin_;
  }
  size_t capacity() const {
    return end_of_storage_ - begin_;
  }

  T& front() {
    assert(size() != 0);
    return begin_[0];
  }
  const T& front() const {
    assert(size() != 0);
    return begin_[0];
  }

  T& back() {
    assert(size() != 0);
    return end_[-1];
  }
  const T& back() const {
    assert(size() != 0);
    return end_[-1];
  }

  T& at(size_t index) {
    if (index >= size()) {
      throw std::out_of_range("SmallVector out of range.");
    }
    return begin_[index];
  }
  const T& at(size_t index) const {
    return const_cast<SmallVector*>(this)->at(index);
  }

  T& operator[](size_t index) {
    assert(size() > index);
    return begin_[index];
  }
  const T& operator[](size_t index) const {
    return const_cast<SmallVector*>(this)->operator[](index);
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    if (MLX_UNLIKELY(end_ == end_of_storage_)) {
      grow();
    }
    void* storage = end_;
    end_ += 1;
    new (storage) T(std::forward<Args>(args)...);
  }

  void push_back(T x) {
    emplace_back(std::move(x));
  }

  void pop_back(size_t count = 1) {
    assert(size() >= count);
    end_ -= count;
    std::destroy_n(end_, count);
  }

  iterator insert(iterator pos, T value) {
    return insert(pos, static_cast<size_t>(1), std::move(value));
  }

  iterator insert(iterator pos, size_t count, T value) {
    assert(pos <= end_);
    size_t offset = pos - begin_;
    size_t old_size = size();
    resize(old_size + count);
    pos = begin_ + offset;
    iterator old_end = begin_ + old_size;
    assert(old_end <= end_);
    std::move_backward(pos, old_end, end_);
    if constexpr (kHasTrivialElement) {
      std::fill_n(pos, count, value);
    } else {
      std::fill_n(pos + 1, count - 1, value);
      *pos = std::move(value);
    }
    return pos;
  }

  template <typename Iter, typename = std::enable_if_t<is_iterator_v<Iter>>>
  iterator insert(iterator pos, Iter begin, Iter end) {
    if constexpr (std::is_same_v<std::decay_t<Iter>, iterator>) {
      // The implementation can not take overlapping range.
      assert(!(begin >= pos && begin < pos + std::distance(begin, end)));
      assert(!(end > pos && end <= pos + std::distance(begin, end)));
    }

    assert(pos <= end_);
    size_t offset = pos - begin_;
    size_t count = std::distance(begin, end);
    size_t old_size = size();
    resize(old_size + count);
    pos = begin_ + offset;
    iterator old_end = begin_ + old_size;
    assert(old_end <= end_);
    std::move_backward(pos, old_end, end_);
    std::copy(begin, end, pos);
    return pos;
  }

  iterator insert(iterator pos, std::initializer_list<const T> values) {
    return insert(pos, values.begin(), values.end());
  }

  iterator erase(iterator erase_start, iterator erase_end) {
    assert(erase_start >= begin_);
    assert(erase_start <= erase_end);
    assert(erase_end <= end_);
    iterator new_end = std::move(erase_end, end_, erase_start);
    std::destroy_n(new_end, std::distance(new_end, end_));
    end_ = new_end;
    return erase_start;
  }

  iterator erase(iterator pos) {
    return erase(pos, pos + 1);
  }

  void resize(size_t new_size) {
    if (new_size > capacity()) {
      grow(new_size);
    }
    T* new_end = begin_ + new_size;
    if constexpr (!kHasTrivialElement) {
      if (new_end > end_) {
        std::uninitialized_default_construct(end_, new_end);
      } else {
        std::destroy_n(new_end, end_ - new_end);
      }
    }
    end_ = new_end;
  }

  void resize(size_t new_size, const T& initial_value) {
    if (new_size > capacity()) {
      grow(new_size);
    }
    T* new_end = begin_ + new_size;
    if (new_end > end_) {
      std::uninitialized_fill(end_, new_end, initial_value);
    } else {
      std::destroy_n(new_end, end_ - new_end);
    }
    end_ = new_end;
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > capacity()) {
      grow(new_capacity);
    }
  }

  // Clear without reverting back to inline storage.
  void clear() {
    std::destroy_n(begin_, end_ - begin_);
    end_ = begin_;
  }

 private:
  // Grows the backing store by a factor of two, and at least to {min_capacity}.
  // TODO: Move to private after removing external code using this method.
  MLX_NOINLINE void grow(size_t min_capacity = 0) {
    size_t new_capacity = std::max(min_capacity, 2 * capacity());
    // Round up to power of 2.
    new_capacity--;
    new_capacity |= new_capacity >> 1;
    new_capacity |= new_capacity >> 2;
    new_capacity |= new_capacity >> 4;
    new_capacity |= new_capacity >> 8;
    new_capacity |= new_capacity >> 16;
    if constexpr (sizeof(size_t) == sizeof(uint64_t)) {
      new_capacity |= new_capacity >> 32;
    }
    new_capacity++;

    T* new_storage = allocator_.allocate(new_capacity);
    if (new_storage == nullptr) {
      throw std::bad_alloc();
    }

    size_t in_use = end_ - begin_;
    std::uninitialized_move(begin_, end_, new_storage);
    free_storage();
    begin_ = new_storage;
    end_ = new_storage + in_use;
    end_of_storage_ = new_storage + new_capacity;
  }

  MLX_NOINLINE void free_storage() {
    std::destroy_n(begin_, end_ - begin_);
    if (is_big()) {
      allocator_.deallocate(begin_, end_of_storage_ - begin_);
    }
  }

  // Clear and go back to inline storage. Dynamic storage is *not* freed. For
  // internal use only.
  void reset_to_inline_storage() {
    if constexpr (!kHasTrivialElement) {
      if (!is_big())
        std::destroy_n(begin_, end_ - begin_);
    }
    begin_ = inline_storage_begin();
    end_ = begin_;
    end_of_storage_ = begin_ + kSize;
  }

  bool is_big() const {
    return begin_ != inline_storage_begin();
  }

  T* inline_storage_begin() {
    return reinterpret_cast<T*>(inline_storage_);
  }
  const T* inline_storage_begin() const {
    return reinterpret_cast<const T*>(inline_storage_);
  }

  Allocator allocator_;

  // Invariants:
  // 1. The elements in the range between `begin_` (included) and `end_` (not
  //    included) will be initialized at all times.
  // 2. All other elements outside the range, both in the inline storage and in
  //    the dynamic storage (if it exists), will be uninitialized at all times.

  T* begin_ = inline_storage_begin();
  T* end_ = begin_;
  T* end_of_storage_ = begin_ + kSize;

  alignas(T) char inline_storage_[sizeof(T) * kSize];

  static constexpr bool kHasTrivialElement =
      std::is_trivially_copyable<T>::value &&
      std::is_trivially_destructible<T>::value;
};

template <typename>
struct is_vector : std::false_type {};

template <typename T, size_t Size, typename Allocator>
struct is_vector<SmallVector<T, Size, Allocator>> : std::true_type {};

template <typename T, typename Allocator>
struct is_vector<std::vector<T, Allocator>> : std::true_type {};

template <typename Vec>
inline constexpr bool is_vector_v = is_vector<Vec>::value;

#undef MLX_HAS_BUILTIN
#undef MLX_HAS_ATTRIBUTE
#undef MLX_LIKELY
#undef MLX_UNLIKELY
#undef MLX_NOINLINE

} // namespace mlx::core
