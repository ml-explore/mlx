// Copyright © 2025 Apple Inc.

#pragma once

#include <cstring>
#include <list>
#include <unordered_map>
#include <utility>

namespace mlx::core {

template <
    typename K,
    typename V,
    template <typename...> typename M = std::unordered_map>
class LRUCache {
 public:
  using value_type = std::pair<K, V>;
  using list_type = std::list<value_type>;
  using iterator = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;
  using map_type = M<K, iterator>;

  explicit LRUCache(size_t capacity) : capacity_(capacity) {
    if (capacity == 0) {
      throw std::runtime_error("LRUCache requires capacity > 0.");
    }
  }

  size_t size() const {
    return map_.size();
  }
  size_t capacity() const {
    return capacity_;
  }
  bool empty() const {
    return vlist_.empty();
  }

  void resize(size_t new_capacity) {
    capacity_ = new_capacity;
    trim();
  }

  iterator begin() {
    return vlist_.begin();
  }
  const_iterator begin() const {
    return vlist_.begin();
  }
  iterator end() {
    return vlist_.end();
  }
  const_iterator end() const {
    return vlist_.end();
  }

  void clear() {
    map_.clear();
    vlist_.clear();
  }

  iterator find(const K& key) {
    auto it = map_.find(key);
    if (it == map_.end())
      return end();
    vlist_.splice(vlist_.begin(), vlist_, it->second);
    return it->second;
  }

  template <typename U>
  std::pair<iterator, bool> emplace(const K& key, U&& value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return {it->second, false};
    }

    vlist_.emplace_front(key, std::forward<U>(value));
    map_[key] = vlist_.begin();

    trim();

    return {vlist_.begin(), true};
  }

  iterator erase(iterator pos) {
    map_.erase(pos->first);
    return vlist_.erase(pos);
  }

  V& operator[](const K& key) {
    auto it = find(key);
    if (it == end()) {
      it = emplace(key, V{}).first;
    }
    return it->second;
  }

 private:
  void trim() {
    while (map_.size() > capacity_) {
      auto last = std::prev(vlist_.end());
      map_.erase(last->first);
      vlist_.pop_back();
    }
  }

  list_type vlist_;
  map_type map_;
  size_t capacity_;
};

// Turn a POD struct into a container key by doing bytes compare.
template <typename T>
struct BytesKey {
  T pod;
  static_assert(std::is_standard_layout_v<T>, "T is not POD");

  BytesKey(T pod) : pod(std::move(pod)) {}

  BytesKey(const BytesKey& other) {
    memcpy(&pod, &other.pod, sizeof(T));
  }

  BytesKey(BytesKey&& other) {
    memcpy(&pod, &other.pod, sizeof(T));
  }

  bool operator==(const BytesKey& other) const {
    auto* ptr1 = reinterpret_cast<const uint8_t*>(&pod);
    auto* ptr2 = reinterpret_cast<const uint8_t*>(&other.pod);
    return memcmp(ptr1, ptr2, sizeof(T)) == 0;
  }
};

// Compute hash according to the bytes value of T.
template <typename T>
struct BytesHash {
  static_assert(std::is_standard_layout_v<T>, "T is not POD");

  size_t operator()(const T& pod) const {
    auto* ptr = reinterpret_cast<const uint8_t*>(&pod);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < sizeof(T); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return value;
  }
};

template <typename K, typename V>
using BytesKeyHashMap = std::unordered_map<K, V, BytesHash<K>>;

template <typename K, typename V>
using LRUBytesKeyCache = LRUCache<BytesKey<K>, V, BytesKeyHashMap>;

} // namespace mlx::core
