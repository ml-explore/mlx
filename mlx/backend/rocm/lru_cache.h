// Copyright © 2025 Apple Inc.

#pragma once

#include <cstdlib>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx::core::rocm {

// LRU cache with byte-based keys
template <typename Key, typename Value>
class LRUBytesKeyCache {
 public:
  LRUBytesKeyCache(const char* env_var, size_t default_capacity)
      : capacity_(default_capacity) {
    if (const char* env = std::getenv(env_var)) {
      capacity_ = std::stoul(env);
    }
  }

  std::optional<Value> get(const Key& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return std::nullopt;
    }
    // Move to front (most recently used)
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
  }

  void put(const Key& key, const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      // Update existing entry and move to front
      it->second->second = value;
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      return;
    }

    // Evict if at capacity
    while (cache_list_.size() >= capacity_) {
      auto last = cache_list_.back();
      cache_map_.erase(last.first);
      cache_list_.pop_back();
    }

    // Insert new entry at front
    cache_list_.emplace_front(key, value);
    cache_map_[key] = cache_list_.begin();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_list_.clear();
    cache_map_.clear();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_list_.size();
  }

 private:
  size_t capacity_;
  std::list<std::pair<Key, Value>> cache_list_;
  std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator>
      cache_map_;
  mutable std::mutex mutex_;
};

// Simple LRU cache with size_t keys
template <typename Value>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  std::optional<Value> get(size_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return std::nullopt;
    }
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
  }

  void put(size_t key, const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      it->second->second = value;
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      return;
    }

    while (cache_list_.size() >= capacity_) {
      auto last = cache_list_.back();
      cache_map_.erase(last.first);
      cache_list_.pop_back();
    }

    cache_list_.emplace_front(key, value);
    cache_map_[key] = cache_list_.begin();
  }

 private:
  size_t capacity_;
  std::list<std::pair<size_t, Value>> cache_list_;
  std::unordered_map<size_t, typename std::list<std::pair<size_t, Value>>::iterator>
      cache_map_;
  mutable std::mutex mutex_;
};

} // namespace mlx::core::rocm
