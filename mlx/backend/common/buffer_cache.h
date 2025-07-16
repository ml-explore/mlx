// Copyright © 2025 Apple Inc.

#pragma once

#include <cassert>
#include <functional>
#include <map>

namespace mlx::core {

template <typename T>
class BufferCache {
 public:
  BufferCache(
      size_t page_size,
      std::function<size_t(T*)> get_size,
      std::function<void(T*)> free)
      : page_size_(page_size),
        get_size_(std::move(get_size)),
        free_(std::move(free)) {}

  ~BufferCache() {
    clear();
  }

  BufferCache(const BufferCache&) = delete;
  BufferCache& operator=(const BufferCache&) = delete;

  T* reuse_from_cache(size_t size) {
    // Find the closest buffer in pool.
    auto it = buffer_pool_.lower_bound(size);
    if (it == buffer_pool_.end() ||
        it->first >= std::min(2 * size, size + 2 * page_size_)) {
      return nullptr;
    }

    // Collect from the cache.
    T* buf = it->second->buf;
    pool_size_ -= it->first;

    // Remove from record.
    remove_from_list(it->second);
    buffer_pool_.erase(it);
    return buf;
  }

  void recycle_to_cache(T* buf) {
    assert(buf);
    // Add to cache.
    BufferHolder* bh = new BufferHolder(buf);
    add_at_head(bh);
    size_t size = get_size_(buf);
    pool_size_ += size;
    buffer_pool_.emplace(size, bh);
  }

  int release_cached_buffers(size_t min_bytes_to_free) {
    if (min_bytes_to_free >= 0.9 * pool_size_) {
      return clear();
    } else {
      int n_release = 0;
      size_t total_bytes_freed = 0;

      while (tail_ && (total_bytes_freed < min_bytes_to_free)) {
        // Release buffer.
        size_t size = get_size_(tail_->buf);
        total_bytes_freed += size;
        free_(tail_->buf);
        n_release++;

        // Remove from record.
        auto its = buffer_pool_.equal_range(size);
        auto it = std::find_if(its.first, its.second, [this](const auto& el) {
          return el.second == tail_;
        });
        assert(it != buffer_pool_.end());
        buffer_pool_.erase(it);
        remove_from_list(tail_);
      }

      pool_size_ -= total_bytes_freed;
      return n_release;
    }
  }

  int clear() {
    int n_release = 0;
    for (auto& [size, holder] : buffer_pool_) {
      free_(holder->buf);
      n_release++;
      delete holder;
    }
    buffer_pool_.clear();
    pool_size_ = 0;
    head_ = nullptr;
    tail_ = nullptr;
    return n_release;
  }

  size_t cache_size() const {
    return pool_size_;
  }

  size_t page_size() const {
    return page_size_;
  }

 private:
  struct BufferHolder {
   public:
    explicit BufferHolder(T* buf_) : buf(buf_) {}

    BufferHolder* prev{nullptr};
    BufferHolder* next{nullptr};
    T* buf;
  };

  void add_at_head(BufferHolder* to_add) {
    if (!head_) {
      head_ = to_add;
      tail_ = to_add;
    } else {
      head_->prev = to_add;
      to_add->next = head_;
      head_ = to_add;
    }
  }

  void remove_from_list(BufferHolder* to_remove) {
    if (to_remove->prev && to_remove->next) { // if middle
      to_remove->prev->next = to_remove->next;
      to_remove->next->prev = to_remove->prev;
    } else if (to_remove->prev && to_remove == tail_) { // if tail
      tail_ = to_remove->prev;
      tail_->next = nullptr;
    } else if (to_remove == head_ && to_remove->next) { // if head
      head_ = to_remove->next;
      head_->prev = nullptr;
    } else if (to_remove == head_ && to_remove == tail_) { // if only element
      head_ = nullptr;
      tail_ = nullptr;
    }

    delete to_remove;
  }

  std::multimap<size_t, BufferHolder*> buffer_pool_;
  BufferHolder* head_{nullptr};
  BufferHolder* tail_{nullptr};
  size_t pool_size_{0};

  const size_t page_size_;
  std::function<size_t(T*)> get_size_;
  std::function<void(T*)> free_;
};

} // namespace mlx::core
