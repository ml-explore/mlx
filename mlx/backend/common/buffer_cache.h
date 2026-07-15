// Copyright © 2025 Apple Inc.

#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>

namespace mlx::core {

template <typename T>
class BufferCache {
 public:
  // min_utilization: only hand out a cached buffer if
  //   requested_size / cached_size >= min_utilization
  // 1.0f = exact size only (fixed-shape training: 100% fill of reused blocks).
  // 0.5f = classic MLX (allow up to ~2× oversize).
  BufferCache(
      size_t page_size,
      std::function<size_t(T*)> get_size,
      std::function<void(T*)> free,
      float min_utilization = 0.5f)
      : page_size_(page_size),
        get_size_(std::move(get_size)),
        free_(std::move(free)),
        min_utilization_(std::clamp(min_utilization, 0.f, 1.f)) {}

  ~BufferCache() {
    clear();
  }

  BufferCache(const BufferCache&) = delete;
  BufferCache& operator=(const BufferCache&) = delete;

  T* reuse_from_cache(size_t size) {
    // Exact size first — fixed B×T training hits this every step.
    {
      auto range = buffer_pool_.equal_range(size);
      if (range.first != range.second) {
        return take(range.first);
      }
    }

    if (min_utilization_ >= 1.0f) {
      return nullptr;
    }

    // Closest larger buffer still meeting utilization + page tolerance.
    // Reject if utilization would drop below min (e.g. 50% → free that slab
    // later rather than waste half the allocation).
    auto it = buffer_pool_.lower_bound(size);
    const size_t max_by_util = static_cast<size_t>(
        static_cast<float>(size) / std::max(min_utilization_, 1e-6f));
    const size_t max_by_page = size + 2 * page_size_;
    const size_t cap = std::min(max_by_util, max_by_page);
    while (it != buffer_pool_.end() && it->first <= cap) {
      if (static_cast<float>(size) >=
          min_utilization_ * static_cast<float>(it->first)) {
        return take(it);
      }
      ++it;
    }
    return nullptr;
  }

  void recycle_to_cache(T* buf) {
    assert(buf);
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
        size_t size = get_size_(tail_->buf);
        total_bytes_freed += size;
        free_(tail_->buf);
        n_release++;

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

  T* take(typename std::multimap<size_t, BufferHolder*>::iterator it) {
    T* buf = it->second->buf;
    pool_size_ -= it->first;
    remove_from_list(it->second);
    buffer_pool_.erase(it);
    return buf;
  }

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
    if (to_remove->prev && to_remove->next) {
      to_remove->prev->next = to_remove->next;
      to_remove->next->prev = to_remove->prev;
    } else if (to_remove->prev && to_remove == tail_) {
      tail_ = to_remove->prev;
      tail_->next = nullptr;
    } else if (to_remove == head_ && to_remove->next) {
      head_ = to_remove->next;
      head_->prev = nullptr;
    } else if (to_remove == head_ && to_remove == tail_) {
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
  const float min_utilization_;
};

} // namespace mlx::core
