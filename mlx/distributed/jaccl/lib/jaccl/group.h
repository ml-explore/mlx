// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace jaccl {

/**
 * Abstract base class for a JACCL communication group.
 *
 * Groups are created and held through shared_ptr. A group made by `split`
 * bootstraps over its parent's side channel and so keeps the parent alive,
 * which is why `shared_from_this` has to be available here.
 */
class Group : public std::enable_shared_from_this<Group> {
 public:
  virtual ~Group() {}

  virtual int rank() = 0;
  virtual int size() = 0;

  /**
   * Build a new group from the members of this one that pass the same color,
   * ordered by key and then by rank in the parent, which is the rule MPI uses.
   *
   * Collective over this group: every member has to call it, including a
   * member that ends up in no child, because the colors are exchanged over
   * this group. Pass a negative color to take part without joining a child.
   */
  virtual std::shared_ptr<Group> split(int color, int key) {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

  virtual void
  all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  /**
   * Reduce scatter with a sum reduction.
   *
   * The input holds size() contiguous chunks of n_bytes each (total
   * size() * n_bytes bytes). After the call, output (n_bytes bytes) on rank r
   * contains the elementwise sum over all ranks of the r-th input chunk.
   */
  virtual void
  sum_scatter(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
  virtual void barrier() = 0;
};

/**
 * Type IDs for dispatch in the standalone JACCL library.
 *
 * Users pass one of these to all_sum/all_max/all_min so JACCL knows how to
 * interpret the data for typed reduction operations.
 */
enum Dtype {
  Bool = 0,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  BFloat16,
  Float32,
  Float64,
  Complex64,
};

} // namespace jaccl
