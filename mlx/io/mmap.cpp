// Copyright © 2026 Apple Inc.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/io.h"
#include "mlx/utils.h"

namespace mlx::core {

array mmap_weights(
    const std::string& file,
    int64_t byte_offset,
    Shape shape,
    Dtype dtype) {
  size_t nelem = 1;
  for (auto s : shape) {
    if (s < 0) {
      throw std::invalid_argument("[mmap_weights] negative dimension");
    }
    nelem *= static_cast<size_t>(s);
  }
  size_t nbytes = nelem * size_of(dtype);
  if (nbytes == 0) {
    throw std::invalid_argument("[mmap_weights] empty tensor");
  }

  int fd = open(file.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::invalid_argument("[mmap_weights] cannot open: " + file);
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || byte_offset < 0 ||
      static_cast<size_t>(byte_offset) + nbytes >
          static_cast<size_t>(st.st_size)) {
    close(fd);
    throw std::invalid_argument("[mmap_weights] range out of bounds: " + file);
  }

  // Map from the enclosing page boundary; the tensor begins `delta` bytes in.
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t base_off = (static_cast<size_t>(byte_offset) / page) * page;
  const size_t delta = static_cast<size_t>(byte_offset) - base_off;
  const size_t map_len = ((delta + nbytes + page - 1) / page) * page;

  void* base = mmap(
      nullptr,
      map_len,
      PROT_READ,
      MAP_SHARED,
      fd,
      static_cast<off_t>(base_off));
  close(fd); // the mapping keeps its own reference
  if (base == MAP_FAILED) {
    throw std::runtime_error("[mmap_weights] mmap failed: " + file);
  }

  // Metal setBuffer offsets must be aligned to the element size (mlx's own
  // sliced arrays rely on the same). NOTE: stock safetensors gives NO
  // alignment guarantee — real checkpoints put tensors at odd offsets — so
  // callers typically need an aligned store.
  if (delta % size_of(dtype) != 0) {
    munmap(base, map_len);
    throw std::invalid_argument(
        "[mmap_weights] byte_offset must be aligned to the element size");
  }

  auto buf = allocator::make_buffer(base, map_len);
  if (buf.ptr() == nullptr) {
    munmap(base, map_len);
    throw std::runtime_error(
        "[mmap_weights] make_buffer failed (Metal unavailable or mapping "
        "rejected)");
  }

  array out(shape, dtype, nullptr, {});
  Strides strides(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= shape[i];
  }
  array::Flags flags{};
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = shape.size() <= 1;
  out.set_data(
      buf,
      nelem,
      std::move(strides),
      flags,
      static_cast<int64_t>(delta),
      // The buffer wraps an mmap'd file region: release the Metal wrapper
      // (never recycle into the allocator cache) and drop the mapping. Runs
      // on a Metal completion-handler thread — pure C++ only.
      [base, map_len](allocator::Buffer b) {
        allocator::release(b);
        munmap(base, map_len);
      });
  out.set_status(array::Status::available);
  return out;
}

} // namespace mlx::core
