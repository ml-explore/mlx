// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/io/load.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

} // namespace

void Load::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  reader_->seek(offset_, std::ios_base::beg);
  reader_->read(out.data<char>(), out.nbytes());

  if (swap_endianness_) {
    switch (out.itemsize()) {
      case 2:
        swap_endianness<2>(out.data<uint8_t>(), out.data_size());
        break;
      case 4:
        swap_endianness<4>(out.data<uint8_t>(), out.data_size());
        break;
      case 8:
        swap_endianness<8>(out.data<uint8_t>(), out.data_size());
        break;
    }
  }
}

} // namespace mlx::core
