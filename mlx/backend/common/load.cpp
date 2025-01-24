// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/backend/common/load.h"

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

namespace mlx::core {

void load(
    array& out,
    size_t offset,
    const std::shared_ptr<io::Reader>& reader,
    bool swap_endianness_) {
  reader->read(out.data<char>(), out.nbytes(), offset);

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
