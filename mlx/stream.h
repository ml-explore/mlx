// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/device.h"

namespace mlx::core {

struct Stream {
  int index;
  Device device;
  explicit Stream(int index, Device device) : index(index), device(device) {}
};

/** Get the default stream for the given device. */
Stream default_stream(Device d);

/** Make the stream the default for its device. */
void set_default_stream(Stream s);

/** Make a new stream on the given device. */
Stream new_stream(Device d);

/** Get the stream with the given index. */
Stream get_stream(int index);

inline bool operator==(const Stream& lhs, const Stream& rhs) {
  return lhs.index == rhs.index;
}

inline bool operator!=(const Stream& lhs, const Stream& rhs) {
  return !(lhs == rhs);
}

/* Synchronize with the default stream. */
void synchronize();

/* Synchronize with the provided stream. */
void synchronize(Stream);

} // namespace mlx::core
