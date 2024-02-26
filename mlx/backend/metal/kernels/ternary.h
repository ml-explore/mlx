// Copyright © 2023-2024 Apple Inc.

#pragma once

struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};
