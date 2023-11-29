#pragma once

#include <chrono>

namespace timer {

using namespace std::chrono;

template <typename R, typename P>
inline double seconds(duration<R, P> x) {
  return duration_cast<nanoseconds>(x).count() / 1e9;
}

inline auto time() {
  return high_resolution_clock::now();
}

} // namespace timer
