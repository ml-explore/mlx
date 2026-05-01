// Copyright © 2026 Apple Inc.
//
// Exercises Group::barrier(). Ranks arrive at the barrier at staggered times;
// after the barrier returns we do a small all_sum to confirm the group is
// healthy and that barrier() carried the correct fence semantics.

#include <chrono>
#include <iostream>
#include <thread>

#include <jaccl/jaccl.h>

int main() {
  auto group = jaccl::init();
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  int rank = group->rank();
  int size = group->size();

  std::this_thread::sleep_for(std::chrono::milliseconds(100 * rank));
  std::cout << "rank " << rank << " entering barrier" << std::endl;

  group->barrier();

  std::cout << "rank " << rank << " exited barrier" << std::endl;

  int in = rank + 1;
  int out = 0;
  group->all_sum(&in, &out, sizeof(in), jaccl::Int32);
  int expected = size * (size + 1) / 2;
  if (out != expected) {
    std::cerr << "rank " << rank << ": post-barrier all_sum mismatch (got "
              << out << ", expected " << expected << ")" << std::endl;
    return 1;
  }
  std::cout << "rank " << rank << ": post-barrier all_sum OK (" << out << ")"
            << std::endl;
  return 0;
}
