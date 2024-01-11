// Copyright © 2023 Apple Inc.
#include <iostream> // TODO
#include "mlx/transforms.h"

namespace mlx::core {

// class CompilerCache {
//   std::unordered_map
// }

template <typename T, typename... U>
size_t getAddress(std::function<T(U...)> f) {
  typedef T(fnType)(U...);
  fnType** fnPointer = f.template target<fnType*>();
  return (size_t)*fnPointer;
}

int g(int, int) {
  return 2;
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun) {
  // Not doing too much at the moment
  //  std::cout << getAddress(fun) << std::endl;
  return [&fun](const std::vector<array>& inputs) {
    std::cout << getAddress(fun) << std::endl;
    //    getAddress(std::function<int(int, int)>(g));
    //
    //    std::cout << getAddress(fun) << std::endl;
    // Step 1 check the cache for the function.
    // If it's in the cache check the shapes and types
    // If they match then run the cached function,
    //
    // What exactly is the cached function?
    // The return has to be the outputs of fun(inputs) which point to the
    // correct inputs So we need to store a tape of primitives -> inputs (shape,
    // dtype), outputs (shape, dtype) We need a level of indirection id to input
    // to store the inputs so we can
    //  T
    // Because eval will just want some pointers to arrays
    // So you go through and set the
    return fun(inputs);
  };
}

} // namespace mlx::core
