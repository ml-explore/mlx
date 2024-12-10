// Copyright © 2023 Apple Inc.

#include <cassert>
#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

void array_basics() {
  // Make a scalar array:
  mx::array x(1.0);

  // Get the value out of it:
  auto s = x.item<float>();
  assert(s == 1.0);

  // Scalars have a size of 1:
  size_t size = x.size();
  assert(size == 1);

  // Scalars have 0 dimensions:
  int ndim = x.ndim();
  assert(ndim == 0);

  // The shape should be an empty vector:
  auto shape = x.shape();
  assert(shape.empty());

  // The datatype should be float32:
  auto dtype = x.dtype();
  assert(dtype == mx::float32);

  // Specify the dtype when constructing the array:
  x = mx::array(1, mx::int32);
  assert(x.dtype() == mx::int32);
  x.item<int>(); // OK
  // x.item<float>();  // Undefined!

  // Make a multidimensional array:
  x = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  // mlx is row-major by default so the first row of this array
  // is [1.0, 2.0] and the second row is [3.0, 4.0]

  // Make an array of shape {2, 2} filled with ones:
  auto y = mx::ones({2, 2});

  // Pointwise add x and y:
  auto z = mx::add(x, y);

  // Same thing:
  z = x + y;

  // mlx is lazy by default. At this point `z` only
  // has a shape and a type but no actual data:
  assert(z.dtype() == mx::float32);
  assert(z.shape(0) == 2);
  assert(z.shape(1) == 2);

  // To actually run the computation you must evaluate `z`.
  // Under the hood, mlx records operations in a graph.
  // The variable `z` is a node in the graph which points to its operation
  // and inputs. When `eval` is called on an array (or arrays), the array and
  // all of its dependencies are recursively evaluated to produce the result.
  // Once an array is evaluated, it has data and is detached from its inputs.
  mx::eval(z);

  // Of course the array can still be an input to other operations. You can
  // even call eval on the array again, this will just be a no-op:
  mx::eval(z); // no-op

  // Some functions or methods on arrays implicitly evaluate them. For example
  // accessing a value in an array or printing the array implicitly evaluate it:
  z = mx::ones({1});
  z.item<float>(); // implicit evaluation

  z = mx::ones({2, 2});
  std::cout << z << std::endl; // implicit evaluation
}

void automatic_differentiation() {
  auto fn = [](mx::array x) { return mx::square(x); };

  // Computing the derivative function of a function
  auto grad_fn = mx::grad(fn);
  // Call grad_fn on the input to get the derivative
  auto x = mx::array(1.5);
  auto dfdx = grad_fn(x);
  // dfdx is 2 * x

  // Get the second derivative by composing grad with grad
  auto d2fdx2 = mx::grad(mx::grad(fn))(x);
  // d2fdx2 is 2
}

int main() {
  array_basics();
  automatic_differentiation();
}
