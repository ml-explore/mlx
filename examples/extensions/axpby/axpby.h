// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors elementwise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
array axpby(
    const array& x, // Input array x
    const array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Axpby : public Primitive {
 public:
  explicit Axpby(Stream stream, float alpha, float beta)
      : Primitive(stream), alpha_(alpha), beta_(beta){};

  /**
   * A primitive must know how to evaluate itself on the CPU/GPU
   * for the given inputs and populate the output array.
   *
   * To avoid unecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  /** The Jacobian-vector product. */
  array jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const array& cotan,
      const std::vector<int>& argnums) override;

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  /** Print the primitive. */
  void print(std::ostream& os) override {
    os << "Axpby";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

 private:
  float alpha_;
  float beta_;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, array& out);
};

} // namespace mlx::core