// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace my_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
mx::array axpby(
    const mx::array& x, // Input array x
    const mx::array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    mx::StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Axpby : public mx::Primitive {
 public:
  explicit Axpby(mx::Stream stream, float alpha, float beta)
      : mx::Primitive(stream), alpha_(alpha), beta_(beta) {};

  /**
   * A primitive must know how to evaluate itself on the CPU/GPU
   * for the given inputs and populate the output array.
   *
   * To avoid unnecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The Jacobian-vector product. */
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  /** Print the primitive. */
  void print(std::ostream& os) override {
    os << "Axpby";
  }

  /** Equivalence check **/
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float alpha_;
  float beta_;

  /** Fall back implementation for evaluation on CPU */
  void eval(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs);
};

} // namespace my_ext
