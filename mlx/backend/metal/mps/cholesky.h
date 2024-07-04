// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/mps/gemm.h"

#include <Metal/Metal.hpp>

namespace MTL::Private::Class {
_MTL_PRIVATE_DEF_CLS(MPSMatrixDecompositionCholesky);
} // namespace MTL::Private::Class

namespace MTL::Private::Selector {
_MTL_PRIVATE_DEF_SEL(
    initWithDevice_lower_order_,
    "initWithDevice:lower:order:");
_MTL_PRIVATE_DEF_SEL(
    encodeToCommandBuffer_sourceMatrix_resultMatrix_status,
    "encodeToCommandBuffer:sourceMatrix:resultMatrix:status:");
} // namespace MTL::Private::Selector

namespace MPS {

class MatrixDecompositionCholesky
    : public NS::Referencing<MatrixDecompositionCholesky, Kernel> {
 public:
  static class MatrixDecompositionCholesky* alloc();

  MatrixDecompositionCholesky*
  init(MTL::Device* device, bool lower, NS::UInteger order);

  void encodeToCommandBuffer(
      MTL::CommandBuffer* commandBuffer,
      Matrix* sourceMatrix,
      Matrix* resultMatrix,
      MTL::Buffer* status);
};

_MTL_INLINE MatrixDecompositionCholesky* MatrixDecompositionCholesky::alloc() {
  return NS::Object::alloc<MatrixDecompositionCholesky>(
      _MPS_PRIVATE_CLS(MPSMatrixDecompositionCholesky));
}

_MTL_INLINE MatrixDecompositionCholesky* MatrixDecompositionCholesky::init(
    MTL::Device* device,
    bool lower,
    NS::UInteger order) {
  return Object::sendMessage<MatrixDecompositionCholesky*>(
      this,
      _MPS_PRIVATE_SEL(initWithDevice_lower_order_),
      device,
      lower,
      order);
}

_MTL_INLINE void MatrixDecompositionCholesky::encodeToCommandBuffer(
    MTL::CommandBuffer* commandBuffer,
    Matrix* sourceMatrix,
    Matrix* resultMatrix,
    MTL::Buffer* status) {
  return Object::sendMessage<void>(
      this,
      _MPS_PRIVATE_SEL(encodeToCommandBuffer_sourceMatrix_resultMatrix_status),
      commandBuffer,
      sourceMatrix,
      resultMatrix,
      status);
}

} // namespace MPS
