// Copyright © 2023 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>

#define _MPS_PRIVATE_CLS(symbol) (MTL::Private::Class::s_k##symbol)
#define _MPS_PRIVATE_SEL(accessor) (MTL::Private::Selector::s_k##accessor)

namespace MTL::Private::Class {
_MTL_PRIVATE_DEF_CLS(MPSMatrixDescriptor);
_MTL_PRIVATE_DEF_CLS(MPSMatrix);
_MTL_PRIVATE_DEF_CLS(MPSVectorDescriptor);
_MTL_PRIVATE_DEF_CLS(MPSVector);
_MTL_PRIVATE_DEF_CLS(MPSKernel);
_MTL_PRIVATE_DEF_CLS(MPSMatrixMultiplication);
_MTL_PRIVATE_DEF_CLS(MPSMatrixVectorMultiplication);
} // namespace MTL::Private::Class

namespace MTL::Private::Selector {
_MTL_PRIVATE_DEF_SEL(
    matrixDescriptorWithRows_columns_rowBytes_dataType,
    "matrixDescriptorWithRows:columns:rowBytes:dataType:");
_MTL_PRIVATE_DEF_SEL(
    matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType,
    "matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:");
_MTL_PRIVATE_DEF_SEL(rows, "rows");
_MTL_PRIVATE_DEF_SEL(initWithBuffer_descriptor, "initWithBuffer:descriptor:");
_MTL_PRIVATE_DEF_SEL(
    initWithDevice_,
    "initWithDevice:transposeLeft:transposeRight:"
    "resultRows:resultColumns:interiorColumns:alpha:beta:");
_MTL_PRIVATE_DEF_SEL(
    encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix,
    "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:");
_MTL_PRIVATE_DEF_SEL(setLeftMatrixOrigin_, "setLeftMatrixOrigin:");
_MTL_PRIVATE_DEF_SEL(setRightMatrixOrigin_, "setRightMatrixOrigin:");
_MTL_PRIVATE_DEF_SEL(setResultMatrixOrigin_, "setResultMatrixOrigin:");
_MTL_PRIVATE_DEF_SEL(setBatchStart_, "setBatchStart:");
_MTL_PRIVATE_DEF_SEL(setBatchSize_, "setBatchSize:");
_MTL_PRIVATE_DEF_SEL(
    vectorDescriptorWithLength_dataType,
    "vectorDescriptorWithLength:dataType:");
_MTL_PRIVATE_DEF_SEL(
    vectorDescriptorWithLength_vectors_vectorBytes_dataType,
    "vectorDescriptorWithLength:vectors:vectorBytes:dataType:");
_MTL_PRIVATE_DEF_SEL(
    initWithDevice_transpose_rows_columns_alpha_beta,
    "initWithDevice:transpose:rows:columns:alpha:beta:");
_MTL_PRIVATE_DEF_SEL(
    encodeToCommandBuffer_inputMatrix_inputVector_resultVector,
    "encodeToCommandBuffer:inputMatrix:inputVector:resultVector:");
} // namespace MTL::Private::Selector

namespace MPS {

typedef enum DataType : uint32_t {
  DataTypeFloatBit = 0x10000000,
  DataTypeAlternateEncodingBit = 0x80000000,
  DataTypeFloat16 = DataTypeFloatBit | 16,
  DataTypeFloat32 = DataTypeFloatBit | 32,
  DataTypeBFloat16 = DataTypeAlternateEncodingBit | DataTypeFloat16
} DataType;

class MatrixDescriptor : public NS::Copying<MatrixDescriptor> {
 public:
  static class MatrixDescriptor* matrixDescriptor(
      NS::UInteger rows,
      NS::UInteger columns,
      NS::UInteger rowBytes,
      NS::UInteger dataType);
  static class MatrixDescriptor* matrixDescriptor(
      NS::UInteger rows,
      NS::UInteger columns,
      NS::UInteger matrices,
      NS::UInteger rowBytes,
      NS::UInteger matrixBytes,
      NS::UInteger dataType);
  NS::UInteger rows() const;
};

class Matrix : public NS::Referencing<Matrix> {
 public:
  static class Matrix* alloc();
  Matrix* init(MTL::Buffer* buffer, MatrixDescriptor* descriptor);
  Matrix* init(const MTL::Buffer* buffer, MatrixDescriptor* descriptor);
};

class Kernel : public NS::Referencing<Kernel> {
 public:
  NS::String* label() const;
  MTL::Device* device() const;
};

class MatrixMultiplication
    : public NS::Referencing<MatrixMultiplication, Kernel> {
 public:
  static class MatrixMultiplication* alloc();

  MatrixMultiplication* init(
      MTL::Device* device,
      bool transposeLeft,
      bool transposeRight,
      NS::UInteger resultRows,
      NS::UInteger resultColumns,
      NS::UInteger interiorColumns,
      double alpha,
      double beta);

  void encodeToCommandBuffer(
      MTL::CommandBuffer* commandBuffer,
      Matrix* leftMatrix,
      Matrix* rightMatrix,
      Matrix* resultMatrix);

  void setLeftMatrixOrigin(MTL::Origin origin);
  void setRightMatrixOrigin(MTL::Origin origin);
  void setResultMatrixOrigin(MTL::Origin origin);
  void setBatchStart(NS::UInteger batchStart);
  void setBatchSize(NS::UInteger batchSize);
};

class VectorDescriptor : public NS::Copying<VectorDescriptor> {
 public:
  static class VectorDescriptor* vectorDescriptor(
      NS::UInteger length,
      NS::UInteger dataType);
  static class VectorDescriptor* vectorDescriptor(
      NS::UInteger length,
      NS::UInteger vectors,
      NS::UInteger vectorBytes,
      NS::UInteger dataType);
};

class Vector : public NS::Referencing<Vector> {
 public:
  static class Vector* alloc();
  Vector* init(MTL::Buffer* buffer, VectorDescriptor* descriptor);
  Vector* init(const MTL::Buffer* buffer, VectorDescriptor* descriptor);
};

class MatrixVectorMultiplication
    : public NS::Referencing<MatrixVectorMultiplication, Kernel> {
 public:
  static class MatrixVectorMultiplication* alloc();

  MatrixVectorMultiplication* init(
      MTL::Device* device,
      bool transpose,
      NS::UInteger rows,
      NS::UInteger columns,
      double alpha,
      double beta);

  void encodeToCommandBuffer(
      MTL::CommandBuffer* commandBuffer,
      Matrix* inputMatrix,
      Vector* inputVector,
      Vector* resultVector);
};

_MTL_INLINE MatrixDescriptor* MatrixDescriptor::matrixDescriptor(
    NS::UInteger rows,
    NS::UInteger columns,
    NS::UInteger rowBytes,
    NS::UInteger dataType) {
  return Object::sendMessage<MatrixDescriptor*>(
      _MPS_PRIVATE_CLS(MPSMatrixDescriptor),
      _MPS_PRIVATE_SEL(matrixDescriptorWithRows_columns_rowBytes_dataType),
      rows,
      columns,
      rowBytes,
      dataType);
}

_MTL_INLINE MatrixDescriptor* MatrixDescriptor::matrixDescriptor(
    NS::UInteger rows,
    NS::UInteger columns,
    NS::UInteger matrices,
    NS::UInteger rowBytes,
    NS::UInteger matrixBytes,
    NS::UInteger dataType) {
  return Object::sendMessage<MatrixDescriptor*>(
      _MPS_PRIVATE_CLS(MPSMatrixDescriptor),
      _MPS_PRIVATE_SEL(
          matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType),
      rows,
      columns,
      matrices,
      rowBytes,
      matrixBytes,
      dataType);
}

_MTL_INLINE NS::UInteger MatrixDescriptor::rows() const {
  return Object::sendMessage<NS::UInteger>(this, _MPS_PRIVATE_SEL(rows));
}

_MTL_INLINE Matrix* Matrix::alloc() {
  return NS::Object::alloc<Matrix>(_MPS_PRIVATE_CLS(MPSMatrix));
}

_MTL_INLINE Matrix* Matrix::init(
    MTL::Buffer* buffer,
    MatrixDescriptor* descriptor) {
  return Object::sendMessage<Matrix*>(
      this, _MPS_PRIVATE_SEL(initWithBuffer_descriptor), buffer, descriptor);
}

_MTL_INLINE Matrix* Matrix::init(
    const MTL::Buffer* buffer,
    MatrixDescriptor* descriptor) {
  return init(const_cast<MTL::Buffer*>(buffer), descriptor);
}

_MTL_INLINE NS::String* Kernel::label() const {
  return Object::sendMessage<NS::String*>(this, _MPS_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::Device* Kernel::device() const {
  return Object::sendMessage<MTL::Device*>(this, _MPS_PRIVATE_SEL(device));
}

_MTL_INLINE MatrixMultiplication* MatrixMultiplication::alloc() {
  return NS::Object::alloc<MatrixMultiplication>(
      _MPS_PRIVATE_CLS(MPSMatrixMultiplication));
}

_MTL_INLINE MatrixMultiplication* MatrixMultiplication::init(
    MTL::Device* device,
    bool transposeLeft,
    bool transposeRight,
    NS::UInteger resultRows,
    NS::UInteger resultColumns,
    NS::UInteger interiorColumns,
    double alpha,
    double beta) {
  return Object::sendMessage<MatrixMultiplication*>(
      this,
      _MPS_PRIVATE_SEL(initWithDevice_),
      device,
      transposeLeft,
      transposeRight,
      resultRows,
      resultColumns,
      interiorColumns,
      alpha,
      beta);
}

_MTL_INLINE void MatrixMultiplication::encodeToCommandBuffer(
    MTL::CommandBuffer* commandBuffer,
    Matrix* leftMatrix,
    Matrix* rightMatrix,
    Matrix* resultMatrix) {
  return Object::sendMessage<void>(
      this,
      _MPS_PRIVATE_SEL(
          encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix),
      commandBuffer,
      leftMatrix,
      rightMatrix,
      resultMatrix);
}

_MTL_INLINE void MatrixMultiplication::setLeftMatrixOrigin(MTL::Origin origin) {
  Object::sendMessage<void>(
      this, _MPS_PRIVATE_SEL(setLeftMatrixOrigin_), origin);
}

_MTL_INLINE void MatrixMultiplication::setRightMatrixOrigin(
    MTL::Origin origin) {
  Object::sendMessage<void>(
      this, _MPS_PRIVATE_SEL(setRightMatrixOrigin_), origin);
}

_MTL_INLINE void MatrixMultiplication::setResultMatrixOrigin(
    MTL::Origin origin) {
  Object::sendMessage<void>(
      this, _MPS_PRIVATE_SEL(setResultMatrixOrigin_), origin);
}

_MTL_INLINE void MatrixMultiplication::setBatchStart(NS::UInteger batchStart) {
  Object::sendMessage<void>(this, _MPS_PRIVATE_SEL(setBatchStart_), batchStart);
}

_MTL_INLINE void MatrixMultiplication::setBatchSize(NS::UInteger batchSize) {
  Object::sendMessage<void>(this, _MPS_PRIVATE_SEL(setBatchSize_), batchSize);
}

_MTL_INLINE VectorDescriptor* VectorDescriptor::vectorDescriptor(
    NS::UInteger length,
    NS::UInteger dataType) {
  return Object::sendMessage<VectorDescriptor*>(
      _MPS_PRIVATE_CLS(MPSVectorDescriptor),
      _MPS_PRIVATE_SEL(vectorDescriptorWithLength_dataType),
      length,
      dataType);
}

_MTL_INLINE VectorDescriptor* VectorDescriptor::vectorDescriptor(
    NS::UInteger length,
    NS::UInteger vectors,
    NS::UInteger vectorBytes,
    NS::UInteger dataType) {
  return Object::sendMessage<VectorDescriptor*>(
      _MPS_PRIVATE_CLS(MPSVectorDescriptor),
      _MPS_PRIVATE_SEL(vectorDescriptorWithLength_vectors_vectorBytes_dataType),
      length,
      vectors,
      vectorBytes,
      dataType);
}

_MTL_INLINE Vector* Vector::alloc() {
  return NS::Object::alloc<Vector>(_MPS_PRIVATE_CLS(MPSVector));
}

_MTL_INLINE Vector* Vector::init(
    MTL::Buffer* buffer,
    VectorDescriptor* descriptor) {
  return Object::sendMessage<Vector*>(
      this, _MPS_PRIVATE_SEL(initWithBuffer_descriptor), buffer, descriptor);
}

_MTL_INLINE Vector* Vector::init(
    const MTL::Buffer* buffer,
    VectorDescriptor* descriptor) {
  return init(const_cast<MTL::Buffer*>(buffer), descriptor);
}

_MTL_INLINE MatrixVectorMultiplication* MatrixVectorMultiplication::alloc() {
  return NS::Object::alloc<MatrixVectorMultiplication>(
      _MPS_PRIVATE_CLS(MPSMatrixVectorMultiplication));
}

_MTL_INLINE MatrixVectorMultiplication* MatrixVectorMultiplication::init(
    MTL::Device* device,
    bool transpose,
    NS::UInteger rows,
    NS::UInteger columns,
    double alpha,
    double beta) {
  return Object::sendMessage<MatrixVectorMultiplication*>(
      this,
      _MPS_PRIVATE_SEL(initWithDevice_transpose_rows_columns_alpha_beta),
      device,
      transpose,
      rows,
      columns,
      alpha,
      beta);
}

_MTL_INLINE void MatrixVectorMultiplication::encodeToCommandBuffer(
    MTL::CommandBuffer* commandBuffer,
    Matrix* inputMatrix,
    Vector* inputVector,
    Vector* resultVector) {
  return Object::sendMessage<void>(
      this,
      _MPS_PRIVATE_SEL(
          encodeToCommandBuffer_inputMatrix_inputVector_resultVector),
      commandBuffer,
      inputMatrix,
      inputVector,
      resultVector);
}

} // namespace MPS