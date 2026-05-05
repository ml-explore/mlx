// Copyright © 2026 Apple Inc.

#include <Metal/Metal.hpp>

#include <memory>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "python/src/dlpack_consumer.h"
#include "python/src/dlpack_format.h"

mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& t,
    std::shared_ptr<DLPackOwner> owner) {
  if (!mx::metal::is_available()) {
    throw std::invalid_argument(
        "[array] Metal device tensors require an MLX build with Metal "
        "support enabled and a Metal-capable host.");
  }
  if (t.data == nullptr) {
    throw std::invalid_argument(
        "[array] kDLMetal capsule has null MTLBuffer pointer.");
  }
  // For kDLMetal, DLPack stipulates `data` is an MTL::Buffer*.
  auto mtl_buffer = static_cast<MTL::Buffer*>(t.data);
  if (mtl_buffer->storageMode() != MTL::StorageModeShared) {
    throw std::invalid_argument(
        "[array] foreign MTLBuffer must use MTLStorageModeShared. MLX "
        "currently relies on shared-mode buffers for read/write access. "
        "Allocate the producer-side buffer with MTLResourceStorageModeShared "
        "before exporting via DLPack.");
  }
  if (t.byte_offset != 0) {
    throw std::invalid_argument(
        "[array] kDLMetal capsule with non-zero byte_offset is not "
        "supported yet.");
  }
  auto shape = validate_and_extract_shape(t);
  if (!is_row_contiguous(shape, t.strides)) {
    throw std::invalid_argument(
        "[array] non-row-contiguous DLPack strides are not supported. "
        "Reshape on the producer side before exporting.");
  }

  auto dtype = dlpack_to_mlx_dtype(t.dtype);
  size_t nbytes = checked_num_bytes(shape, dtype);
  if (nbytes > mtl_buffer->length()) {
    throw std::invalid_argument(
        "[array] kDLMetal capsule shape/dtype requires more bytes than "
        "the exported MTLBuffer contains.");
  }

  // Wrap the foreign MTL::Buffer* directly. The producer retains the
  // underlying allocation; we drive the capsule's deleter when the wrapping
  // mx::array (and any aliases) are destroyed.
  mx::allocator::Buffer wrapped(static_cast<void*>(mtl_buffer));
  mx::Deleter deleter = [owner](mx::allocator::Buffer) mutable {
    // Drop our shared_ptr; if this was the last reference, the owner's
    // destructor invokes the DLPack deleter.
    owner.reset();
  };

  return mx::array(wrapped, std::move(shape), dtype, std::move(deleter));
}
