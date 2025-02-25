// Copyright © 2025 Apple Inc.

#include "mlx/backend/webgpu/allocator.h"

#include "mlx/array.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace allocator {

Allocator& allocator() {
  return webgpu::allocator();
}

void* Buffer::raw_ptr() {
  return static_cast<webgpu::DoubleBuffer*>(ptr_)->cpu_data();
}

} // namespace allocator

namespace webgpu {

DoubleBuffer::DoubleBuffer(size_t size)
    : size_(size), cpu_data_(std::malloc(size)) {}

DoubleBuffer::DoubleBuffer(betann::Device& device, Dtype dtype, size_t size)
    : size_(size),
      gpu_data_(device.CreateBuffer(
          size * gpu_size_factor(dtype),
          betann::BufferUsage::Storage | betann::BufferUsage::CopySrc)) {}

DoubleBuffer::~DoubleBuffer() {
  std::free(cpu_data_);
}

WgpuAllocator::WgpuAllocator() : device_(webgpu::device(Device::gpu)) {}

Buffer WgpuAllocator::malloc(size_t size, bool allow_swap) {
  return Buffer(new DoubleBuffer(size));
}

void WgpuAllocator::free(Buffer buffer) {
  delete static_cast<DoubleBuffer*>(buffer.ptr());
}

size_t WgpuAllocator::size(Buffer buffer) const {
  return static_cast<DoubleBuffer*>(buffer.ptr())->size();
}

void WgpuAllocator::ensure_cpu_data(array& arr, const void* data) {
  auto* dbuf = static_cast<DoubleBuffer*>(arr.buffer().ptr());
  if (dbuf->cpu_data() || dbuf->size() == 0)
    return;
  void* cpu_data = std::malloc(dbuf->size());
  size_t num_elements = dbuf->size() / arr.itemsize();
  switch (arr.dtype()) {
    case int32:
    case uint32:
    case float16:
    case float32:
      std::memcpy(cpu_data, data, dbuf->size());
      break;
    case bool_:
      std::transform(
          static_cast<const uint32_t*>(data),
          static_cast<const uint32_t*>(data) + num_elements,
          static_cast<bool*>(cpu_data),
          [](uint32_t e) { return static_cast<bool>(e); });
      break;
    case uint8:
      std::transform(
          static_cast<const uint32_t*>(data),
          static_cast<const uint32_t*>(data) + num_elements,
          static_cast<uint8_t*>(cpu_data),
          [](uint32_t e) { return static_cast<uint8_t>(e); });
      break;
    case uint16:
      std::transform(
          static_cast<const uint32_t*>(data),
          static_cast<const uint32_t*>(data) + num_elements,
          static_cast<uint16_t*>(cpu_data),
          [](uint32_t e) { return static_cast<uint16_t>(e); });
      break;
    case int8:
      std::transform(
          static_cast<const int32_t*>(data),
          static_cast<const int32_t*>(data) + num_elements,
          static_cast<int8_t*>(cpu_data),
          [](int32_t e) { return static_cast<int8_t>(e); });
      break;
    case int16:
      std::transform(
          static_cast<const int32_t*>(data),
          static_cast<const int32_t*>(data) + num_elements,
          static_cast<int16_t*>(cpu_data),
          [](int32_t e) { return static_cast<int16_t>(e); });
      break;
    default:
      throw_unsupported_dtype_error(arr.dtype());
  }
  dbuf->set_cpu_data(cpu_data);
}

void WgpuAllocator::ensure_gpu_data(array& arr) {
  auto* dbuf = static_cast<DoubleBuffer*>(arr.buffer().ptr());
  if (dbuf->gpu_data() || dbuf->size() == 0)
    return;
  size_t num_elements = dbuf->size() / arr.itemsize();
  switch (arr.dtype()) {
    case int32:
    case uint32:
    case float16:
    case float32:
      dbuf->set_gpu_data(
          device_.CreateBufferFromData(dbuf->cpu_data(), dbuf->size()));
      break;
    case bool_:
      dbuf->set_gpu_data(device_.CreateBufferTransformTo<uint32_t>(
          static_cast<bool*>(dbuf->cpu_data()), num_elements));
      break;
    case uint8:
      dbuf->set_gpu_data(device_.CreateBufferTransformTo<uint32_t>(
          static_cast<uint8_t*>(dbuf->cpu_data()), num_elements));
      break;
    case uint16:
      dbuf->set_gpu_data(device_.CreateBufferTransformTo<uint32_t>(
          static_cast<uint16_t*>(dbuf->cpu_data()), num_elements));
      break;
    case int8:
      dbuf->set_gpu_data(device_.CreateBufferTransformTo<int32_t>(
          static_cast<int8_t*>(dbuf->cpu_data()), num_elements));
      break;
    case int16:
      dbuf->set_gpu_data(device_.CreateBufferTransformTo<int32_t>(
          static_cast<int16_t*>(dbuf->cpu_data()), num_elements));
      break;
    default:
      throw_unsupported_dtype_error(arr.dtype());
  }
}

Buffer WgpuAllocator::malloc_gpu(array& arr) {
  return malloc_gpu(arr, arr.nbytes());
}

Buffer WgpuAllocator::malloc_gpu(array& arr, size_t size) {
  return Buffer(new DoubleBuffer(device_, arr.dtype(), size));
}

WgpuAllocator& allocator() {
  static WgpuAllocator allocator_;
  return allocator_;
}

betann::Device& device(mlx::core::Device) {
  static betann::Device device;
  return device;
}

betann::Device& device(array& arr) {
  return device(arr.primitive().device());
}

} // namespace webgpu

namespace metal {

size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
void reset_peak_memory() {}
size_t get_cache_memory() {
  return 0;
}
size_t set_memory_limit(size_t, bool) {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}

std::unordered_map<std::string, std::variant<std::string, size_t>>
device_info() {
  throw std::runtime_error("[webgpu::device_info] Not implemented");
};

void clear_cache() {}

} // namespace metal

} // namespace mlx::core
