// Copyright © 2023 Apple Inc.

#include <functional>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx::core {

namespace {

std::pair<size_t, std::vector<size_t>> cum_prod(const std::vector<int>& shape) {
  std::vector<size_t> strides(shape.size());
  size_t cum_prod = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = cum_prod;
    cum_prod *= shape[i];
  }
  return {cum_prod, strides};
}

} // namespace

array::array(const std::complex<float>& val, Dtype dtype /* = complex64 */)
    : array_desc_(std::make_shared<ArrayDesc>(std::vector<int>{}, dtype)) {
  auto cval = static_cast<complex64_t>(val);
  init(&cval);
}

array::array(
    const std::vector<int>& shape,
    Dtype dtype,
    std::unique_ptr<Primitive> primitive,
    const std::vector<array>& inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          shape,
          dtype,
          std::move(primitive),
          inputs)) {}

array::array(std::initializer_list<float> data)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::vector<int>{static_cast<int>(data.size())},
          float32)) {
  init(data.begin());
}

/* Build an array from a shared buffer */
array::array(
    allocator::Buffer data,
    const std::vector<int>& shape,
    Dtype dtype,
    deleter_t deleter)
    : array_desc_(std::make_shared<ArrayDesc>(shape, dtype)) {
  set_data(data, deleter);
}

void array::detach() {
  array_desc_->inputs.clear();
  array_desc_->primitive = nullptr;
}

void array::eval(bool retain_graph /* = false */) {
  mlx::core::eval({*this}, retain_graph);
}

void array::set_data(allocator::Buffer buffer, deleter_t d) {
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->data_ptr = buffer.raw_ptr();
  array_desc_->data_size = size();
  array_desc_->flags.contiguous = true;
  array_desc_->flags.row_contiguous = true;
  auto max_dim = std::max_element(shape().begin(), shape().end());
  array_desc_->flags.col_contiguous = size() <= 1 || size() == *max_dim;
}

void array::set_data(
    allocator::Buffer buffer,
    size_t data_size,
    std::vector<size_t> strides,
    Flags flags,
    deleter_t d) {
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->data_ptr = buffer.raw_ptr();
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const std::vector<size_t>& strides,
    Flags flags,
    size_t data_size,
    size_t offset /* = 0 */) {
  array_desc_->data = other.array_desc_->data;
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  auto char_offset = sizeof(char) * itemsize() * offset;
  array_desc_->data_ptr = static_cast<void*>(
      static_cast<char*>(other.array_desc_->data_ptr) + char_offset);
}

void array::copy_shared_buffer(const array& other) {
  copy_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::ArrayDesc::ArrayDesc(const std::vector<int>& shape, Dtype dtype)
    : shape(shape), dtype(dtype) {
  std::tie(size, strides) = cum_prod(shape);
}

array::ArrayDesc::ArrayDesc(
    const std::vector<int>& shape,
    Dtype dtype,
    std::unique_ptr<Primitive> primitive,
    const std::vector<array>& inputs)
    : shape(shape),
      dtype(dtype),
      primitive(std::move(primitive)),
      inputs(inputs) {
  std::tie(size, strides) = cum_prod(shape);
  for (auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

// Needed because the Primitive type used in array.h is incomplete and the
// compiler needs to see the call to the desctructor after the type is complete.
array::ArrayDesc::~ArrayDesc() = default;

array::ArrayIterator::reference array::ArrayIterator::operator*() const {
  auto start = std::vector<int>(arr.ndim(), 0);
  auto end = arr.shape();
  auto shape = arr.shape();
  shape.erase(shape.begin());
  start[0] = idx;
  end[0] = idx + 1;
  return reshape(slice(arr, start, end), shape);
};

} // namespace mlx::core
