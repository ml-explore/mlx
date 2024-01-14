// Copyright © 2023 Apple Inc.

#include <functional>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

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

/** Return true if we are currently performing a function transformation in
 * order to keep the graph when evaluating tracer arrays. */
bool in_tracing() {
  return detail::InTracing::in_tracing();
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
    std::shared_ptr<Primitive> primitive,
    const std::vector<array>& inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          shape,
          dtype,
          std::move(primitive),
          inputs)) {}

array::array(
    std::vector<int> shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array>&& inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::move(shape),
          dtype,
          std::move(primitive),
          std::move(inputs))) {}

std::vector<array> array::make_arrays(
    const std::vector<std::vector<int>>& shapes,
    const std::vector<Dtype>& dtypes,
    std::shared_ptr<Primitive> primitive,
    const std::vector<array>& inputs) {
  std::vector<array> outputs;
  for (int i = 0; i < shapes.size(); ++i) {
    outputs.push_back(array(shapes[i], dtypes[i], primitive, inputs));
  }
  for (int i = 0; i < outputs.size(); ++i) {
    auto siblings = outputs;
    siblings.erase(siblings.begin() + i);
    outputs[i].set_siblings(std::move(siblings), i);
  }
  return outputs;
}

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
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->inputs.clear();
    s.array_desc_->siblings.clear();
    s.array_desc_->position = 0;
    s.array_desc_->primitive = nullptr;
  }
  array_desc_->inputs.clear();
  array_desc_->siblings.clear();
  array_desc_->position = 0;
  array_desc_->primitive = nullptr;
}

void array::eval() {
  mlx::core::eval({*this});
}

bool array::is_tracer() const {
  return array_desc_->is_tracer && in_tracing();
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
    std::shared_ptr<Primitive> primitive,
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

array::ArrayDesc::ArrayDesc(
    std::vector<int>&& shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array>&& inputs)
    : shape(std::move(shape)),
      dtype(dtype),
      primitive(std::move(primitive)),
      inputs(std::move(inputs)) {
  std::tie(size, strides) = cum_prod(shape);
  for (auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

array::ArrayIterator::ArrayIterator(const array& arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

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
