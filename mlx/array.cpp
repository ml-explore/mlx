// Copyright © 2023-2024 Apple Inc.
#include <functional>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

namespace {

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
    std::vector<int> shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::move(shape),
          dtype,
          std::move(primitive),
          std::move(inputs))) {}

std::vector<array> array::make_arrays(
    std::vector<std::vector<int>> shapes,
    const std::vector<Dtype>& dtypes,
    const std::shared_ptr<Primitive>& primitive,
    const std::vector<array>& inputs) {
  std::vector<array> outputs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    outputs.emplace_back(std::move(shapes[i]), dtypes[i], primitive, inputs);
  }
  // For each node in |outputs|, its siblings are the other nodes.
  for (size_t i = 0; i < outputs.size(); ++i) {
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

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::vector<int>{static_cast<int>(data.size())},
          dtype)) {
  init(data.begin());
}

/* Build an array from a shared buffer */
array::array(
    allocator::Buffer data,
    std::vector<int> shape,
    Dtype dtype,
    deleter_t deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
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
  // Ensure the array is ready to be read
  if (status() == Status::scheduled) {
    event().wait();
    set_status(Status::available);
  } else if (status() == Status::unscheduled) {
    mlx::core::eval({*this});
  }
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

void array::move_shared_buffer(
    array other,
    const std::vector<size_t>& strides,
    Flags flags,
    size_t data_size,
    size_t offset /* = 0 */) {
  array_desc_->data = std::move(other.array_desc_->data);
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  auto char_offset = sizeof(char) * itemsize() * offset;
  array_desc_->data_ptr = static_cast<void*>(
      static_cast<char*>(other.array_desc_->data_ptr) + char_offset);
}

void array::move_shared_buffer(array other) {
  move_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::~array() {
  if (array_desc_ == nullptr) {
    return;
  }

  // Ignore arrays that will be detached
  if (status() != array::Status::unscheduled) {
    return;
  }
  // Break circular reference for non-detached arrays with siblings
  if (auto n = siblings().size(); n > 0) {
    bool do_detach = true;
    // If all siblings have siblings.size() references except
    // the one we are currently destroying (which has siblings.size() + 1)
    // then there are no more external references
    do_detach &= (array_desc_.use_count() == (n + 1));
    for (auto& s : siblings()) {
      do_detach &= (s.array_desc_.use_count() == n);
      if (!do_detach) {
        break;
      }
    }
    if (do_detach) {
      for (auto& s : siblings()) {
        for (auto& ss : s.siblings()) {
          ss.array_desc_ = nullptr;
        }
        s.array_desc_->siblings.clear();
      }
    }
  }
}

void array::ArrayDesc::init() {
  strides.resize(shape.size());
  size = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = size;
    size *= shape[i];
  }
  for (auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

array::ArrayDesc::ArrayDesc(std::vector<int> shape, Dtype dtype)
    : shape(std::move(shape)), dtype(dtype), status(Status::available) {
  init();
}

array::ArrayDesc::ArrayDesc(
    std::vector<int> shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : shape(std::move(shape)),
      dtype(dtype),
      status(Status::unscheduled),
      primitive(std::move(primitive)),
      inputs(std::move(inputs)) {
  init();
}

array::ArrayDesc::~ArrayDesc() {
  // When an array description is destroyed it will delete a bunch of arrays
  // that may also destory their corresponding descriptions and so on and so
  // forth.
  //
  // This calls recursively the destructor and can result in stack overflow, we
  // instead put them in a vector and destroy them one at a time resulting in a
  // max stack depth of 2.
  std::vector<std::shared_ptr<ArrayDesc>> for_deletion;

  for (array& a : inputs) {
    if (a.array_desc_.use_count() == 1) {
      for_deletion.push_back(std::move(a.array_desc_));
    }
  }

  while (!for_deletion.empty()) {
    // top is going to be deleted at the end of the block *after* the arrays
    // with inputs have been moved into the vector
    auto top = std::move(for_deletion.back());
    for_deletion.pop_back();

    for (array& a : top->inputs) {
      if (a.array_desc_.use_count() == 1) {
        for_deletion.push_back(std::move(a.array_desc_));
      }
    }
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
