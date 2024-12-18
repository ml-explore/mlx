// Copyright © 2023-2024 Apple Inc.
#include <functional>
#include <unordered_map>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

array::array(const std::complex<float>& val, Dtype dtype /* = complex64 */)
    : array_desc_(std::make_shared<ArrayDesc>(Shape{}, dtype)) {
  auto cval = static_cast<complex64_t>(val);
  init(&cval);
}

array::array(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::move(shape),
          dtype,
          std::move(primitive),
          std::move(inputs))) {}

std::vector<array> array::make_arrays(
    std::vector<Shape> shapes,
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
          Shape{static_cast<ShapeElem>(data.size())},
          float32)) {
  init(data.begin());
}

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(std::make_shared<ArrayDesc>(
          Shape{static_cast<ShapeElem>(data.size())},
          dtype)) {
  init(data.begin());
}

/* Build an array from a shared buffer */
array::array(allocator::Buffer data, Shape shape, Dtype dtype, Deleter deleter)
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

bool array::is_available() const {
  if (status() == Status::available) {
    return true;
  } else if (status() == Status::evaluated && event().is_signaled()) {
    set_status(Status::available);
    return true;
  }
  return false;
}

void array::wait() {
  if (!is_available()) {
    event().wait();
    set_status(Status::available);
  }
}

void array::eval() {
  // Ensure the array is ready to be read
  if (status() == Status::unscheduled) {
    mlx::core::eval({*this});
  } else {
    wait();
  }
}

bool array::is_tracer() const {
  return (array_desc_->is_tracer && detail::in_tracing()) ||
      detail::retain_graph();
}

void array::set_data(allocator::Buffer buffer, Deleter d) {
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
    Strides strides,
    Flags flags,
    Deleter d) {
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->data_ptr = buffer.raw_ptr();
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const Strides& strides,
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
    const Strides& strides,
    Flags flags,
    size_t data_size,
    size_t offset /* = 0 */) {
  array_desc_->data = std::move(other.array_desc_->data);
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  auto char_offset = sizeof(char) * itemsize() * offset;
  auto data_ptr = other.array_desc_->data_ptr;
  other.array_desc_->data_ptr = nullptr;
  array_desc_->data_ptr =
      static_cast<void*>(static_cast<char*>(data_ptr) + char_offset);
}

void array::move_shared_buffer(array other) {
  move_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::~array() {
  if (array_desc_ == nullptr) {
    return;
  }

  // Ignore arrays that might be detached during eval
  if (status() == array::Status::scheduled) {
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
          // Set to null here to avoid descending into array destructor
          // for siblings
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
  for (const auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

array::ArrayDesc::ArrayDesc(Shape shape, Dtype dtype)
    : shape(std::move(shape)), dtype(dtype), status(Status::available) {
  init();
}

array::ArrayDesc::ArrayDesc(
    Shape shape,
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
  // that may also destroy their corresponding descriptions and so on and so
  // forth.
  //
  // This calls recursively the destructor and can result in stack overflow, we
  // instead put them in a vector and destroy them one at a time resulting in a
  // max stack depth of 2.
  if (inputs.empty()) {
    return;
  }

  std::vector<std::shared_ptr<ArrayDesc>> for_deletion;

  auto append_deletable_inputs = [&for_deletion](ArrayDesc& ad) {
    std::unordered_map<std::uintptr_t, array> input_map;
    for (array& a : ad.inputs) {
      if (a.array_desc_) {
        input_map.insert({a.id(), a});
        for (auto& s : a.siblings()) {
          input_map.insert({s.id(), s});
        }
      }
    }
    ad.inputs.clear();
    for (auto& [_, a] : input_map) {
      bool is_deletable =
          (a.array_desc_.use_count() <= a.siblings().size() + 1);
      // An array with siblings is deletable only if all of its siblings
      // are deletable
      for (auto& s : a.siblings()) {
        if (!is_deletable) {
          break;
        }
        int is_input = (input_map.find(s.id()) != input_map.end());
        is_deletable &=
            s.array_desc_.use_count() <= a.siblings().size() + is_input;
      }
      if (is_deletable) {
        for_deletion.push_back(std::move(a.array_desc_));
      }
    }
  };

  append_deletable_inputs(*this);

  while (!for_deletion.empty()) {
    // top is going to be deleted at the end of the block *after* the arrays
    // with inputs have been moved into the vector
    auto top = std::move(for_deletion.back());
    for_deletion.pop_back();
    append_deletable_inputs(*top);

    // Clear out possible siblings to break circular references
    for (auto& s : top->siblings) {
      // Set to null here to avoid descending into top-level
      // array destructor for siblings
      s.array_desc_ = nullptr;
    }
    top->siblings.clear();
  }
}

array::ArrayIterator::ArrayIterator(const array& arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

array::ArrayIterator::reference array::ArrayIterator::operator*() const {
  auto start = Shape(arr.ndim(), 0);
  auto end = arr.shape();
  auto shape = arr.shape();
  shape.erase(shape.begin());
  start[0] = idx;
  end[0] = idx + 1;
  return reshape(slice(arr, start, end), shape);
};

} // namespace mlx::core
