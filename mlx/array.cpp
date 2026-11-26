// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <functional>

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
    : array_desc_(
          std::make_shared<ArrayDesc>(
              std::move(shape),
              dtype,
              std::move(primitive),
              std::move(inputs))) {
  if (has_primitive() && this->primitive().stream().device == Device::gpu) {
    for (auto& in : this->inputs()) {
      if (in.dtype() == float64) {
        throw std::invalid_argument("float64 is not supported on the GPU");
      }
    }
    if (this->dtype() == float64) {
      throw std::invalid_argument("float64 is not supported on the GPU");
    }
  }
}

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

array array::unsafe_weak_copy(const array& other) {
  auto cpy = array(other.shape(), other.dtype(), nullptr, {});
  cpy.set_data(
      Data(other.buffer(), nullptr),
      other.data_size(),
      other.strides(),
      other.flags(),
      other.offset());
  return cpy;
}

array::array(std::initializer_list<float> data)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              float32)) {
  init(data.begin());
}

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              dtype)) {
  init(data.begin());
}

array::array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  auto buffer = allocator::make_buffer(data, nbytes());
  if (buffer.ptr() == nullptr) {
    set_data(allocator::malloc(nbytes()));
    auto ptr = static_cast<char*>(data);
    std::copy(ptr, ptr + nbytes(), this->data<char>());
    deleter(data);
  } else {
    auto wrapped_deleter = [deleter](allocator::Buffer buffer) {
      auto ptr = buffer.raw_ptr();
      allocator::release(buffer);
      return deleter(ptr);
    };
    set_data(Data(buffer, std::move(wrapped_deleter)));
  }
}

array::array(allocator::Data data, Shape shape, Dtype dtype)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  set_data(std::move(data));
}

void array::detach() {
  array_desc_->primitive = nullptr;
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->primitive = nullptr;
  }
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->inputs.clear();
    s.array_desc_->siblings.clear();
    s.array_desc_->position = 0;
  }
  array_desc_->inputs.clear();
  array_desc_->siblings.clear();
  array_desc_->position = 0;
}

bool array::is_available() const {
  if (status() == Status::available) {
    return true;
  } else if (
      status() == Status::evaluated &&
      (!event().valid() || event().is_signaled())) {
    detach_event();
    set_status(Status::available);
    return true;
  }
  return false;
}

void array::wait() {
  if (!is_available()) {
    if (event().valid()) {
      event().wait();
      detach_event();
    }
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

void array::set_data(allocator::Data data) {
  array_desc_->data = std::make_shared<Data>(std::move(data));
  array_desc_->offset = 0;
  array_desc_->data_size = size();
  array_desc_->flags.contiguous = true;
  array_desc_->flags.row_contiguous = true;
  auto max_dim = std::max_element(shape().begin(), shape().end());
  array_desc_->flags.col_contiguous = size() <= 1 || size() == *max_dim;
}

void array::set_data(
    allocator::Data data,
    size_t data_size,
    Strides strides,
    Flags flags,
    int64_t offset) {
  array_desc_->data = std::make_shared<Data>(std::move(data));
  array_desc_->offset = offset;
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const Strides& strides,
    Flags flags,
    size_t data_size,
    int64_t offset /* = 0 */) {
  array_desc_->data = other.array_desc_->data;
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  array_desc_->offset =
      sizeof(char) * itemsize() * offset + other.array_desc_->offset;
}

void array::copy_shared_buffer(const array& other) {
  copy_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

void array::release(std::shared_ptr<ArrayDesc> desc) {
  if (desc == nullptr) {
    return;
  }

  // Detached/detaching
  if (desc->primitive == nullptr) {
    return;
  }

  // Break circular reference for non-detached arrays with siblings
  if (auto n = desc->siblings.size(); n > 0) {
    bool do_detach = true;
    // If all siblings have siblings.size() references except
    // the one we are currently releasing (which has siblings.size() + 1)
    // then there are no more external references
    do_detach &= (desc.use_count() == (n + 1));
    for (auto& s : desc->siblings) {
      do_detach &= (s.array_desc_.use_count() == n);
      if (!do_detach) {
        break;
      }
    }
    if (do_detach) {
      for (auto& s : desc->siblings) {
        for (auto& ss : s.siblings()) {
          // Set to null here to avoid descending into reset() for siblings
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
      primitive(std::move(primitive)),
      status(Status::unscheduled),
      inputs(std::move(inputs)) {
  init();
}

array::ArrayDesc::~ArrayDesc() {
  // Walk the graph instead of recursing, which can overflow the stack. The way
  // back up is kept in the slot the walk went down through.
  if (inputs.empty()) {
    return;
  }

  // Same rule as release(): only this slot and the siblings refer to a.
  auto is_last_reference = [](const array& a) {
    if (!a.array_desc_) {
      return false;
    }
    auto n = a.siblings().size();
    return std::cmp_equal(a.array_desc_.use_count() - 1, n) &&
        std::ranges::all_of(a.siblings(), [n](const array& s) {
             return std::cmp_equal(s.array_desc_.use_count(), n);
           });
  };

  // Break the sibling cycle so that d holds the only reference to each.
  auto take_siblings = [](ArrayDesc& d) {
    std::ranges::for_each(d.siblings, [](array& s) {
      std::ranges::for_each(
          s.array_desc_->siblings, [](array& ss) { ss.array_desc_ = nullptr; });
      s.array_desc_->siblings.clear();
    });
  };

  // Inputs first, then the siblings of a taken array.
  auto slot = [](ArrayDesc& d) -> array& {
    return d.inputs.empty() ? d.siblings.back() : d.inputs.back();
  };
  auto pop_slot = [](ArrayDesc& d) {
    if (d.inputs.empty()) {
      d.siblings.pop_back();
    } else {
      d.inputs.pop_back();
    }
  };

  // current points to this without owning it.
  auto current = std::shared_ptr<ArrayDesc>(std::shared_ptr<ArrayDesc>{}, this);
  std::shared_ptr<ArrayDesc> parent;
  while (true) {
    auto& node = *current;
    bool at_top = current.get() == this;
    if (!node.inputs.empty() || (!at_top && !node.siblings.empty())) {
      auto& next = slot(node);
      bool is_sibling = node.inputs.empty();
      if (!is_sibling && !is_last_reference(next)) {
        node.inputs.pop_back();
        continue;
      }
      auto& desc = *next.array_desc_;
      if (desc.inputs.empty() && desc.siblings.empty()) {
        pop_slot(node);
        continue;
      }
      if (!is_sibling) {
        take_siblings(desc);
      }
      // Nothing is left in node after this slot, so free it now.
      bool is_last_slot = is_sibling
          ? node.siblings.size() == 1
          : node.inputs.size() == 1 && node.siblings.empty();
      if (!at_top && is_last_slot) {
        auto down = std::move(next.array_desc_);
        pop_slot(node);
        current = std::move(down);
        continue;
      }
      auto down = std::exchange(next.array_desc_, std::move(parent));
      parent = std::exchange(current, std::move(down));
      continue;
    }
    if (at_top) {
      break;
    }
    // node is empty: go back up and destroy it.
    auto& up = *parent;
    auto done = std::move(current);
    current = std::exchange(parent, std::move(slot(up).array_desc_));
    pop_slot(up);
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
