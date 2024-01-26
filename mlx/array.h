// Copyright © 2023 Apple Inc.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/dtype.h"

namespace mlx::core {

// Forward declaration
class Primitive;
using deleter_t = std::function<void(allocator::Buffer)>;

class array {
  /* An array is really a node in a graph. It contains a shared ArrayDesc
   * object */

 public:
  /** Construct a scalar array with zero dimensions. */
  template <typename T>
  explicit array(T val, Dtype dtype = TypeToDtype<T>());

  /* Special case since std::complex can't be implicitly converted to other
   * types. */
  explicit array(const std::complex<float>& val, Dtype dtype = complex64);

  template <typename It>
  array(
      It data,
      const std::vector<int>& shape,
      Dtype dtype =
          TypeToDtype<typename std::iterator_traits<It>::value_type>());

  template <typename T>
  array(std::initializer_list<T> data, Dtype dtype = TypeToDtype<T>());

  /* Special case so empty lists default to float32. */
  array(std::initializer_list<float> data);

  template <typename T>
  array(
      std::initializer_list<T> data,
      const std::vector<int>& shape,
      Dtype dtype = TypeToDtype<T>());

  /* Build an array from a buffer */
  array(
      allocator::Buffer data,
      const std::vector<int>& shape,
      Dtype dtype,
      deleter_t deleter = allocator::free);

  /** Assignment to rvalue does not compile. */
  array& operator=(const array& other) && = delete;
  array& operator=(array&& other) && = delete;

  /** Default copy and move constructors otherwise. */
  array& operator=(array&& other) & = default;
  array(const array& other) = default;
  array(array&& other) = default;

  array& operator=(const array& other) & {
    if (this->id() != other.id()) {
      this->array_desc_ = other.array_desc_;
    }
    return *this;
  };

  /** The size of the array's datatype in bytes. */
  size_t itemsize() const {
    return size_of(dtype());
  };

  /** The number of elements in the array. */
  size_t size() const {
    return array_desc_->size;
  };

  /** The number of bytes in the array. */
  size_t nbytes() const {
    return size() * itemsize();
  };

  /** The number of dimensions of the array. */
  size_t ndim() const {
    return array_desc_->shape.size();
  };

  /** The shape of the array as a vector of integers. */
  const std::vector<int>& shape() const {
    return array_desc_->shape;
  };

  /**
   *  Get the size of the corresponding dimension.
   *
   *  This function supports negative indexing and provides
   *  bounds checking. */
  int shape(int dim) const {
    return shape().at(dim < 0 ? dim + ndim() : dim);
  };

  /** The strides of the array. */
  const std::vector<size_t>& strides() const {
    return array_desc_->strides;
  };

  /** Get the arrays data type. */
  Dtype dtype() const {
    return array_desc_->dtype;
  };

  /** Evaluate the array. */
  void eval();

  /** Get the value from a scalar array. */
  template <typename T>
  T item();

  struct ArrayIterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = size_t;
    using value_type = const array;
    using reference = value_type;

    explicit ArrayIterator(const array& arr, int idx = 0);

    reference operator*() const;

    ArrayIterator& operator+(difference_type diff) {
      idx += diff;
      return *this;
    }

    ArrayIterator& operator++() {
      idx++;
      return *this;
    }

    friend bool operator==(const ArrayIterator& a, const ArrayIterator& b) {
      return a.arr.id() == b.arr.id() && a.idx == b.idx;
    };
    friend bool operator!=(const ArrayIterator& a, const ArrayIterator& b) {
      return !(a == b);
    };

   private:
    const array& arr;
    int idx;
  };

  ArrayIterator begin() const {
    return ArrayIterator(*this);
  }
  ArrayIterator end() const {
    return ArrayIterator(*this, shape(0));
  }

  /**
   * The following methods should be used with caution.
   * They are intended for use by the backend implementation and the
   * API may change.
   */

  array(
      const std::vector<int>& shape,
      Dtype dtype,
      std::shared_ptr<Primitive> primitive,
      const std::vector<array>& inputs);

  array(
      std::vector<int> shape,
      Dtype dtype,
      std::shared_ptr<Primitive> primitive,
      std::vector<array>&& inputs);

  static std::vector<array> make_arrays(
      const std::vector<std::vector<int>>& shapes,
      const std::vector<Dtype>& dtypes,
      std::shared_ptr<Primitive> primitive,
      const std::vector<array>& inputs);

  /** A unique identifier for an array. */
  std::uintptr_t id() const {
    return reinterpret_cast<std::uintptr_t>(array_desc_.get());
  }

  /** A unique identifier for an arrays primitive. */
  std::uintptr_t primitive_id() const {
    return reinterpret_cast<std::uintptr_t>(array_desc_->primitive.get());
  }

  struct Data {
    allocator::Buffer buffer;
    deleter_t d;
    Data(allocator::Buffer buffer, deleter_t d = allocator::free)
        : buffer(buffer), d(d){};
    // Not copyable
    Data(const Data& d) = delete;
    Data& operator=(const Data& d) = delete;
    ~Data() {
      d(buffer);
    }
  };

  struct Flags {
    // True if there are no gaps in the underlying data. Each item
    // in the underlying data buffer belongs to at least one index.
    bool contiguous : 1;

    bool row_contiguous : 1;
    bool col_contiguous : 1;
  };

  /** The array's primitive. */
  Primitive& primitive() const {
    return *(array_desc_->primitive);
  };

  /** A shared pointer to the array's primitive. */
  std::shared_ptr<Primitive>& primitive_ptr() const {
    return array_desc_->primitive;
  };

  /** Check if the array has an attached primitive or is a leaf node. */
  bool has_primitive() const {
    return array_desc_->primitive != nullptr;
  };

  /** The array's inputs. */
  const std::vector<array>& inputs() const {
    return array_desc_->inputs;
  };

  std::vector<array>& inputs() {
    return array_desc_->inputs;
  }

  /** The array's siblings. */
  const std::vector<array>& siblings() const {
    return array_desc_->siblings;
  };

  void set_siblings(std::vector<array> siblings, uint16_t position) {
    array_desc_->siblings = std::move(siblings);
    array_desc_->position = position;
  }

  /** The outputs of the array's primitive (i.e. this array and
   * its siblings) in the order the primitive expects. */
  std::vector<array> outputs() const {
    auto idx = array_desc_->position;
    std::vector<array> outputs;
    outputs.reserve(siblings().size() + 1);
    outputs.insert(outputs.end(), siblings().begin(), siblings().begin() + idx);
    outputs.push_back(*this);
    outputs.insert(outputs.end(), siblings().begin() + idx, siblings().end());
    return outputs;
  };

  /** Detach the array from the graph. */
  void detach();

  /** Get the Flags bit-field. */
  const Flags& flags() const {
    return array_desc_->flags;
  };

  /** The size (in elements) of the underlying buffer the array points to. */
  size_t data_size() const {
    return array_desc_->data_size;
  };

  allocator::Buffer& buffer() {
    return array_desc_->data->buffer;
  };
  const allocator::Buffer& buffer() const {
    return array_desc_->data->buffer;
  };

  template <typename T>
  T* data() {
    return static_cast<T*>(array_desc_->data_ptr);
  };

  template <typename T>
  const T* data() const {
    return static_cast<T*>(array_desc_->data_ptr);
  };

  // Check if the array has been evaluated
  bool is_evaled() const {
    return array_desc_->data != nullptr;
  }

  // Mark the array as a tracer array (true) or not.
  void set_tracer(bool is_tracer) {
    array_desc_->is_tracer = is_tracer;
  }
  // Check if the array is a tracer array
  bool is_tracer() const;

  void set_data(allocator::Buffer buffer, deleter_t d = allocator::free);

  void set_data(
      allocator::Buffer buffer,
      size_t data_size,
      std::vector<size_t> strides,
      Flags flags,
      deleter_t d = allocator::free);

  void copy_shared_buffer(
      const array& other,
      const std::vector<size_t>& strides,
      Flags flags,
      size_t data_size,
      size_t offset = 0);

  void copy_shared_buffer(const array& other);

  void overwrite_descriptor(const array& other) {
    array_desc_ = other.array_desc_;
  }

 private:
  // Initialize the arrays data
  template <typename It>
  void init(const It src);

  struct ArrayDesc {
    std::vector<int> shape;
    std::vector<size_t> strides;
    size_t size;
    Dtype dtype;
    std::shared_ptr<Primitive> primitive{nullptr};

    // Indicates an array is being used in a graph transform
    // and should not be detached from the graph
    bool is_tracer{false};

    // This is a shared pointer so that *different* arrays
    // can share the underlying data buffer.
    std::shared_ptr<Data> data{nullptr};

    // Properly offset data pointer
    void* data_ptr{nullptr};

    // The size in elements of the data buffer the array accesses
    // This can be different than the actual size of the array if it
    // has been broadcast or irregularly strided.
    size_t data_size;

    // Contains useful meta data about the array
    Flags flags;

    std::vector<array> inputs;
    // An array to keep track of the siblings from a multi-output
    // primitive.
    std::vector<array> siblings;
    // The arrays position in the output list
    uint32_t position{0};

    explicit ArrayDesc(const std::vector<int>& shape, Dtype dtype);

    explicit ArrayDesc(
        const std::vector<int>& shape,
        Dtype dtype,
        std::shared_ptr<Primitive> primitive,
        const std::vector<array>& inputs);

    explicit ArrayDesc(
        std::vector<int>&& shape,
        Dtype dtype,
        std::shared_ptr<Primitive> primitive,
        std::vector<array>&& inputs);
  };

  // The ArrayDesc contains the details of the materialized array including the
  // shape, strides, the data type. It also includes
  // the primitive which knows how to compute the array's data from its inputs
  // and a the list of array's inputs for the primitive.
  std::shared_ptr<ArrayDesc> array_desc_{nullptr};
};

template <typename T>
array::array(T val, Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(std::vector<int>{}, dtype)) {
  init(&val);
}

template <typename It>
array::array(
  It data,
  const std::vector<int>& shape,
  Dtype dtype /* = TypeToDtype<typename std::iterator_traits<It>::value_type>() */) :
    array_desc_(std::make_shared<ArrayDesc>(shape, dtype)) {
  init(data);
}

template <typename T>
array::array(
    std::initializer_list<T> data,
    Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::vector<int>{static_cast<int>(data.size())},
          dtype)) {
  init(data.begin());
}

template <typename T>
array::array(
    std::initializer_list<T> data,
    const std::vector<int>& shape,
    Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(shape, dtype)) {
  if (data.size() != size()) {
    throw std::invalid_argument(
        "Data size and provided shape mismatch in array construction.");
  }
  init(data.begin());
}

template <typename T>
T array::item() {
  if (size() != 1) {
    throw std::invalid_argument("item can only be called on arrays of size 1.");
  }
  eval();
  return *data<T>();
}

template <typename It>
void array::init(It src) {
  set_data(allocator::malloc(size() * size_of(dtype())));
  switch (dtype()) {
    case bool_:
      std::copy(src, src + size(), data<bool>());
      break;
    case uint8:
      std::copy(src, src + size(), data<uint8_t>());
      break;
    case uint16:
      std::copy(src, src + size(), data<uint16_t>());
      break;
    case uint32:
      std::copy(src, src + size(), data<uint32_t>());
      break;
    case uint64:
      std::copy(src, src + size(), data<uint64_t>());
      break;
    case int8:
      std::copy(src, src + size(), data<int8_t>());
      break;
    case int16:
      std::copy(src, src + size(), data<int16_t>());
      break;
    case int32:
      std::copy(src, src + size(), data<int32_t>());
      break;
    case int64:
      std::copy(src, src + size(), data<int64_t>());
      break;
    case float16:
      std::copy(src, src + size(), data<float16_t>());
      break;
    case float32:
      std::copy(src, src + size(), data<float>());
      break;
    case bfloat16:
      std::copy(src, src + size(), data<bfloat16_t>());
      break;
    case complex64:
      std::copy(src, src + size(), data<complex64_t>());
      break;
  }
}

} // namespace mlx::core
