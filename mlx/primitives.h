// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_set>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"

#define DEFINE_VMAP()                                                 \
  virtual std::pair<std::vector<array>, std::vector<int>> vmap(       \
      const std::vector<array>& inputs, const std::vector<int>& axes) \
      override;

#define DEFINE_GRADS()                           \
  std::vector<array> jvp(                        \
      const std::vector<array>& primals,         \
      const std::vector<array>& tangents,        \
      const std::vector<int>& argnums) override; \
                                                 \
  std::vector<array> vjp(                        \
      const std::vector<array>& primals,         \
      const std::vector<array>& cotangents,      \
      const std::vector<int>& argnums,           \
      const std::vector<array>& outputs) override;

#define DEFINE_PRINT(PRIMITIVE)           \
  void print(std::ostream& os) override { \
    os << #PRIMITIVE;                     \
  }

#define DEFINE_DEFAULT_IS_EQUIVALENT()                        \
  bool is_equivalent(const Primitive& other) const override { \
    return true;                                              \
  }

#define DEFINE_INPUT_OUTPUT_SHAPE()                                  \
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) \
      override {                                                     \
    return {inputs[0].shape()};                                      \
  }

namespace mlx::core {

// Abstract base class
class Primitive {
 public:
  explicit Primitive(Stream stream) : stream_(stream) {}

  /** The device the primitive will run on. */
  const Device& device() {
    return stream().device;
  }

  /** The stream the primitive will run on. */
  const Stream& stream() {
    return stream_;
  }

  /**
   * A primitive must know how to evaluate itself on
   * the CPU/GPU for the given inputs and populate the output arrays.
   *
   * To avoid unnecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  virtual void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;
  virtual void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;

  /**
   * The Jacobian-vector product.
   */
  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums);

  /**
   * The vector-Jacobian product.
   */
  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs);

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the output arrays
   * representing the vectorized computation and the axes which
   * corresponds to the vectorized dimensions of each output.
   */
  virtual std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes);

  /** Print the primitive. */
  virtual void print(std::ostream& os) = 0;

  /** Equivalence check defaults to false unless overridden by the primitive */
  virtual bool is_equivalent(const Primitive& other) const {
    return false;
  }

  /** Get the output shapes of the primitive. This is not required to be
   * implemented by derived classes, in which case it will throw. */
  virtual std::vector<Shape> output_shapes(const std::vector<array>& inputs);

  virtual ~Primitive() = default;
  Primitive(const Primitive& other) = delete;
  Primitive(Primitive&& other) = delete;
  Primitive& operator=(const Primitive& other) = delete;
  Primitive& operator=(Primitive&& other) = delete;

 private:
  // Every primitive stores the stream it should run in
  Stream stream_;
};

class UnaryPrimitive : public Primitive {
  /**
   * An abstract base class for a primitive with a single output.
   */
 public:
  explicit UnaryPrimitive(Stream stream) : Primitive(stream) {}

  virtual void eval_cpu(const std::vector<array>& inputs, array& output) = 0;
  virtual void eval_gpu(const std::vector<array>& inputs, array& output) = 0;

  inline void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    eval_cpu(inputs, outputs[0]);
  }
  inline void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    eval_gpu(inputs, outputs[0]);
  }

  virtual ~UnaryPrimitive() = default;
  UnaryPrimitive(const UnaryPrimitive& other) = delete;
  UnaryPrimitive(UnaryPrimitive&& other) = delete;
  UnaryPrimitive& operator=(const UnaryPrimitive& other) = delete;
  UnaryPrimitive& operator=(UnaryPrimitive&& other) = delete;
};

class Abs : public UnaryPrimitive {
 public:
  explicit Abs(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Abs)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Add : public UnaryPrimitive {
 public:
  explicit Add(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Add)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class AddMM : public UnaryPrimitive {
 public:
  explicit AddMM(Stream stream, float alpha, float beta)
      : UnaryPrimitive(stream), alpha_(alpha), beta_(beta) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_VMAP()
  DEFINE_PRINT(AddMM)

  bool is_equivalent(const Primitive& other) const override;
  std::pair<float, float> state() const {
    return {alpha_, beta_};
  };

 private:
  const float alpha_;
  const float beta_;
};

class Arange : public UnaryPrimitive {
 public:
  explicit Arange(Stream stream, double start, double stop, double step)
      : UnaryPrimitive(stream), start_(start), stop_(stop), step_(step) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Arange)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  std::tuple<double, double, double> state() const {
    return {start_, stop_, step_};
  };

 private:
  double start_;
  double stop_;
  double step_;
};

class ArcCos : public UnaryPrimitive {
 public:
  explicit ArcCos(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcCos)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcCosh : public UnaryPrimitive {
 public:
  explicit ArcCosh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcCosh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcSin : public UnaryPrimitive {
 public:
  explicit ArcSin(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcSin)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcSinh : public UnaryPrimitive {
 public:
  explicit ArcSinh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcSinh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcTan : public UnaryPrimitive {
 public:
  explicit ArcTan(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcTan)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcTan2 : public UnaryPrimitive {
 public:
  explicit ArcTan2(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcTan2)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArcTanh : public UnaryPrimitive {
 public:
  explicit ArcTanh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArcTanh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ArgPartition : public UnaryPrimitive {
 public:
  explicit ArgPartition(Stream stream, int kth, int axis)
      : UnaryPrimitive(stream), kth_(kth), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArgPartition)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  std::pair<int, int> state() const {
    return {kth_, axis_};
  };

 private:
  int kth_;
  int axis_;
};

class ArgReduce : public UnaryPrimitive {
 public:
  enum ReduceType {
    ArgMin,
    ArgMax,
  };

  explicit ArgReduce(Stream stream, ReduceType reduce_type, int axis)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ArgReduce)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  std::pair<ReduceType, int> state() const {
    return {reduce_type_, axis_};
  };

 private:
  ReduceType reduce_type_;
  int axis_;
};

class ArgSort : public UnaryPrimitive {
 public:
  explicit ArgSort(Stream stream, int axis)
      : UnaryPrimitive(stream), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(ArgSort)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  int state() const {
    return axis_;
  };

 private:
  int axis_;
};

class AsType : public UnaryPrimitive {
 public:
  explicit AsType(Stream stream, Dtype dtype)
      : UnaryPrimitive(stream), dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(AsType)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  Dtype state() const {
    return dtype_;
  };

 private:
  Dtype dtype_;
};

class AsStrided : public UnaryPrimitive {
 public:
  explicit AsStrided(Stream stream, Shape shape, Strides strides, size_t offset)
      : UnaryPrimitive(stream),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        offset_(offset) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(AsStrided)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(shape_, strides_, offset_);
  }

 private:
  Shape shape_;
  Strides strides_;
  size_t offset_;

  void eval(const std::vector<array>& inputs, array& out);
};

class BitwiseBinary : public UnaryPrimitive {
 public:
  enum Op { And, Or, Xor, LeftShift, RightShift };

  explicit BitwiseBinary(Stream stream, Op op)
      : UnaryPrimitive(stream), op_(op) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  bool is_equivalent(const Primitive& other) const override;
  void print(std::ostream& os) override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return op_;
  }

 private:
  Op op_;
};

class BitwiseInvert : public UnaryPrimitive {
 public:
  explicit BitwiseInvert(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(BitwiseInvert)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class BlockMaskedMM : public UnaryPrimitive {
 public:
  explicit BlockMaskedMM(Stream stream, int block_size)
      : UnaryPrimitive(stream), block_size_(block_size) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(BlockMaskedMM)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return block_size_;
  }

 private:
  int block_size_;
};

class GatherMM : public UnaryPrimitive {
 public:
  explicit GatherMM(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(GatherMM)
  DEFINE_DEFAULT_IS_EQUIVALENT()
};

class BroadcastAxes : public UnaryPrimitive {
 public:
  explicit BroadcastAxes(Stream stream, std::vector<int> ignore_axes = {})
      : UnaryPrimitive(stream), ignore_axes_(std::move(ignore_axes)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(BroadcastAxes)
  bool is_equivalent(const Primitive& other) const override;
  static Shape output_shape(
      const std::vector<array>& inputs,
      const std::vector<int>& ignore_axes);
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return ignore_axes_;
  }

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::vector<int> ignore_axes_;
};

class Broadcast : public UnaryPrimitive {
 public:
  explicit Broadcast(Stream stream, const Shape& shape)
      : UnaryPrimitive(stream), shape_(shape) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Broadcast)
  static Shape output_shape(const std::vector<array>& inputs);
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  bool is_equivalent(const Primitive& other) const override;
  std::vector<int> state() const {
    return shape_;
  };

 private:
  Shape shape_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Ceil : public UnaryPrimitive {
 public:
  explicit Ceil(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Ceil)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Compiled : public Primitive {
 public:
  /*
   * The inputs, outputs and tape are either tracers or constants.
   * - The tape should not contain the inputs, but it should contain the
   *   outputs.
   * - The tape should also have only one array per primitive for multi-output
   *   primitives.
   * - The constant_ids contains ids of arrays in the input list that are safe
   *   to treat as scalar constants.
   */
  explicit Compiled(
      Stream stream,
      std::vector<array> inputs,
      std::vector<array> outputs,
      std::vector<array> tape,
      std::unordered_set<uintptr_t> constant_ids);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  void print(std::ostream& os) override;
  bool is_equivalent(const Primitive& other) const override;

  std::string lib_name() const {
    return kernel_lib_;
  }

 private:
  const std::vector<array> inputs_;
  const std::vector<array> outputs_;
  const std::vector<array> tape_;
  const std::unordered_set<uintptr_t> constant_ids_;

  std::string kernel_lib_;
};

class Concatenate : public UnaryPrimitive {
 public:
  explicit Concatenate(Stream stream, int axis)
      : UnaryPrimitive(stream), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Concatenate)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return axis_;
  }

 private:
  int axis_;
};

class Conjugate : public UnaryPrimitive {
 public:
  explicit Conjugate(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(Conjugate)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Contiguous : public UnaryPrimitive {
 public:
  explicit Contiguous(Stream stream, bool allow_col_major)
      : UnaryPrimitive(stream), allow_col_major_(allow_col_major) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Contiguous)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive& other) const override;

 private:
  bool allow_col_major_;
};

class Convolution : public UnaryPrimitive {
 public:
  explicit Convolution(
      Stream stream,
      const std::vector<int>& kernel_strides,
      const std::vector<int>& padding,
      const std::vector<int>& kernel_dilation,
      const std::vector<int>& input_dilation,
      const int groups = 1,
      const bool flip = false)
      : UnaryPrimitive(stream),
        padding_(padding),
        kernel_strides_(kernel_strides),
        kernel_dilation_(kernel_dilation),
        input_dilation_(input_dilation),
        groups_(groups),
        flip_(flip) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(Convolution)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        groups_,
        flip_);
  }

 private:
  std::vector<int> padding_;
  std::vector<int> kernel_strides_;
  std::vector<int> kernel_dilation_;
  std::vector<int> input_dilation_;
  int groups_;
  bool flip_;
};

class Copy : public UnaryPrimitive {
 public:
  explicit Copy(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Copy)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Cos : public UnaryPrimitive {
 public:
  explicit Cos(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Cos)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Cosh : public UnaryPrimitive {
 public:
  explicit Cosh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Cosh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class CustomTransforms : public Primitive {
 public:
  explicit CustomTransforms(
      Stream stream,
      int num_outputs,
      std::function<std::vector<array>(
          const std::vector<array>&,
          const std::vector<array>&,
          const std::vector<array>&)> vjp,
      std::function<std::vector<array>(
          const std::vector<array>&,
          const std::vector<array>&,
          const std::vector<int>&)> jvp,
      std::function<std::pair<std::vector<array>, std::vector<int>>(
          const std::vector<array>&,
          const std::vector<int>&)> vmap)
      : Primitive(stream),
        num_outputs_(num_outputs),
        vjp_fun_(std::move(vjp)),
        jvp_fun_(std::move(jvp)),
        vmap_fun_(std::move(vmap)) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_GRADS();
  DEFINE_VMAP();
  DEFINE_PRINT(CustomTransforms);

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

  int num_outputs_;

  std::function<std::vector<array>(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<array>&)>
      vjp_fun_;
  std::function<std::vector<array>(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&)>
      jvp_fun_;
  std::function<std::pair<std::vector<array>, std::vector<int>>(
      const std::vector<array>&,
      const std::vector<int>&)>
      vmap_fun_;
};

class Depends : public Primitive {
 public:
  explicit Depends(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotan,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(Depends);

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

class Divide : public UnaryPrimitive {
 public:
  explicit Divide(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Divide)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class DivMod : public Primitive {
 public:
  explicit DivMod(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(DivMod)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return std::vector{inputs[0].shape(), inputs[0].shape()};
  }
};

class Select : public UnaryPrimitive {
 public:
  explicit Select(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Select)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Remainder : public UnaryPrimitive {
 public:
  explicit Remainder(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Remainder)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Equal : public UnaryPrimitive {
 public:
  explicit Equal(Stream stream, bool equal_nan = false)
      : UnaryPrimitive(stream), equal_nan_(equal_nan) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

  void print(std::ostream& os) override {
    if (equal_nan_) {
      os << "NaNEqual";
    } else {
      os << "Equal";
    }
  }
  auto state() const {
    return equal_nan_;
  };

 private:
  bool equal_nan_;
};

class Erf : public UnaryPrimitive {
 public:
  explicit Erf(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Erf)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ErfInv : public UnaryPrimitive {
 public:
  explicit ErfInv(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ErfInv)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Exp : public UnaryPrimitive {
 public:
  explicit Exp(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Exp)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Expm1 : public UnaryPrimitive {
 public:
  explicit Expm1(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Expm1)
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class ExpandDims : public UnaryPrimitive {
 public:
  explicit ExpandDims(Stream stream, std::vector<int> axes)
      : UnaryPrimitive(stream), axes_(std::move(axes)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(ExpandDims)

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  bool is_equivalent(const Primitive& other) const override;

  static Shape output_shape(const array& input, const std::vector<int>& axes);
  auto state() const {
    return axes_;
  }

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::vector<int> axes_;
};

class FFT : public UnaryPrimitive {
 public:
  explicit FFT(
      Stream stream,
      const std::vector<size_t>& axes,
      bool inverse,
      bool real)
      : UnaryPrimitive(stream), axes_(axes), inverse_(inverse), real_(real) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(FFT)

  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(axes_, inverse_, real_);
  }

 private:
  std::vector<size_t> axes_;
  bool inverse_;
  bool real_;
};

class Flatten : public UnaryPrimitive {
 public:
  explicit Flatten(Stream stream, int start_axis, int end_axis)
      : UnaryPrimitive(stream), start_axis_(start_axis), end_axis_(end_axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Flatten)
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  bool is_equivalent(const Primitive& other) const override;

  static Shape output_shape(const array& input, int start_axis, int end_axis);
  auto state() const {
    return std::make_pair(start_axis_, end_axis_);
  }

 private:
  int start_axis_;
  int end_axis_;
  void eval(const std::vector<array>& inputs, array& out);
};

class Floor : public UnaryPrimitive {
 public:
  explicit Floor(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Floor)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Full : public UnaryPrimitive {
 public:
  explicit Full(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Full)
  DEFINE_DEFAULT_IS_EQUIVALENT()
};

class Gather : public UnaryPrimitive {
 public:
  explicit Gather(Stream stream, std::vector<int> axes, Shape slice_sizes)
      : UnaryPrimitive(stream),
        axes_(std::move(axes)),
        slice_sizes_(std::move(slice_sizes)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Gather)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  std::pair<std::vector<int>, std::vector<int>> state() const {
    return {axes_, slice_sizes_};
  }

 private:
  std::vector<int> axes_;
  Shape slice_sizes_;
};

class GatherAxis : public UnaryPrimitive {
 public:
  explicit GatherAxis(Stream stream, int axis)
      : UnaryPrimitive(stream), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(GatherAxis)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return axis_;
  }

 private:
  int axis_;
};

class Greater : public UnaryPrimitive {
 public:
  explicit Greater(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Greater)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class GreaterEqual : public UnaryPrimitive {
 public:
  explicit GreaterEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(GreaterEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Hadamard : public UnaryPrimitive {
 public:
  explicit Hadamard(Stream stream, float scale)
      : UnaryPrimitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Hadamard)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return scale_;
  }

 private:
  float scale_;
};

class Imag : public UnaryPrimitive {
 public:
  explicit Imag(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Imag)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Less : public UnaryPrimitive {
 public:
  explicit Less(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Less)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LessEqual : public UnaryPrimitive {
 public:
  explicit LessEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LessEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Load : public UnaryPrimitive {
 public:
  explicit Load(
      Stream stream,
      std::shared_ptr<io::Reader> reader,
      size_t offset,
      bool swap_endianness = false)
      : UnaryPrimitive(stream),
        reader_(std::move(reader)),
        offset_(offset),
        swap_endianness_(swap_endianness) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Load)

 private:
  std::shared_ptr<io::Reader> reader_;
  size_t offset_;
  bool swap_endianness_;
};

class Log : public UnaryPrimitive {
 public:
  enum Base { two, ten, e };

  explicit Log(Stream stream, Base base)
      : UnaryPrimitive(stream), base_(base) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

  Base state() const {
    return base_;
  };

  void print(std::ostream& os) override {
    switch (base_) {
      case e:
        os << "Log";
        break;
      case two:
        os << "Log2";
        break;
      case ten:
        os << "Log10";
        break;
    }
  }

 private:
  Base base_;
};

class Log1p : public UnaryPrimitive {
 public:
  explicit Log1p(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Log1p)
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LogicalNot : public UnaryPrimitive {
 public:
  explicit LogicalNot(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LogicalNot)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LogicalAnd : public UnaryPrimitive {
 public:
  explicit LogicalAnd(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LogicalAnd)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LogicalOr : public UnaryPrimitive {
 public:
  explicit LogicalOr(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LogicalOr)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LogAddExp : public UnaryPrimitive {
 public:
  explicit LogAddExp(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LogAddExp)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class LogSumExp : public UnaryPrimitive {
 public:
  explicit LogSumExp(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(LogSumExp)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
};

class Matmul : public UnaryPrimitive {
 public:
  explicit Matmul(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_VMAP()
  DEFINE_PRINT(Matmul)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
};

class Maximum : public UnaryPrimitive {
 public:
  explicit Maximum(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Maximum)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Minimum : public UnaryPrimitive {
 public:
  explicit Minimum(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Minimum)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Multiply : public UnaryPrimitive {
 public:
  explicit Multiply(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Multiply)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Negative : public UnaryPrimitive {
 public:
  explicit Negative(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Negative)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class NotEqual : public UnaryPrimitive {
 public:
  explicit NotEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(NotEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class NumberOfElements : public UnaryPrimitive {
 public:
  explicit NumberOfElements(
      Stream stream,
      std::vector<int> axes,
      bool inverted,
      Dtype dtype)
      : UnaryPrimitive(stream),
        axes_(std::move(axes)),
        inverted_(inverted),
        dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(NumberOfElements)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {{}};
  }
  std::tuple<std::vector<int>, bool, Dtype> state() const {
    return {axes_, inverted_, dtype_};
  }

 private:
  std::vector<int> axes_;
  bool inverted_;
  Dtype dtype_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Pad : public UnaryPrimitive {
 public:
  explicit Pad(
      Stream stream,
      const std::vector<int>& axes,
      const Shape& low_pad_size,
      const Shape& high_pad_size)
      : UnaryPrimitive(stream),
        axes_(axes),
        low_pad_size_(low_pad_size),
        high_pad_size_(high_pad_size) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Pad)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(axes_, low_pad_size_, high_pad_size_);
  }

 private:
  std::vector<int> axes_;
  Shape low_pad_size_;
  Shape high_pad_size_;
};

class Partition : public UnaryPrimitive {
 public:
  explicit Partition(Stream stream, int kth, int axis)
      : UnaryPrimitive(stream), kth_(kth), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Partition)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(kth_, axis_);
  };

 private:
  int kth_;
  int axis_;
};

class Power : public UnaryPrimitive {
 public:
  explicit Power(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Power)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class QuantizedMatmul : public UnaryPrimitive {
 public:
  explicit QuantizedMatmul(
      Stream stream,
      int group_size,
      int bits,
      bool transpose)
      : UnaryPrimitive(stream),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(QuantizedMatmul)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(group_size_, bits_, transpose_);
  }

 private:
  int group_size_;
  int bits_;
  bool transpose_;
};

class GatherQMM : public UnaryPrimitive {
 public:
  explicit GatherQMM(Stream stream, int group_size, int bits, bool transpose)
      : UnaryPrimitive(stream),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(GatherQMM)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(group_size_, bits_, transpose_);
  }

 private:
  int group_size_;
  int bits_;
  bool transpose_;
};

class RandomBits : public UnaryPrimitive {
 public:
  explicit RandomBits(Stream stream, const Shape& shape, int width)
      : UnaryPrimitive(stream), shape_(shape), width_(width) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(RandomBits)
  bool is_equivalent(const Primitive& other) const override;
  std::pair<std::vector<int>, int> state() const {
    return {shape_, width_};
  };

 private:
  Shape shape_;
  int width_;
};

class Real : public UnaryPrimitive {
 public:
  explicit Real(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Real)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Reshape : public UnaryPrimitive {
 public:
  explicit Reshape(Stream stream, const Shape& shape)
      : UnaryPrimitive(stream), shape_(shape) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Reshape)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<int> state() const {
    return shape_;
  };
  static Shape output_shape(const array& input, Shape shape);
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

 private:
  Shape shape_;
};

class Reduce : public UnaryPrimitive {
 public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };

  explicit Reduce(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

  void print(std::ostream& os) override {
    switch (reduce_type_) {
      case And:
        os << "And";
        break;
      case Or:
        os << "Or";
        break;
      case Sum:
        os << "Sum";
        break;
      case Prod:
        os << "Prod";
        break;
      case Min:
        os << "Min";
        break;
      case Max:
        os << "Max";
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;
  std::pair<ReduceType, std::vector<int>> state() const {
    return {reduce_type_, axes_};
  };

 private:
  ReduceType reduce_type_;
  std::vector<int> axes_;
};

class Round : public UnaryPrimitive {
 public:
  explicit Round(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Round)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Scan : public UnaryPrimitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod };

  explicit Scan(
      Stream stream,
      ReduceType reduce_type,
      int axis,
      bool reverse,
      bool inclusive)
      : UnaryPrimitive(stream),
        reduce_type_(reduce_type),
        axis_(axis),
        reverse_(reverse),
        inclusive_(inclusive) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS();

  void print(std::ostream& os) override {
    os << "Cum";
    switch (reduce_type_) {
      case Sum:
        os << "Sum";
        break;
      case Prod:
        os << "Prod";
        break;
      case Min:
        os << "Min";
        break;
      case Max:
        os << "Max";
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(reduce_type_, axis_, reverse_, inclusive_);
  }

 private:
  ReduceType reduce_type_;
  int axis_;
  bool reverse_;
  bool inclusive_;
};

class Scatter : public UnaryPrimitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod, None };

  explicit Scatter(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP();
  DEFINE_GRADS();

  void print(std::ostream& os) override {
    os << "Scatter";
    switch (reduce_type_) {
      case Sum:
        os << " Sum";
        break;
      case Prod:
        os << " Prod";
        break;
      case Min:
        os << " Min";
        break;
      case Max:
        os << " Max";
        break;
      case None:
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;
  std::pair<ReduceType, std::vector<int>> state() const {
    return {reduce_type_, axes_};
  };

 private:
  ReduceType reduce_type_;
  std::vector<int> axes_;
};

class ScatterAxis : public UnaryPrimitive {
 public:
  enum ReduceType { Sum, None };

  explicit ScatterAxis(Stream stream, ReduceType reduce_type, int axis)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()

  void print(std::ostream& os) override {
    os << "ScatterAxis";
    switch (reduce_type_) {
      case Sum:
        os << " Sum";
        break;
      case None:
        break;
    }
  }

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  std::pair<ReduceType, int> state() const {
    return {reduce_type_, axis_};
  }

 private:
  ReduceType reduce_type_;
  int axis_;
};

class Sigmoid : public UnaryPrimitive {
 public:
  explicit Sigmoid(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Sigmoid)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Sign : public UnaryPrimitive {
 public:
  explicit Sign(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Sign)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Sin : public UnaryPrimitive {
 public:
  explicit Sin(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Sin)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Sinh : public UnaryPrimitive {
 public:
  explicit Sinh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Sinh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Slice : public UnaryPrimitive {
 public:
  explicit Slice(
      Stream stream,
      const Shape& start_indices,
      const Shape& end_indices,
      const Shape& strides)
      : UnaryPrimitive(stream),
        start_indices_(start_indices),
        end_indices_(end_indices),
        strides_(strides) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Slice)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(start_indices_, end_indices_, strides_);
  }

 private:
  Shape start_indices_;
  Shape end_indices_;
  Shape strides_;
};

class SliceUpdate : public UnaryPrimitive {
 public:
  explicit SliceUpdate(
      Stream stream,
      const Shape& start_indices,
      const Shape& end_indices,
      const Shape& strides)
      : UnaryPrimitive(stream),
        start_indices_(start_indices),
        end_indices_(end_indices),
        strides_(strides) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(SliceUpdate)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(start_indices_, end_indices_, strides_);
  }

 private:
  Shape start_indices_;
  Shape end_indices_;
  Shape strides_;
};

class DynamicSlice : public UnaryPrimitive {
 public:
  explicit DynamicSlice(Stream stream, std::vector<int> axes, Shape slice_size)
      : UnaryPrimitive(stream),
        axes_(std::move(axes)),
        slice_size_(std::move(slice_size)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(DynamicSlice)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_pair(axes_, slice_size_);
  }

 private:
  std::vector<int> axes_;
  Shape slice_size_;
};

class DynamicSliceUpdate : public UnaryPrimitive {
 public:
  explicit DynamicSliceUpdate(Stream stream, std::vector<int> axes)
      : UnaryPrimitive(stream), axes_(std::move(axes)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(DynamicSliceUpdate)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return axes_;
  }

 private:
  std::vector<int> axes_;
};

class Softmax : public UnaryPrimitive {
 public:
  explicit Softmax(Stream stream, bool precise)
      : UnaryPrimitive(stream), precise_(precise) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Softmax)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return precise_;
  };

 private:
  bool precise_;
};

class Sort : public UnaryPrimitive {
 public:
  explicit Sort(Stream stream, int axis)
      : UnaryPrimitive(stream), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Sort)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return axis_;
  }

 private:
  int axis_;
};

class Split : public Primitive {
 public:
  explicit Split(Stream stream, const Shape& indices, int axis)
      : Primitive(stream), indices_(indices), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Split)
  bool is_equivalent(const Primitive& other) const override;
  std::pair<std::vector<int>, int> state() const {
    return {indices_, axis_};
  };

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

  Shape indices_;
  int axis_;
};

class Square : public UnaryPrimitive {
 public:
  explicit Square(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Square)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Sqrt : public UnaryPrimitive {
 public:
  explicit Sqrt(Stream stream, bool recip = false)
      : UnaryPrimitive(stream), recip_(recip) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return recip_;
  }

  void print(std::ostream& os) override {
    if (recip_) {
      os << "Rsqrt";
    } else {
      os << "Sqrt";
    }
  }

 private:
  bool recip_;
};

class StopGradient : public UnaryPrimitive {
 public:
  explicit StopGradient(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_PRINT(StopGradient)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Subtract : public UnaryPrimitive {
 public:
  explicit Subtract(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Subtract)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Squeeze : public UnaryPrimitive {
 public:
  explicit Squeeze(Stream stream, std::vector<int> axes)
      : UnaryPrimitive(stream), axes_(std::move(axes)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Squeeze)

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  bool is_equivalent(const Primitive& other) const override;

  static Shape output_shape(const array& input, const std::vector<int>& axes);
  auto state() const {
    return axes_;
  };

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::vector<int> axes_;
};

class Tan : public UnaryPrimitive {
 public:
  explicit Tan(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Tan)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Tanh : public UnaryPrimitive {
 public:
  explicit Tanh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Tanh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

class Unflatten : public UnaryPrimitive {
 public:
  explicit Unflatten(Stream stream, int axis, Shape shape)
      : UnaryPrimitive(stream), axis_(axis), shape_(std::move(shape)) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Unflatten)
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  bool is_equivalent(const Primitive& other) const override;

  static Shape output_shape(const array& input, int axis, const Shape& shape);
  auto state() const {
    return std::make_pair(axis_, shape_);
  }

 private:
  int axis_;
  Shape shape_;
  void eval(const std::vector<array>& inputs, array& out);
};

class View : public UnaryPrimitive {
 public:
  explicit View(Stream stream, Dtype dtype)
      : UnaryPrimitive(stream), dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  void print(std::ostream& os) override;
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return dtype_;
  }

 private:
  Dtype dtype_;
};

class Transpose : public UnaryPrimitive {
 public:
  explicit Transpose(Stream stream, const std::vector<int>& axes)
      : UnaryPrimitive(stream), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_PRINT(Transpose)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  std::vector<int> state() const {
    return axes_;
  };

 private:
  std::vector<int> axes_;

  void eval(const std::vector<array>& inputs, array& out);
};

/* QR Factorization primitive. */
class QRF : public Primitive {
 public:
  explicit QRF(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(QRF)
};

/* SVD primitive. */
class SVD : public Primitive {
 public:
  explicit SVD(Stream stream, bool compute_uv)
      : Primitive(stream), compute_uv_(compute_uv) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_PRINT(SVD)
  auto state() const {
    return compute_uv_;
  }

 private:
  bool compute_uv_;
};

/* Matrix inversion primitive. */
class Inverse : public UnaryPrimitive {
 public:
  explicit Inverse(Stream stream, bool tri, bool upper)
      : UnaryPrimitive(stream), tri_(tri), upper_(upper) {}

  void eval_cpu(const std::vector<array>& inputs, array& output) override;
  void eval_gpu(const std::vector<array>& inputs, array& output) override;

  DEFINE_VMAP()
  DEFINE_PRINT(Inverse)
  auto state() const {
    return std::make_pair(tri_, upper_);
  }

 private:
  bool tri_;
  bool upper_;
};

class Cholesky : public UnaryPrimitive {
 public:
  explicit Cholesky(Stream stream, bool upper)
      : UnaryPrimitive(stream), upper_(upper) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;
  auto state() const {
    return upper_;
  }

  DEFINE_VMAP()
  DEFINE_PRINT(Cholesky)

 private:
  bool upper_;
};

class Eigh : public Primitive {
 public:
  explicit Eigh(Stream stream, std::string uplo, bool compute_eigenvectors)
      : Primitive(stream),
        uplo_(std::move(uplo)),
        compute_eigenvectors_(compute_eigenvectors) {}
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_VMAP()
  DEFINE_PRINT(Eigh)

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(uplo_, compute_eigenvectors_);
  }

 private:
  std::string uplo_;
  bool compute_eigenvectors_;
};

/* LU Factorization primitive. */
class LUF : public Primitive {
 public:
  explicit LUF(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(LUF)
};

} // namespace mlx::core
