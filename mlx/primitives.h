// Copyright © 2023 Apple Inc.

#pragma once

#include "array.h"
#include "device.h"
#include "load.h"
#include "stream.h"

#define DEFINE_GRADS()                           \
  array jvp(                                     \
      const std::vector<array>& primals,         \
      const std::vector<array>& tangents,        \
      const std::vector<int>& argnums) override; \
                                                 \
  std::vector<array> vjp(                        \
      const std::vector<array>& primals,         \
      const array& cotan,                        \
      const std::vector<int>& argnums) override;

#define DEFINE_PRINT(PRIMITIVE)           \
  void print(std::ostream& os) override { \
    os << #PRIMITIVE;                     \
  }

#define DEFINE_DEFAULT_IS_EQUIVALENT()                        \
  bool is_equivalent(const Primitive& other) const override { \
    return true;                                              \
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
   * the CPU/GPU for the given inputs and populate the output array.
   *
   * To avoid unecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  virtual void eval_cpu(const std::vector<array>& inputs, array& out) = 0;
  virtual void eval_gpu(const std::vector<array>& inputs, array& out) = 0;

  /**
   * The Jacobian-vector product.
   */
  virtual array jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums);

  /**
   * The vector-Jacobian product.
   */
  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const array& cotan,
      const std::vector<int>& argnums);

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  virtual std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes);

  /** Print the primitive. */
  virtual void print(std::ostream& os) = 0;

  /** Equivalence check defaults to false unless overriden by the primitive */
  virtual bool is_equivalent(const Primitive& other) const {
    return false;
  }

  virtual ~Primitive() = default;
  Primitive(const Primitive& other) = delete;
  Primitive(Primitive&& other) = delete;
  Primitive& operator=(const Primitive& other) = delete;
  Primitive& operator=(Primitive&& other) = delete;

 private:
  // Every primitive stores the stream it should run in
  Stream stream_;
};

class Abs : public Primitive {
 public:
  explicit Abs(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Abs)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Add : public Primitive {
 public:
  explicit Add(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Add)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Arange : public Primitive {
 public:
  explicit Arange(Stream stream, double start, double stop, double step)
      : Primitive(stream), start_(start), stop_(stop), step_(step){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Arange)
  bool is_equivalent(const Primitive& other) const override;

 private:
  double start_;
  double stop_;
  double step_;

  void eval(const std::vector<array>& inputs, array& out);
};

class ArcCos : public Primitive {
 public:
  explicit ArcCos(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcCos)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArcCosh : public Primitive {
 public:
  explicit ArcCosh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcCosh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArcSin : public Primitive {
 public:
  explicit ArcSin(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcSin)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArcSinh : public Primitive {
 public:
  explicit ArcSinh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcSinh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArcTan : public Primitive {
 public:
  explicit ArcTan(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcTan)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArcTanh : public Primitive {
 public:
  explicit ArcTanh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArcTanh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ArgPartition : public Primitive {
 public:
  explicit ArgPartition(Stream stream, int kth, int axis)
      : Primitive(stream), kth_(kth), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(ArgPartition)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int kth_;
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class ArgReduce : public Primitive {
 public:
  enum ReduceType {
    ArgMin,
    ArgMax,
  };

  explicit ArgReduce(Stream stream, ReduceType reduce_type, int axis)
      : Primitive(stream), reduce_type_(reduce_type), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(ArgReduce)
  bool is_equivalent(const Primitive& other) const override;

 private:
  ReduceType reduce_type_;
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class ArgSort : public Primitive {
 public:
  explicit ArgSort(Stream stream, int axis) : Primitive(stream), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(ArgSort)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class AsType : public Primitive {
 public:
  explicit AsType(Stream stream, Dtype dtype)
      : Primitive(stream), dtype_(dtype){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(AsType)
  bool is_equivalent(const Primitive& other) const override;

 private:
  Dtype dtype_;

  void eval(const std::vector<array>& inputs, array& out);
};

class AsStrided : public Primitive {
 public:
  explicit AsStrided(
      Stream stream,
      const std::vector<int>& shape,
      const std::vector<size_t>& strides,
      size_t offset)
      : Primitive(stream), shape_(shape), strides_(strides), offset_(offset){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(AsStrided)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;
  std::vector<size_t> strides_;
  size_t offset_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Broadcast : public Primitive {
 public:
  explicit Broadcast(Stream stream, const std::vector<int>& shape)
      : Primitive(stream), shape_(shape){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Broadcast)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Ceil : public Primitive {
 public:
  explicit Ceil(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Ceil)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Concatenate : public Primitive {
 public:
  explicit Concatenate(Stream stream, int axis)
      : Primitive(stream), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Concatenate)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Convolution : public Primitive {
 public:
  explicit Convolution(
      Stream stream,
      const std::vector<int>& padding,
      const std::vector<int>& kernel_strides,
      const std::vector<int>& kernel_dilation,
      const std::vector<int>& input_dilation)
      : Primitive(stream),
        padding_(padding),
        kernel_strides_(kernel_strides),
        kernel_dilation_(kernel_dilation),
        input_dilation_(input_dilation){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const array& cotan,
      const std::vector<int>& argnums) override;

  DEFINE_PRINT(Convolution)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> padding_;
  std::vector<int> kernel_strides_;
  std::vector<int> kernel_dilation_;
  std::vector<int> input_dilation_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Copy : public Primitive {
 public:
  explicit Copy(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Copy)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Cos : public Primitive {
 public:
  explicit Cos(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Cos)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Cosh : public Primitive {
 public:
  explicit Cosh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Cosh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Divide : public Primitive {
 public:
  explicit Divide(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Divide)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Remainder : public Primitive {
 public:
  explicit Remainder(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Remainder)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Equal : public Primitive {
 public:
  explicit Equal(Stream stream, bool equal_nan = false)
      : Primitive(stream), equal_nan_(equal_nan){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Equal)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
  bool equal_nan_;
};

class Erf : public Primitive {
 public:
  explicit Erf(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Erf)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ErfInv : public Primitive {
 public:
  explicit ErfInv(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ErfInv)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Exp : public Primitive {
 public:
  explicit Exp(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Exp)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class FFT : public Primitive {
 public:
  explicit FFT(
      Stream stream,
      const std::vector<size_t>& axes,
      bool inverse,
      bool real)
      : Primitive(stream), axes_(axes), inverse_(inverse), real_(real){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(FFT)

  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<size_t> axes_;
  bool inverse_;
  bool real_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Floor : public Primitive {
 public:
  explicit Floor(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Floor)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Full : public Primitive {
 public:
  explicit Full(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Full)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Gather : public Primitive {
 public:
  explicit Gather(
      Stream stream,
      const std::vector<int>& axes,
      const std::vector<int>& slice_sizes)
      : Primitive(stream), axes_(axes), slice_sizes_(slice_sizes){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Gather)
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::vector<int> axes_;
  std::vector<int> slice_sizes_;
};

class Greater : public Primitive {
 public:
  explicit Greater(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Greater)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class GreaterEqual : public Primitive {
 public:
  explicit GreaterEqual(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(GreaterEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Less : public Primitive {
 public:
  explicit Less(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Less)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LessEqual : public Primitive {
 public:
  explicit LessEqual(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LessEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Load : public Primitive {
 public:
  explicit Load(
      Stream stream,
      std::shared_ptr<io::Reader> reader,
      size_t offset,
      bool swap_endianness = false)
      : Primitive(stream),
        reader_(reader),
        offset_(offset),
        swap_endianness_(swap_endianness){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Load)

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::shared_ptr<io::Reader> reader_;
  size_t offset_;
  bool swap_endianness_;
};

class Log : public Primitive {
 public:
  enum Base { two, ten, e };

  explicit Log(Stream stream, Base base) : Primitive(stream), base_(base){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Log)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  Base base_;
  void eval(const std::vector<array>& inputs, array& out);
};

class Log1p : public Primitive {
 public:
  explicit Log1p(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Log1p)

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LogicalNot : public Primitive {
 public:
  explicit LogicalNot(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LogicalNot)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LogAddExp : public Primitive {
 public:
  explicit LogAddExp(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LogAddExp)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Matmul : public Primitive {
 public:
  explicit Matmul(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const array& cotan,
      const std::vector<int>& argnums) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(Matmul)
  DEFINE_DEFAULT_IS_EQUIVALENT()
};

class Maximum : public Primitive {
 public:
  explicit Maximum(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Maximum)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Minimum : public Primitive {
 public:
  explicit Minimum(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Minimum)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Multiply : public Primitive {
 public:
  explicit Multiply(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Multiply)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Negative : public Primitive {
 public:
  explicit Negative(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Negative)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class NotEqual : public Primitive {
 public:
  explicit NotEqual(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(NotEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Pad : public Primitive {
 public:
  explicit Pad(
      Stream stream,
      const std::vector<int>& axes,
      const std::vector<int>& low_pad_size,
      const std::vector<int>& high_pad_size)
      : Primitive(stream),
        axes_(axes),
        low_pad_size_(low_pad_size),
        high_pad_size_(high_pad_size){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Pad)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> axes_;
  std::vector<int> low_pad_size_;
  std::vector<int> high_pad_size_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Partition : public Primitive {
 public:
  explicit Partition(Stream stream, int kth, int axis)
      : Primitive(stream), kth_(kth), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Partition)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int kth_;
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Power : public Primitive {
 public:
  explicit Power(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Power)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class QuantizedMatmul : public Primitive {
 public:
  explicit QuantizedMatmul(Stream stream, int group_size, int bits)
      : Primitive(stream), group_size_(group_size), bits_(bits){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(QuantizedMatmul)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int group_size_;
  int bits_;

  void eval(const std::vector<array>& inputs, array& out);
};

class RandomBits : public Primitive {
 public:
  explicit RandomBits(Stream stream, const std::vector<int>& shape, int width)
      : Primitive(stream), shape_(shape), width_(width){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(RandomBits)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;
  int width_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Reshape : public Primitive {
 public:
  explicit Reshape(Stream stream, const std::vector<int>& shape)
      : Primitive(stream), shape_(shape){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Reshape)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Reduce : public Primitive {
 public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };

  explicit Reduce(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : Primitive(stream), reduce_type_(reduce_type), axes_(axes){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const array& cotan,
      const std::vector<int>& argnums) override;

  void print(std::ostream& os) override {
    switch (reduce_type_) {
      case And:
        os << "And";
      case Or:
        os << "And";
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
    os << " Reduce";
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  ReduceType reduce_type_;
  std::vector<int> axes_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Round : public Primitive {
 public:
  explicit Round(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Round)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Scan : public Primitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod };

  explicit Scan(
      Stream stream,
      ReduceType reduce_type,
      int axis,
      bool reverse,
      bool inclusive)
      : Primitive(stream),
        reduce_type_(reduce_type),
        axis_(axis),
        reverse_(reverse),
        inclusive_(inclusive){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

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
    os << " Reduce";
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  ReduceType reduce_type_;
  int axis_;
  bool reverse_;
  bool inclusive_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Scatter : public Primitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod, None };

  explicit Scatter(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : Primitive(stream), reduce_type_(reduce_type), axes_(axes){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Scatter)
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  ReduceType reduce_type_;
  std::vector<int> axes_;
};

class Sigmoid : public Primitive {
 public:
  explicit Sigmoid(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sigmoid)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sign : public Primitive {
 public:
  explicit Sign(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sign)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sin : public Primitive {
 public:
  explicit Sin(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sin)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sinh : public Primitive {
 public:
  explicit Sinh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sinh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Slice : public Primitive {
 public:
  explicit Slice(
      Stream stream,
      const std::vector<int>& start_indices,
      const std::vector<int>& end_indices,
      const std::vector<int>& strides)
      : Primitive(stream),
        start_indices_(start_indices),
        end_indices_(end_indices),
        strides_(strides){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Slice)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> start_indices_;
  std::vector<int> end_indices_;
  std::vector<int> strides_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Softmax : public Primitive {
 public:
  explicit Softmax(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Softmax)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sort : public Primitive {
 public:
  explicit Sort(Stream stream, int axis) : Primitive(stream), axis_(axis){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sort)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Square : public Primitive {
 public:
  explicit Square(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Square)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sqrt : public Primitive {
 public:
  explicit Sqrt(Stream stream, bool recip = false)
      : Primitive(stream), recip_(recip){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sqrt)
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  bool recip_;
};

class StopGradient : public Primitive {
 public:
  explicit StopGradient(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(StopGradient)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Subtract : public Primitive {
 public:
  explicit Subtract(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Subtract)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Tan : public Primitive {
 public:
  explicit Tan(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Tan)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Tanh : public Primitive {
 public:
  explicit Tanh(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Tanh)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Uniform : public Primitive {
 public:
  explicit Uniform(Stream stream) : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_PRINT(Uniform)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Transpose : public Primitive {
 public:
  explicit Transpose(Stream stream, const std::vector<int>& axes)
      : Primitive(stream), axes_(axes){};

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::pair<array, int> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Transpose)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> axes_;

  void eval(const std::vector<array>& inputs, array& out);
};

} // namespace mlx::core
