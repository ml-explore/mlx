// Copyright © 2024 Apple Inc.

#include "mlx/primitives.h"

namespace mlx::core::fast {

// Custom primitive accepts a fallback function which it uses for
// transformations. Transformations are virtual so that derived classes may
// override the default behavior.
class Custom : public Primitive {
 public:
  explicit Custom(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback)
      : Primitive(stream), fallback_(fallback){};

  virtual std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
};

class RMSNorm : public Custom {
 public:
  RMSNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(RMSNorm)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  float eps_;
};

class RMSNormVJP : public Custom {
 public:
  RMSNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(RMSNormVJP)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  float eps_;
};

class LayerNorm : public Custom {
 public:
  LayerNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(LayerNorm)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  float eps_;
};

class LayerNormVJP : public Custom {
 public:
  LayerNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(LayerNormVJP)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  float eps_;
};

class RoPE : public Custom {
 public:
  RoPE(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int dims,
      bool traditional,
      float base,
      float scale,
      int offset,
      bool forward)
      : Custom(stream, fallback),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        offset_(offset),
        forward_(forward){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(RoPE)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  int offset_;
  bool forward_;
};

class ScaledDotProductAttention : public Custom {
 public:
  explicit ScaledDotProductAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      const float scale,
      const bool needs_mask)
      : Custom(stream, fallback), scale_(scale), needs_mask_(needs_mask){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  };

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    eval_gpu(inputs, outputs[0]);
  };

  void eval_gpu(const std::vector<array>& inputs, array& out);
  bool is_equivalent(const Primitive& other) const override;

  DEFINE_PRINT(ScaledDotProductAttention)

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  float scale_;
  bool needs_mask_;
};

} // namespace mlx::core::fast
