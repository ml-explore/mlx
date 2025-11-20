// Copyright © 2024 Apple Inc.

#include <optional>
#include <variant>

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
      : Primitive(stream), fallback_(std::move(fallback)) {}

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

 protected:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
};

class RMSNorm : public Custom {
 public:
  RMSNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(RMSNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class RMSNormVJP : public Custom {
 public:
  RMSNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(RMSNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNorm : public Custom {
 public:
  LayerNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(LayerNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNormVJP : public Custom {
 public:
  LayerNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(LayerNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
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
      bool forward)
      : Custom(stream, std::move(fallback)),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        forward_(forward) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(RoPE)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr, dims_, traditional_, base_, scale_, forward_);
  }

 private:
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  bool forward_;
};

class ScaledDotProductAttention : public Custom {
 public:
  ScaledDotProductAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      bool do_causal,
      bool has_sinks,
      bool output_logsumexp)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        do_causal_(do_causal),
        has_sinks_(has_sinks),
        output_logsumexp_(output_logsumexp) {}

  static bool use_fallback(
      const array& q,
      const array& k,
      const array& v,
      bool has_mask,
      bool has_arr_mask,
      bool do_causal,
      bool is_training,
      bool output_logsumexp,
      Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  bool is_equivalent(const Primitive& other) const override;

  DEFINE_NAME(ScaledDotProductAttention);
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr, scale_, do_causal_, has_sinks_, output_logsumexp_);
  }

 private:
  float scale_;
  bool do_causal_;
  bool has_sinks_;
  bool output_logsumexp_;
};

class ScaledDotProductAttentionVJP : public Custom {
 public:
  ScaledDotProductAttentionVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      bool do_causal,
      bool has_sinks)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        do_causal_(do_causal),
        has_sinks_(has_sinks) {}

  static bool use_fallback(const array& q, Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(ScaledDotProductAttentionVJP);
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(nullptr, scale_, do_causal_, has_sinks_);
  }

 private:
  float scale_;
  bool do_causal_;
  bool has_sinks_;
};

class ConvertFP8 : public Primitive {
 public:
  explicit ConvertFP8(Stream stream, bool to_fp8)
      : Primitive(stream), to_fp8_(to_fp8) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  const char* name() const override {
    if (to_fp8_) {
      return "ToFP8";
    } else {
      return "FromFP8";
    }
  }
  bool state() const {
    return to_fp8_;
  };

  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE();

 private:
  bool to_fp8_;
};

class Quantize : public Custom {
 public:
  explicit Quantize(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int bits,
      QuantizationMode mode,
      bool dequantize)
      : Custom(stream, std::move(fallback)),
        group_size_(group_size),
        bits_(bits),
        mode_(mode),
        dequantize_(dequantize) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(Quantize);

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(nullptr, group_size_, bits_, mode_, dequantize_);
  }

 private:
  int group_size_;
  int bits_;
  QuantizationMode mode_;
  bool dequantize_;
};

using ScalarArg = std::variant<bool, int, float>;

class CustomKernel : public Primitive {
 public:
  CustomKernel(
      Stream stream,
      std::string name,
      std::string source,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      std::vector<std::tuple<bool, bool, bool>> shape_infos,
      bool ensure_row_contiguous,
      std::optional<float> init_value,
      std::vector<ScalarArg> scalar_arguments,
      bool is_precompiled,
      int shared_memory)
      : Primitive(stream),
        name_(std::move(name)),
        source_(std::move(source)),
        grid_(grid),
        threadgroup_(threadgroup),
        shape_infos_(std::move(shape_infos)),
        ensure_row_contiguous_(ensure_row_contiguous),
        init_value_(init_value),
        scalar_arguments_(std::move(scalar_arguments)),
        is_precompiled_(is_precompiled),
        shared_memory_(shared_memory) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("Custom kernels only run on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(CustomKernel);
  auto state() const {
    return std::make_tuple(
        name_,
        source_,
        grid_,
        threadgroup_,
        shape_infos_,
        ensure_row_contiguous_,
        init_value_,
        scalar_arguments_,
        is_precompiled_,
        shared_memory_);
  }

 private:
  std::string name_;
  std::string source_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> threadgroup_;
  std::vector<std::tuple<bool, bool, bool>> shape_infos_;
  bool ensure_row_contiguous_;
  std::optional<float> init_value_;
  std::vector<ScalarArg> scalar_arguments_;
  bool is_precompiled_;
  int shared_memory_;
};

} // namespace mlx::core::fast
