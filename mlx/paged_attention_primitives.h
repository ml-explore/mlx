// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI in MSVC.
#define _USE_MATH_DEFINES

#include <optional>

#include "mlx/primitives.h"

namespace mlx::core::paged_attention {

class PagedAttention : public UnaryPrimitive {
 public:
  explicit PagedAttention(
      Stream stream,
      bool use_v1,
      int max_context_len,
      int head_size,
      int block_size,
      int num_kv_heads,
      int max_num_blocks_per_seq,
      int max_num_partitions,
      int q_stride,
      int kv_block_stride,
      int kv_head_stride,
      int num_heads,
      int num_seqs,
      float softmax_scale,
      std::optional<float> softcapping = std::nullopt)
      : UnaryPrimitive(stream),
        use_v1_(use_v1),
        max_context_len_(max_context_len),
        head_size_(head_size),
        block_size_(block_size),
        num_kv_heads_(num_kv_heads),
        max_num_blocks_per_seq_(max_num_blocks_per_seq),
        max_num_partitions_(max_num_partitions),
        q_stride_(q_stride),
        kv_block_stride_(kv_block_stride),
        kv_head_stride_(kv_head_stride),
        num_heads_(num_heads),
        num_seqs_(num_seqs),
        softmax_scale_(softmax_scale),
        softcapping_(softcapping) {}

  void eval_cpu(const std::vector<array>& inputs, array& outputs) override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, array& outputs) override;

  DEFINE_PRINT(PagedAttention);

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(
        max_context_len_,
        head_size_,
        block_size_,
        softmax_scale_,
        softcapping_);
  }

 private:
  bool use_v1_;
  int max_context_len_;
  int head_size_;
  int block_size_;
  int num_kv_heads_;
  int max_num_blocks_per_seq_;
  int max_num_partitions_;
  int q_stride_;
  int kv_block_stride_;
  int kv_head_stride_;
  int num_heads_;
  int num_seqs_;
  float softmax_scale_;
  std::optional<float> softcapping_ = std::nullopt;
};

} // namespace mlx::core::paged_attention