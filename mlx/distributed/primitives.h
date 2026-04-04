// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/primitives.h"

namespace mlx::core::distributed {

// Optimized collective communication algorithms
enum class CollectiveAlgorithm {
  DEFAULT,   // Let the library choose the best algorithm
  LINEAR,    // Linear exchange (O(n) communication steps)
  RING,      // Ring-based all-reduce
  RECURSIVE_DOUBLING, // Recursive doubling (logarithmic steps)
  TREE,      // Tree-based reduction
  BROADCAST, // Broadcast-based gather
};

// Optimized all-reduce with algorithm selection
MLX_API array all_reduce_opt(
    const array& x,
    const std::string& op,
    std::optional<Group> group = std::nullopt,
    CollectiveAlgorithm algo = CollectiveAlgorithm::DEFAULT,
    StreamOrDevice s = {});

// Optimized all-gather with algorithm selection
MLX_API array all_gather_opt(
    const array& x,
    std::optional<Group> group = std::nullopt,
    CollectiveAlgorithm algo = CollectiveAlgorithm::DEFAULT,
    StreamOrDevice s = {});

// Optimized reduce-scatter with algorithm selection
MLX_API array reduce_scatter_opt(
    const array& x,
    const std::string& op,
    std::optional<Group> group = std::nullopt,
    CollectiveAlgorithm algo = CollectiveAlgorithm::DEFAULT,
    StreamOrDevice s = {});

// Pipeline parallelism utilities for overlapping computation and communication
struct PipelineStage {
  int stage_id;
  int num_stages;
  std::function<array(const array&)> compute_fn;
  
  PipelineStage(int id, int num, std::function<array(const array&)>&& compute)
      : stage_id(id), num_stages(num), compute_fn(std::move(compute)) {}
};

// Execute pipeline stages with pipelined execution
MLX_API array execute_pipeline(
    const std::vector<PipelineStage>& stages,
    const array& input,
    std::optional<Group> group = std::nullopt);

// Optimized communication with automatic algorithm selection
MLX_API array all_reduce(
    const array& x,
    const std::string& op = "sum",
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

class DistPrimitive : public Primitive {
 public:
  DistPrimitive(Stream stream, Group group)
      : Primitive(stream), group_(group) {}

  const Group& group() const {
    return group_;
  }

 private:
  Group group_;
};

class AllReduce : public DistPrimitive {
 public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };

  AllReduce(Stream stream, Group group, ReduceType reduce_type)
      : DistPrimitive(stream, group), reduce_type_(reduce_type) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;
  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  const char* name() const override {
    switch (reduce_type_) {
      case And:
        return "And AllReduce";
      case Or:
        return "Or AllReduce";
      case Sum:
        return "Sum AllReduce";
      case Prod:
        return "Prod AllReduce";
      case Min:
        return "Min AllReduce";
      case Max:
        return "Max AllReduce";
    }
    return "<unknwon AllReduce>";
  }

 private:
  ReduceType reduce_type_;
};

class AllGather : public DistPrimitive {
 public:
  AllGather(Stream stream, Group group) : DistPrimitive(stream, group) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;
  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(AllGather);
};

class Send : public DistPrimitive {
 public:
  Send(Stream stream, Group group, int dst)
      : DistPrimitive(stream, group), dst_(dst) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  DEFINE_NAME(Send);

 private:
  int dst_;
};

class Recv : public DistPrimitive {
 public:
  Recv(Stream stream, Group group, int src)
      : DistPrimitive(stream, group), src_(src) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(Recv);

 private:
  int src_;
};

class ReduceScatter : public DistPrimitive {
 public:
  enum ReduceType { Sum, Min, Max };
  ReduceScatter(Stream stream, Group group, ReduceType reduce_type)
      : DistPrimitive(stream, group), reduce_type_(reduce_type) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  const char* name() const override {
    switch (reduce_type_) {
      case Sum:
        return "Sum ReduceScatter";
      case Min:
        return "Min ReduceScatter";
      case Max:
        return "Max ReduceScatter";
    }
    return "<unknwon ReduceScatter>";
  }

 private:
  ReduceType reduce_type_;
};
} // namespace mlx::core::distributed
