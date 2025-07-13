// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/primitives.h"

namespace mlx::core::distributed {

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

} // namespace mlx::core::distributed
