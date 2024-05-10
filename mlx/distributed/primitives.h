// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"
#include "mlx/primitives.h"

namespace mlx::core::distributed {

class DistPrimitive : public Primitive {
 public:
  DistPrimitive(std::shared_ptr<Group> group)
      : Primitive(communication_stream()), group_(group) {}

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error(
        "Communication primitives cannot be run on the GPU");
  }

  const std::shared_ptr<Group>& group() const {
    return group_;
  }

 private:
  std::shared_ptr<Group> group_;
};

class AllReduce : public DistPrimitive {
 public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };

  AllReduce(std::shared_ptr<Group> group, ReduceType reduce_type)
      : DistPrimitive(group), reduce_type_(reduce_type) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
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
    os << " AllReduce";
  }

 private:
  ReduceType reduce_type_;
};

} // namespace mlx::core::distributed
