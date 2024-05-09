// Copyright © 2024 Apple Inc.

#include <mpi.h>

#include "mlx/dist/dist.h"
#include "mlx/scheduler.h"

namespace mlx::core::dist {

namespace {

struct MPIGroup : public Group {
  MPIGroup() : is_global(true) {
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size_);
    MPI_Comm_rank(comm, &rank_);
  }

  ~MPIGroup() {
    if (is_global) {
      MPI_Finalize();
    }
  }

  virtual int rank() override {
    return rank_;
  }

  virtual int size() override {
    return size_;
  }

  virtual std::shared_ptr<Group> split(int n) override {
    throw std::runtime_error("MPIGroup split not yet implemented");
  }

  bool is_global;
  MPI_Comm comm;
  int rank_;
  int size_;
};

MPI_Datatype mpi_datatype(const array& arr) {
  switch (arr.dtype()) {
    case bool_:
      return MPI_C_BOOL;
    case int8:
      return MPI_INT8_T;
    case uint8:
      return MPI_UINT8_T;
    case int16:
      return MPI_INT16_T;
    case uint16:
      return MPI_UINT16_T;
    case int32:
      return MPI_INT32_T;
    case uint32:
      return MPI_UINT32_T;
    case int64:
      return MPI_INT64_T;
    case uint64:
      return MPI_UINT64_T;
    case float32:
      return MPI_FLOAT;
    case complex64:
      return MPI_C_COMPLEX;
    case float16:
    case bfloat16:
      throw std::runtime_error("MPI doesn't support 16-bit floats");
  }
}

} // namespace

bool is_available() {
  return true;
}

Stream stream() {
  static std::shared_ptr<Stream> comm_stream = nullptr;
  if (comm_stream == nullptr) {
    comm_stream = std::make_shared<Stream>(
        scheduler::scheduler().new_stream(Device::cpu));
  }
  return *comm_stream;
}

std::shared_ptr<Group> init() {
  static std::shared_ptr<MPIGroup> global_group = nullptr;
  if (global_group != nullptr) {
    return global_group;
  }

  MPI_Init(nullptr, nullptr);
  global_group = std::make_shared<MPIGroup>();

  return global_group;
}

void all_reduce_sum(
    std::shared_ptr<Group> group,
    const array& input,
    array& output) {
  auto mpi_group = std::dynamic_pointer_cast<MPIGroup>(group);
  MPI_Allreduce(
      input.data<void>(),
      output.data<void>(),
      input.nbytes(),
      mpi_datatype(input),
      MPI_SUM,
      mpi_group->comm);
}

} // namespace mlx::core::dist
