// Copyright © 2024 Apple Inc.

#include <dlfcn.h>
#include <mpi.h>

#include "mlx/distributed/distributed.h"
#include "mlx/scheduler.h"

#define LOAD_SYMBOL(symbol, variable)                              \
  {                                                                \
    variable = (decltype(variable))dlsym(libmpi_handle_, #symbol); \
    char* error = dlerror();                                       \
    if (error != nullptr) {                                        \
      libmpi_handle_ = nullptr;                                    \
      return;                                                      \
    }                                                              \
  }

namespace mlx::core::distributed {

namespace {

struct MPIWrapper {
  MPIWrapper() {
    libmpi_handle_ = dlopen("libmpi.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (libmpi_handle_ == nullptr) {
      return;
    }

    // API
    LOAD_SYMBOL(MPI_Init, init);
    LOAD_SYMBOL(MPI_Finalize, finalize);
    LOAD_SYMBOL(MPI_Comm_rank, rank);
    LOAD_SYMBOL(MPI_Comm_size, size);
    LOAD_SYMBOL(MPI_Comm_split, comm_split);
    LOAD_SYMBOL(MPI_Comm_free, comm_free);
    LOAD_SYMBOL(MPI_Allreduce, all_reduce);
    LOAD_SYMBOL(MPI_Allgather, all_gather);

    // Objects
    LOAD_SYMBOL(ompi_mpi_comm_world, comm_world_);

    // Ops
    LOAD_SYMBOL(ompi_mpi_op_sum, op_sum_);

    // Datatypes
    LOAD_SYMBOL(ompi_mpi_c_bool, mpi_bool_);
    LOAD_SYMBOL(ompi_mpi_int8_t, mpi_int8_);
    LOAD_SYMBOL(ompi_mpi_uint8_t, mpi_uint8_);
    LOAD_SYMBOL(ompi_mpi_int16_t, mpi_int16_);
    LOAD_SYMBOL(ompi_mpi_uint16_t, mpi_uint16_);
    LOAD_SYMBOL(ompi_mpi_int32_t, mpi_int32_);
    LOAD_SYMBOL(ompi_mpi_uint32_t, mpi_uint32_);
    LOAD_SYMBOL(ompi_mpi_int64_t, mpi_int64_);
    LOAD_SYMBOL(ompi_mpi_uint64_t, mpi_uint64_);
    LOAD_SYMBOL(ompi_mpi_float, mpi_float_);
    LOAD_SYMBOL(ompi_mpi_c_complex, mpi_complex_);
  }

  bool is_available() {
    return libmpi_handle_ != nullptr;
  }

  void init_safe() {
    if (is_available()) {
      init(nullptr, nullptr);
    }
  }

  void finalize_safe() {
    if (is_available()) {
      finalize();
    }
  }

  MPI_Comm world() {
    return comm_world_;
  }

  MPI_Datatype datatype(const array& arr) {
    switch (arr.dtype()) {
      case bool_:
        return mpi_bool_;
      case int8:
        return mpi_int8_;
      case uint8:
        return mpi_uint8_;
      case int16:
        return mpi_int16_;
      case uint16:
        return mpi_uint16_;
      case int32:
        return mpi_int32_;
      case uint32:
        return mpi_uint32_;
      case int64:
        return mpi_int64_;
      case uint64:
        return mpi_uint64_;
      case float32:
        return mpi_float_;
      case complex64:
        return mpi_complex_;
      case float16:
      case bfloat16:
        throw std::runtime_error("MPI doesn't support 16-bit floats");
    }
  }

  MPI_Op op_sum() {
    return op_sum_;
  }

  void* libmpi_handle_;

  // API
  int (*init)(int*, char***);
  int (*finalize)();
  int (*rank)(MPI_Comm, int*);
  int (*size)(MPI_Comm, int*);
  int (*all_reduce)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
  int (*all_gather)(
      const void*,
      int,
      MPI_Datatype,
      void*,
      int,
      MPI_Datatype,
      MPI_Comm);
  int (*comm_split)(MPI_Comm, int, int, MPI_Comm*);
  int (*comm_free)(MPI_Comm*);

  // Objects
  MPI_Comm comm_world_;

  // Ops
  MPI_Op op_sum_;

  // Datatypes
  MPI_Datatype mpi_bool_;
  MPI_Datatype mpi_int8_;
  MPI_Datatype mpi_uint8_;
  MPI_Datatype mpi_int16_;
  MPI_Datatype mpi_uint16_;
  MPI_Datatype mpi_int32_;
  MPI_Datatype mpi_uint32_;
  MPI_Datatype mpi_int64_;
  MPI_Datatype mpi_uint64_;
  MPI_Datatype mpi_float_;
  MPI_Datatype mpi_complex_;
};

MPIWrapper& mpi() {
  static MPIWrapper wrapper;
  return wrapper;
}

struct MPIGroup : public Group {
  MPIGroup() : is_global(true) {
    if (mpi().is_available()) {
      comm = mpi().world();
      mpi().rank(comm, &rank_);
      mpi().size(comm, &size_);
    } else {
      size_ = 1;
      rank_ = 0;
    }
  }

  MPIGroup(MPI_Comm comm_) : is_global(false), comm(comm_) {
    mpi().rank(comm, &rank_);
    mpi().size(comm, &size_);
  }

  ~MPIGroup() {
    if (is_global) {
      mpi().finalize_safe();
    } else {
      mpi().comm_free(&comm);
    }
  }

  virtual int rank() override {
    return rank_;
  }

  virtual int size() override {
    return size_;
  }

  virtual std::shared_ptr<Group> split(int color, int key = -1) override {
    key = (key < 0) ? rank_ : key;
    MPI_Comm new_comm;
    int result = mpi().comm_split(comm, color, key, &new_comm);
    if (result != MPI_SUCCESS) {
      throw std::runtime_error("MPI could not split this group");
    }

    return std::make_shared<MPIGroup>(new_comm);
  }

  bool is_global;
  MPI_Comm comm;
  int rank_;
  int size_;
};

} // namespace

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

} // namespace detail

bool is_available() {
  return mpi().is_available();
}

std::shared_ptr<Group> init() {
  static std::shared_ptr<MPIGroup> global_group = nullptr;
  if (global_group != nullptr) {
    return global_group;
  }

  mpi().init_safe();
  global_group = std::make_shared<MPIGroup>();

  return global_group;
}

void all_reduce_sum(
    std::shared_ptr<Group> group,
    const array& input,
    array& output) {
  auto mpi_group = std::dynamic_pointer_cast<MPIGroup>(group);
  mpi().all_reduce(
      input.data<void>(),
      output.data<void>(),
      input.size(),
      mpi().datatype(input),
      mpi().op_sum(),
      mpi_group->comm);
}

void all_gather(
    std::shared_ptr<Group> group,
    const array& input,
    array& output) {
  auto mpi_group = std::dynamic_pointer_cast<MPIGroup>(group);
  mpi().all_gather(
      input.data<void>(),
      input.size(),
      mpi().datatype(input),
      output.data<void>(),
      input.size(),
      mpi().datatype(output),
      mpi_group->comm);
}

} // namespace mlx::core::distributed
