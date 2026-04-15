// Copyright © 2025 Apple Inc.

#include "mlx/distributed/jaccl/jaccl.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/dtype_utils.h"

#include <jaccl/group.h>
#include <jaccl/jaccl.h>

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

namespace mlx::core::distributed::jaccl {

namespace {

/**
 * Map MLX Dtype to JACCL Dtype for dispatch.
 */
int dtype_to_jaccl_dtype(Dtype dt) {
  switch (dt) {
    case bool_:
      return ::jaccl::Dtype::Bool;
    case int8:
      return ::jaccl::Dtype::Int8;
    case int16:
      return ::jaccl::Dtype::Int16;
    case int32:
      return ::jaccl::Dtype::Int32;
    case int64:
      return ::jaccl::Dtype::Int64;
    case uint8:
      return ::jaccl::Dtype::UInt8;
    case uint16:
      return ::jaccl::Dtype::UInt16;
    case uint32:
      return ::jaccl::Dtype::UInt32;
    case uint64:
      return ::jaccl::Dtype::UInt64;
    case float16:
      return ::jaccl::Dtype::Float16;
    case bfloat16:
      return ::jaccl::Dtype::BFloat16;
    case float32:
      return ::jaccl::Dtype::Float32;
    case float64:
      return ::jaccl::Dtype::Float64;
    case complex64:
      return ::jaccl::Dtype::Complex64;
    default:
      throw std::runtime_error("[jaccl] Unsupported dtype for JACCL operation");
  }
}

/**
 * Adapter that wraps a standalone jaccl::Group to implement
 * MLX's distributed::detail::GroupImpl interface.
 *
 * This bridges mlx::core::array to raw pointers for JACCL operations.
 */
class JACCLGroup : public GroupImpl {
 public:
  JACCLGroup(std::shared_ptr<::jaccl::Group> group)
      : group_(std::move(group)) {}

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }

  int rank() override {
    return group_->rank();
  }

  int size() override {
    return group_->size();
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    int dtype = dtype_to_jaccl_dtype(output.dtype());
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes, dtype, this]() {
      group_->all_sum(in_ptr, out_ptr, n_bytes, dtype);
    });
  }

  void all_max(const array& input, array& output, Stream stream) override {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    int dtype = dtype_to_jaccl_dtype(output.dtype());
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes, dtype, this]() {
      group_->all_max(in_ptr, out_ptr, n_bytes, dtype);
    });
  }

  void all_min(const array& input, array& output, Stream stream) override {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    int dtype = dtype_to_jaccl_dtype(output.dtype());
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes, dtype, this]() {
      group_->all_min(in_ptr, out_ptr, n_bytes, dtype);
    });
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
      group_->all_gather(in_ptr, out_ptr, n_bytes);
    });
  }

  void send(const array& input, int dst, Stream stream) override {
    auto data = input.data<char>();
    size_t n_bytes = input.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.dispatch(
        [data, n_bytes, dst, this]() { group_->send(data, n_bytes, dst); });
  }

  void recv(array& out, int src, Stream stream) override {
    auto data = out.data<char>();
    size_t n_bytes = out.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_output_array(out);
    encoder.dispatch(
        [data, n_bytes, src, this]() { group_->recv(data, n_bytes, src); });
  }

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] sum_scatter not supported.");
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

 private:
  std::shared_ptr<::jaccl::Group> group_;
};

} // namespace

bool is_available() {
  return ::jaccl::is_available();
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  auto group = ::jaccl::init(strict);
  if (group == nullptr) {
    return nullptr;
  }
  return std::make_shared<JACCLGroup>(std::move(group));
}

} // namespace mlx::core::distributed::jaccl
