// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/reduce/all_reduce.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>
#include <sstream>
#include <unordered_set>

namespace mlx::core {

namespace cu {

// Builder to generate prefix functor code that will be
// applied in reduction kernel
struct FusedReducePrefixBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build_prefix_struct() {
    NodeNamer namer;

    std::unordered_set<uintptr_t> constant_set;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        constant_set.insert(inputs[i].id());
      }
    }

    os += "struct " + kernel_name + "_Prefix {\n";

    os += "\n  template <typename T>\n";
    os += "  __device__ __forceinline__ T operator()(T val) const {\n";

    // Read the first input (reduction input) from the passed value
    // Find the non-constant input
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        os += fmt::format(
            "    {} tmp_{} = static_cast<{}>({}); // constant\n",
            type,
            xname,
            type,
            ss.str());
      } else {
        os += fmt::format(
            "    {} tmp_{} = static_cast<{}>(val);\n", type, xname, type);
      }
    }

    for (const auto& x : tape) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = fmt::format(
            "static_cast<{}>(tmp_{})", type, namer.get_name(x.inputs()[0]));
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += fmt::format("tmp_{}, ", namer.get_name(x.inputs()[i]));
        }
        value += fmt::format("tmp_{})", namer.get_name(x.inputs().back()));
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Return the result (last tape item or first input if tape is empty)
    if (!tape.empty()) {
      os += fmt::format(
          "    return static_cast<T>(tmp_{});\n", namer.get_name(tape.back()));
    } else {
      // Find the non-constant input to return
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (!is_constant(i)) {
          os += fmt::format(
              "    return static_cast<T>(tmp_{});\n",
              namer.get_name(inputs[i]));
          break;
        }
      }
    }

    os += "  }\n";

    os += "};\n\n";
  }
};

} // namespace cu

constexpr const char* g_fused_reduce_includes = R"(
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/reduce/all_reduce.cuh"

#include <cooperative_groups.h>

#define inf cuda::std::numeric_limits<float>::infinity()
)";

std::string get_reduce_op_name(Reduce::ReduceType reduce_type) {
  switch (reduce_type) {
    case Reduce::ReduceType::And:
      return "And";
    case Reduce::ReduceType::Or:
      return "Or";
    case Reduce::ReduceType::Sum:
      return "Sum";
    case Reduce::ReduceType::Prod:
      return "Prod";
    case Reduce::ReduceType::Max:
      return "Max";
    case Reduce::ReduceType::Min:
      return "Min";
    default:
      throw std::runtime_error("Unknown reduce type");
  }
}

void fused_all_reduce(
    cu::CommandEncoder& encoder,
    const Reduce& reduce,
    const std::vector<array>& inputs,
    array& out,
    const Stream& stream) {
  nvtx3::scoped_range r("fused_all_reduce");

  // Copied from all_reduce.cu
  constexpr int N_READS = 8;

  const auto& prefix_inputs = reduce.prefix_inputs();
  const auto& prefix_tape = reduce.prefix_tape();
  const auto& prefix_constant_ids = reduce.prefix_constant_ids();

  auto is_constant = [&](size_t i) -> bool {
    return prefix_constant_ids.count(prefix_inputs[i].id()) > 0;
  };

  NodeNamer namer;
  std::ostringstream os;
  std::ostringstream constant_hasher;

  for (const auto& x : prefix_inputs) {
    namer.get_name(x);
  }

  // Build string from tape operations
  for (const auto& a : prefix_tape) {
    os << namer.get_name(a) << kindof(a.dtype()) << a.itemsize();
    os << a.primitive().name();
    for (const auto& inp : a.inputs()) {
      os << namer.get_name(inp);
    }
  }
  // Name the kernel: similar to Compiled::Compiled kernel naming
  for (size_t i = 0; i < prefix_inputs.size(); ++i) {
    const auto& x = prefix_inputs[i];
    if (is_constant(i)) {
      os << "C";
      print_constant(constant_hasher, x);
    } else {
      os << (is_scalar(x) ? "S" : "V");
    }
  }

  os << get_reduce_op_name(reduce.state().first);
  os << dtype_to_cuda_type(prefix_inputs[0].dtype());
  os << std::hash<std::string>{}(constant_hasher.str());

  std::string kernel_name = os.str();

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  // Copied from all_reduce.cu
  auto get_args = [](int size, int N) {
    int threads = std::min(512, (size + N - 1) / N);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int reductions_per_step = threads * N;
    size_t steps_needed =
        (size + reductions_per_step - 1) / reductions_per_step;

    int blocks;
    if (steps_needed < 32) {
      blocks = 1;
    } else if (steps_needed < 128) {
      blocks = 32;
    } else if (steps_needed < 512) {
      blocks = 128;
    } else if (steps_needed < 1024) {
      blocks = 512;
    } else {
      blocks = 1024;
    }

    size_t steps_per_block = (steps_needed + blocks - 1) / blocks;
    size_t block_step = steps_per_block * reductions_per_step;

    return std::make_tuple(blocks, threads, block_step);
  };

  std::string reduce_op = get_reduce_op_name(reduce.state().first);
  std::string in_type = dtype_to_cuda_type(prefix_inputs[0].dtype());
  std::string prefix_type = kernel_name + "_Prefix";

  std::string full_kernel_name =
      fmt::format("mlx::core::cu::{}_all_reduce", kernel_name);

  cu::JitModule& mod = cu::get_jit_module(stream.device, kernel_name, [&]() {
    cu::FusedReducePrefixBuilder builder{
        g_fused_reduce_includes,
        kernel_name,
        prefix_inputs,
        prefix_tape,
        is_constant};
    builder.os +=
        "namespace mlx::core::cu {\n\n"
        "namespace cg = cooperative_groups;\n\n";

    // Generate the prefix struct
    builder.build_prefix_struct();

    builder.os += fmt::format(
        "__global__ void {}_all_reduce({}* in, {}* out, size_t block_step, size_t size) {{\n"
        "  {} prefix{{}};\n"
        "  all_reduce_impl<{}, {}, {}, {}, {}>(in, out, block_step, size, prefix);\n"
        "}}\n",
        kernel_name,
        in_type,
        in_type,
        prefix_type,
        in_type,
        in_type,
        reduce_op,
        N_READS,
        prefix_type);

    builder.os += "\n} // namespace mlx::core::cu\n";

    std::vector<std::string> kernel_names;
    kernel_names.push_back(full_kernel_name);

    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  int blocks, threads;
  size_t block_step;

  size_t insize = inputs[0].size();
  Reduce::ReduceType reduce_type = reduce.state().first;

  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);

  encoder.set_input_array(inputs[0]);

  // If the reduction needs more than 1 block -- use fused kernel with
  // fused prefix, then reduce intermediate to final output using build compiled
  // all_reduce
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
    encoder.add_temporary(intermediate);
    encoder.set_output_array(intermediate);

    // First pass: apply fused prefix and reduce to intermediate
    cu::KernelArgs args;
    args.append(inputs[0]);
    args.append(intermediate);
    args.append<size_t>(block_step);
    args.append<size_t>(insize);

    auto kernel = mod.get_kernel(full_kernel_name);
    encoder.add_kernel_node(kernel, blocks, threads, 0, args.args());

    // Second pass: reduce intermediate to final output
    size_t intermediate_size = intermediate.size();
    std::tie(blocks, threads, block_step) =
        get_args(intermediate_size, N_READS);
    encoder.set_input_array(intermediate);
    encoder.set_output_array(out);

    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;
        auto kernel2 = cu::all_reduce<T, U, OP, N_READS>;
        encoder.add_kernel_node(
            kernel2,
            blocks,
            threads,
            0,
            gpu_ptr<T>(intermediate),
            gpu_ptr<U>(out),
            block_step,
            intermediate_size);
      });
    });
  } else {
    // Single block: direct reduction with fused prefix
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(inputs[0]);
    args.append(out);
    args.append<size_t>(block_step);
    args.append<size_t>(insize);

    auto kernel = mod.get_kernel(full_kernel_name);
    encoder.add_kernel_node(kernel, blocks, threads, 0, args.args());
  }
}

void fused_reduce(
    cu::CommandEncoder& encoder,
    const Reduce& reduce,
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const Stream& stream) {
  if (plan.type == ContiguousAllReduce) {
    fused_all_reduce(encoder, reduce, inputs, out, stream);
    return;
  }

  // TODO: Implement fused row_reduce and col_reduce
  throw std::runtime_error(
      "Fused reduce not yet implemented for this reduction type");
}

} // namespace mlx::core
