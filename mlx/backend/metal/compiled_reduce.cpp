// Copyright © 2026 Apple Inc.

#include <fmt/format.h>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

// Threadgroups to aim for before the reduction is split.
constexpr int64_t target_groups = 1024;
// Row kernel: minimum number of elements for one threadgroup of a split row.
constexpr int64_t row_min_block = 8192;
// Row kernel: maximum threadgroup size. More work per thread is faster.
constexpr int64_t row_max_threads = 256;
// Row kernel: one thread reduces rows of up to this many reads.
constexpr int64_t row_small_reads = 8;
// Column kernel: threads along the reduction.
constexpr int col_threads = 256;
// Column kernel: minimum number of reduced rows per thread for one split.
constexpr int64_t col_min_rows = 32;

inline int64_t ceil_div(int64_t n, int64_t m) {
  return (n + m - 1) / m;
}

std::string reduce_op_name(const array& reduce) {
  auto type = static_cast<const Reduce&>(reduce.primitive()).state().first;
  bool is_bool = reduce.dtype() == bool_;
  switch (type) {
    case Reduce::And:
      return "and";
    case Reduce::Or:
      return "or";
    case Reduce::Sum:
      return "sum";
    case Reduce::Prod:
      return "prod";
    case Reduce::Min:
      return is_bool ? "and" : "min";
    case Reduce::Max:
      return is_bool ? "or" : "max";
  }
  return "";
}

// The space before the reduction as kept x reduced, with each dimension
// collapsed.
// - Row: the reduced axes make one row with stride 1 or 0 for each strided
//   input. The pattern has 'c' for stride 1 and 'b' for stride 0.
// - Col: the reduced axes make one dimension with any stride. The last kept
//   dimension has stride 1 or 0 for each strided input (the pattern).
// The shape and strides are for the kept dimensions, without the last one for
// Col.
struct ReduceLayout {
  enum Type { None, Row, Col };
  Type type{None};
  std::string pattern;
  Shape shape;
  Strides strides;
  Strides reduce_strides;
  int64_t outer{1};
  int64_t cols{1};
  int64_t reduce_size{1};
};

ReduceLayout get_reduce_layout(
    const std::vector<array>& inputs,
    const std::function<bool(size_t)>& is_constant,
    const Shape& shape,
    const std::vector<int>& axes) {
  ReduceLayout layout;

  // Broadcast the strides of the strided inputs to the full shape
  std::vector<Strides> strides;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& x = inputs[i];
    if (is_constant(i) || is_scalar(x)) {
      continue;
    }
    Strides xstrides(shape.size(), 0);
    int offset = shape.size() - x.ndim();
    for (int j = 0; j < x.ndim(); ++j) {
      if (x.shape(j) != 1) {
        xstrides[offset + j] = x.strides()[j];
      }
    }
    strides.push_back(std::move(xstrides));
  }
  if (strides.empty()) {
    return layout;
  }

  // A full reduction can read contiguous inputs in memory order
  if (axes.size() == shape.size()) {
    auto [contiguous, negative_strides] =
        compiled_check_contiguity(inputs, shape);
    if (contiguous && !negative_strides) {
      layout.type = ReduceLayout::Row;
      layout.pattern = std::string(strides.size(), 'c');
      for (auto s : shape) {
        layout.reduce_size *= s;
      }
      return layout;
    }
  }

  // Separate and collapse the kept and the reduced axes
  std::vector<bool> reduced(shape.size(), false);
  for (auto ax : axes) {
    reduced[ax] = true;
  }
  Shape keep_shape;
  Shape red_shape;
  std::vector<Strides> keep_strides(strides.size());
  std::vector<Strides> red_strides(strides.size());
  for (int d = 0; d < shape.size(); ++d) {
    (reduced[d] ? red_shape : keep_shape).push_back(shape[d]);
    for (int i = 0; i < strides.size(); ++i) {
      (reduced[d] ? red_strides : keep_strides)[i].push_back(strides[i][d]);
    }
  }
  std::tie(red_shape, red_strides) =
      collapse_contiguous_dims(red_shape, red_strides);
  std::tie(keep_shape, keep_strides) =
      collapse_contiguous_dims(keep_shape, keep_strides);
  if (red_shape.size() > 1) {
    return layout;
  }
  layout.reduce_size = red_shape.empty() ? 1 : red_shape[0];

  // The pattern of stride 1 ('c') or 0 ('b') of a dimension
  auto get_pattern = [&](const std::vector<Strides>& st, int dim) {
    std::string pattern;
    for (auto& s : st) {
      auto stride = dim < 0 ? 0 : s[dim];
      if (stride != 0 && stride != 1) {
        return std::string();
      }
      pattern += (stride == 1) ? 'c' : 'b';
    }
    return pattern;
  };

  int keep_ndim = keep_shape.size();
  layout.pattern = get_pattern(red_strides, int(red_shape.size()) - 1);
  if (!layout.pattern.empty()) {
    layout.type = ReduceLayout::Row;
  } else if (keep_ndim > 0) {
    layout.pattern = get_pattern(keep_strides, keep_ndim - 1);
    if (layout.pattern.empty()) {
      return layout;
    }
    layout.type = ReduceLayout::Col;
    layout.cols = keep_shape.back();
    keep_shape.pop_back();
    for (int i = 0; i < keep_strides.size(); ++i) {
      keep_strides[i].pop_back();
      layout.reduce_strides.push_back(red_strides[i][0]);
    }
  } else {
    return layout;
  }

  layout.shape = std::move(keep_shape);
  for (auto s : layout.shape) {
    layout.outer *= s;
  }
  for (auto& st : keep_strides) {
    layout.strides.insert(layout.strides.end(), st.begin(), st.end());
  }
  if (layout.outer > UINT32_MAX) {
    layout.type = ReduceLayout::None;
  }
  return layout;
}

// Source parts shared by the row and column kernels.
struct ReduceSource {
  NodeNamer namer;
  std::string out_t;
  std::string acc_t;
  std::string op;
  // Inputs which are read for each element (pattern 'c')
  std::vector<array> elem_inputs;
  // Inputs which are read once per row or column (pattern 'b')
  std::vector<array> fixed_inputs;
};

// Sums and products of 16 bit floats accumulate in float32.
Dtype accumulate_type(const array& reduce) {
  auto op = reduce_op_name(reduce);
  bool half = reduce.dtype() == float16 || reduce.dtype() == bfloat16;
  return (half && (op == "sum" || op == "prod")) ? float32 : reduce.dtype();
}

ReduceSource reduce_source(const std::vector<array>& tape, Dtype out_type) {
  ReduceSource src;
  auto& reduce = tape.back();
  src.out_t = get_type_string(out_type);
  src.acc_t = get_type_string(accumulate_type(reduce));
  auto op = reduce_op_name(reduce);
  op[0] = std::toupper(op[0]);
  src.op = op + "<" + src.acc_t + ">";
  return src;
}

// Write the input and common arguments and return the next buffer index.
int write_arguments(
    std::string& os,
    ReduceSource& src,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::function<bool(size_t)>& is_constant) {
  os += fmt::format(
      "[[host_name(\"{0}\")]]\n[[kernel]] void {0}(\n", kernel_name);
  int cnt = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    os += fmt::format(
        "    device const {0}* {1} [[buffer({2})]],\n",
        get_type_string(inputs[i].dtype()),
        src.namer.get_name(inputs[i]),
        cnt++);
  }
  os += fmt::format(
      "    device {0}* out [[buffer({1})]],\n"
      "    constant const int* shape [[buffer({2})]],\n"
      "    constant const int64_t* strides [[buffer({3})]],\n"
      "    constant const int& ndim [[buffer({4})]],\n"
      "    constant const int64_t& reduce_size [[buffer({5})]],\n"
      "    constant const int64_t& block_size [[buffer({6})]],\n",
      src.out_t,
      cnt,
      cnt + 1,
      cnt + 2,
      cnt + 3,
      cnt + 4,
      cnt + 5);
  return cnt + 6;
}

// Move the strided inputs to the start of their block. `start` is the offset
// of the block for inputs with pattern 'c' and `offset(j)` for all strided
// inputs. Load the constants and scalars.
void write_setup(
    std::string& os,
    ReduceSource& src,
    const std::vector<array>& inputs,
    const std::function<bool(size_t)>& is_constant,
    const std::string& pattern,
    const std::string& start,
    const std::function<std::string(int)>& offset) {
  std::string loc_init;
  std::string loc_update;
  std::string loads;
  int j = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& x = inputs[i];
    auto& xname = src.namer.get_name(x);
    auto type_str = get_type_string(x.dtype());
    if (is_constant(i)) {
      std::ostringstream ss;
      print_constant(ss, x);
      loads += fmt::format(
          "  auto tmp_{0} = static_cast<{1}>({2});\n",
          xname,
          type_str,
          ss.str());
    } else if (is_scalar(x)) {
      loads += fmt::format("  {0} tmp_{1} = {1}[0];\n", type_str, xname);
    } else {
      loc_init += fmt::format("  int64_t loc_{0} = {1};\n", xname, offset(j));
      loc_update += fmt::format(
          "    loc_{0} += l * strides[{1} * ndim + d];\n", xname, j);
      if (pattern[j] == 'c') {
        loads += fmt::format("  {0} += loc_{0} + {1};\n", xname, start);
        src.elem_inputs.push_back(x);
      } else {
        loads += fmt::format("  {0} += loc_{0};\n", xname);
        src.fixed_inputs.push_back(x);
      }
      j++;
    }
  }
  os += loc_init;
  os += "  int64_t r = row;\n";
  os += "  for (int d = ndim - 1; d >= 0; --d) {\n";
  os += "    int64_t l = r % shape[d];\n";
  os += loc_update;
  os += "    r /= shape[d];\n  }\n";
  os += loads;
}

// The body for one element. `elem` and `fixed` give the index into inputs
// with pattern 'c' and 'b'.
std::string reduce_body(
    ReduceSource& src,
    const std::vector<array>& tape,
    const std::string& total,
    const std::function<std::string(const std::string&)>& elem,
    const std::function<std::string(const std::string&)>& fixed) {
  std::string body;
  auto load = [&](const array& x, const std::string& idx) {
    body += fmt::format(
        "  {0} tmp_{1} = {1}[{2}];\n",
        get_type_string(x.dtype()),
        src.namer.get_name(x),
        idx);
  };
  for (auto& x : src.elem_inputs) {
    load(x, elem(src.namer.get_name(x)));
  }
  for (auto& x : src.fixed_inputs) {
    auto idx = fixed(src.namer.get_name(x));
    if (!idx.empty()) {
      load(x, idx);
    }
  }
  build_tape(body, std::vector<array>(tape.begin(), tape.end() - 1), src.namer);
  body += fmt::format(
      "  {0} = op({0}, cast_to<{1}>(tmp_{2}));\n",
      total,
      src.acc_t,
      src.namer.get_name(tape.back().inputs()[0]));
  return body;
}

// Reduce rows. In the small variant one thread reduces one row. Otherwise
// one threadgroup reduces one block of one row. The block is the full row
// unless the row is split, then the output holds partials.
std::string build_row_kernel(
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const std::string& pattern,
    Dtype out_type,
    bool small,
    int n_reads) {
  std::string os;
  auto src = reduce_source(tape, out_type);
  write_arguments(os, src, kernel_name, inputs, is_constant);
  if (small) {
    os += "    uint tid [[thread_position_in_grid]]) {\n";
    os += "  int64_t row = tid;\n";
    os += "  int64_t k0 = 0;\n";
    os += "  int n = reduce_size;\n";
  } else {
    os += fmt::format(
        "    uint3 gid [[threadgroup_position_in_grid]],\n"
        "    uint3 gsize [[threadgroups_per_grid]],\n"
        "    uint3 lsize [[threads_per_threadgroup]],\n"
        "    uint lid [[thread_index_in_threadgroup]],\n"
        "    uint simd_lane_id [[thread_index_in_simdgroup]],\n"
        "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_per_group [[simdgroups_per_threadgroup]]) {{\n"
        "  threadgroup {0} shared_vals[simd_size];\n"
        "  int64_t row = gid.y;\n"
        "  int64_t k0 = gid.x * block_size;\n"
        "  int n = static_cast<int>(metal::min(block_size, "
        "reduce_size - k0));\n",
        src.acc_t);
  }
  os += fmt::format(
      "  constexpr int N_READS = {0};\n  {1} op;\n", n_reads, src.op);
  write_setup(
      os, src, inputs, is_constant, pattern, "k0", [](int) { return "0"; });
  for (auto& x : src.fixed_inputs) {
    os += fmt::format(
        "  {0} tmp_{1} = {1}[0];\n",
        get_type_string(x.dtype()),
        src.namer.get_name(x));
  }

  // Each thread reduces N_READS consecutive elements at a time
  auto body = reduce_body(
      src,
      tape,
      "total",
      [](auto&) { return "k + i"; },
      [](auto&) { return ""; });
  os += fmt::format("  {0} total = {1}::init;\n", src.acc_t, src.op);
  if (small) {
    os += "  for (int k = 0; k < n; k += N_READS) {\n";
  } else {
    os += "  for (int k = lid * N_READS; k < n; k += lsize.x * N_READS) {\n";
  }
  os += "  if (k + N_READS <= n) {\n";
  os += "  for (int i = 0; i < N_READS; i++) {\n" + body + "  }\n";
  os += "  } else {\n";
  os += "  for (int i = 0; k + i < n; i++) {\n" + body + "  }\n";
  os += "  }\n  }\n";

  // Reduce in the threadgroup and write the result
  if (small) {
    os += fmt::format("  out[row] = static_cast<{0}>(total);\n}}\n", src.out_t);
    return os;
  }
  os += fmt::format(
      "  total = op.simd_reduce(total);\n"
      "  if (simd_per_group > 1) {{\n"
      "    if (simd_lane_id == 0) {{\n"
      "      shared_vals[simd_group_id] = total;\n"
      "    }}\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    total = lid < simd_per_group ? shared_vals[lid] : {0}::init;\n"
      "    total = op.simd_reduce(total);\n"
      "  }}\n"
      "  if (lid == 0) {{\n"
      "    out[row * gsize.x + gid.x] = static_cast<{1}>(total);\n"
      "  }}\n"
      "}}\n",
      src.op,
      src.out_t);
  return os;
}

// Reduce columns. A threadgroup of lsize.x x BM threads reduces
// BN = lsize.x * N_READS columns over one block of the reduced rows. Each
// thread reduces N_READS consecutive columns. BM is a power of 2. The output
// holds partials if the rows are split.
std::string build_col_kernel(
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const std::string& pattern,
    Dtype out_type,
    int n_reads) {
  std::string os;
  auto src = reduce_source(tape, out_type);
  int cnt = write_arguments(os, src, kernel_name, inputs, is_constant);
  os += fmt::format(
      "    constant const int64_t* reduce_strides [[buffer({0})]],\n"
      "    constant const int64_t& cols [[buffer({1})]],\n"
      "    uint3 gid [[threadgroup_position_in_grid]],\n"
      "    uint3 gsize [[threadgroups_per_grid]],\n"
      "    uint3 lid [[thread_position_in_threadgroup]],\n"
      "    uint3 lsize [[threads_per_threadgroup]]) {{\n"
      "  constexpr int N_READS = {2};\n"
      "  int BM = lsize.y;\n"
      "  int BN = lsize.x * N_READS;\n"
      "  {4} op;\n"
      "  threadgroup {5} shared_vals[{3} * N_READS];\n"
      "  int64_t row = gid.y;\n"
      "  int64_t m0 = int64_t(gid.x) * BN + lid.x * N_READS;\n"
      "  int64_t k0 = int64_t(gid.z) * block_size;\n"
      "  int n = static_cast<int>(metal::min(block_size, "
      "reduce_size - k0));\n"
      "  int n_cols = static_cast<int>(metal::clamp(cols - m0, "
      "int64_t(0), int64_t(N_READS)));\n",
      cnt,
      cnt + 1,
      n_reads,
      col_threads,
      src.op,
      src.acc_t);
  write_setup(os, src, inputs, is_constant, pattern, "m0", [](int j) {
    return fmt::format("k0 * reduce_strides[{0}]", j);
  });
  int j = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!is_constant(i) && !is_scalar(inputs[i])) {
      os += fmt::format(
          "  int64_t rs_{0} = reduce_strides[{1}];\n",
          src.namer.get_name(inputs[i]),
          j++);
    }
  }

  // Loop over the rows of the block
  auto body = reduce_body(
      src,
      tape,
      "totals[i]",
      [](auto& name) { return fmt::format("k * rs_{0} + i", name); },
      [](auto&) { return ""; });
  std::string fixed_loads;
  for (auto& x : src.fixed_inputs) {
    auto& name = src.namer.get_name(x);
    fixed_loads += fmt::format(
        "  {0} tmp_{1} = {1}[k * rs_{1}];\n", get_type_string(x.dtype()), name);
  }
  os += fmt::format(
      "  {0} totals[N_READS];\n"
      "  for (int i = 0; i < N_READS; i++) {{\n"
      "    totals[i] = {1}::init;\n"
      "  }}\n",
      src.acc_t,
      src.op);
  os += "  if (n_cols == N_READS) {\n";
  os += "  for (int k = lid.y; k < n; k += BM) {\n" + fixed_loads;
  os += "  for (int i = 0; i < N_READS; i++) {\n" + body + "  }\n  }\n";
  os += "  } else {\n";
  os += "  for (int k = lid.y; k < n; k += BM) {\n" + fixed_loads;
  os += "  for (int i = 0; i < n_cols; i++) {\n" + body + "  }\n  }\n";
  os += "  }\n";

  // Reduce across the threadgroup rows and write the result
  os += fmt::format(
      "  for (int i = 0; i < N_READS; i++) {{\n"
      "    shared_vals[lid.y * BN + lid.x * N_READS + i] = totals[i];\n"
      "  }}\n"
      "  for (int s = BM / 2; s > 0; s /= 2) {{\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    if (lid.y < s) {{\n"
      "      for (int i = 0; i < N_READS; i++) {{\n"
      "        int idx = lid.y * BN + lid.x * N_READS + i;\n"
      "        totals[i] = op(totals[i], shared_vals[idx + s * BN]);\n"
      "        shared_vals[idx] = totals[i];\n"
      "      }}\n"
      "    }}\n"
      "  }}\n"
      "  if (lid.y == 0) {{\n"
      "    out += (int64_t(gid.z) * gsize.y + row) * cols + m0;\n"
      "    for (int i = 0; i < n_cols; i++) {{\n"
      "      out[i] = static_cast<{0}>(totals[i]);\n"
      "    }}\n"
      "  }}\n"
      "}}\n",
      src.out_t);
  return os;
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::function<std::string()>& build) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::string kernel = metal::utils();
    concatenate(
        kernel,
        metal::unary_ops(),
        metal::binary_ops(),
        metal::ternary_ops(),
        metal::reduce_utils(),
        build());
    return kernel;
  });
  return d.get_kernel(kernel_name, lib);
}

// Compute the elementwise part in a temporary and reduce it with the
// regular reduction.
void compiled_reduce_fallback(
    const Stream& s,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    const std::vector<array>& inputs,
    const Shape& shape,
    array& out) {
  auto& reduce = tape.back();
  auto& reduce_in = reduce.inputs()[0];
  array tmp(shape, reduce_in.dtype(), nullptr, {});
  if (tmp.size() > 0) {
    Compiled elementwise(
        s,
        trace_inputs,
        {reduce_in},
        std::vector<array>(tape.begin(), tape.end() - 1),
        constant_ids);
    std::vector<array> outs{tmp};
    elementwise.eval_gpu(inputs, outs);
  } else {
    tmp.set_data(allocator::malloc(0));
  }
  metal::get_command_encoder(s).add_temporary(tmp);
  auto [type, axes] = static_cast<const Reduce&>(reduce.primitive()).state();
  Reduce(s, type, axes).eval_gpu({tmp}, out);
}

} // namespace

void compiled_reduce(
    const Stream& s,
    const std::string& lib_name,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    const std::function<bool(size_t)>& is_constant,
    const std::vector<array>& inputs,
    array& out) {
  Shape shape;
  for (auto& in : inputs) {
    shape = broadcast_shapes(shape, in.shape());
  }
  auto axes =
      static_cast<const Reduce&>(tape.back().primitive()).state().second;
  auto layout = get_reduce_layout(inputs, is_constant, shape, axes);
  if (layout.type == ReduceLayout::None) {
    return compiled_reduce_fallback(
        s, trace_inputs, tape, constant_ids, inputs, shape, out);
  }
  bool row = layout.type == ReduceLayout::Row;

  // Read 32 (row) or 8 (column) bytes of the widest input with pattern 'c'
  // per thread and step
  int max_itemsize = 1;
  int n_buffers = row ? 6 : 8;
  for (int i = 0, j = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    n_buffers++;
    if (!is_scalar(inputs[i]) && layout.pattern[j++] == 'c') {
      max_itemsize = std::max(max_itemsize, int(inputs[i].itemsize()));
    }
  }
  int n_reads = row ? std::clamp(32 / max_itemsize, REDUCE_N_READS, 16)
                    : std::clamp(8 / max_itemsize, 2, 8);

  // Narrow columns use fewer threads along the columns and more along the
  // reduction
  int64_t col_lanes = next_power_of_2(ceil_div(layout.cols, n_reads));
  col_lanes = std::clamp<int64_t>(col_lanes, 1, 32);
  int64_t col_bm = col_threads / col_lanes;
  int64_t col_groups = ceil_div(layout.cols, col_lanes * n_reads);

  // Metal has 31 buffer slots and grid dimensions below 2^32 threads
  if (n_buffers > 31 || (!row && layout.outer * col_bm > int64_t(UINT32_MAX))) {
    return compiled_reduce_fallback(
        s, trace_inputs, tape, constant_ids, inputs, shape, out);
  }

  out.set_data(allocator::malloc(std::max(out.nbytes(), size_t(4))));
  if (out.size() == 0) {
    return;
  }

  int64_t outer = layout.outer;
  int64_t size = layout.reduce_size;
  bool small = row && size <= row_small_reads * n_reads;

  // Split the reduction when there are few threadgroups. Blocks are below
  // 2^30 so int32 indices in the kernels do not overflow.
  int64_t groups = row ? outer : outer * col_groups;
  int64_t min_block = row ? row_min_block : col_bm * col_min_rows;
  int64_t n_splits = 1;
  if (!small && groups < target_groups) {
    n_splits =
        std::min(ceil_div(size, min_block), ceil_div(target_groups, groups));
  }
  n_splits = std::max(n_splits, ceil_div(size, int64_t(1) << 30));
  int64_t block_size = ceil_div(size, std::max(n_splits, int64_t(1)));
  n_splits = block_size > 0 ? ceil_div(size, block_size) : 1;

  // Partial results keep the accumulation type
  Dtype out_type = n_splits > 1 ? accumulate_type(tape.back()) : out.dtype();

  auto& d = metal::device(s.device);
  auto kernel_name = fmt::format(
      "{}_reduce_{}_{}_{}_{}",
      lib_name,
      row ? (small ? "small" : "row") : "col",
      layout.pattern,
      n_reads,
      type_to_name(out_type));
  auto kernel = get_reduce_kernel(d, kernel_name, [&]() {
    if (row) {
      return build_row_kernel(
          kernel_name,
          trace_inputs,
          tape,
          is_constant,
          layout.pattern,
          out_type,
          small,
          n_reads);
    } else {
      return build_col_kernel(
          kernel_name,
          trace_inputs,
          tape,
          is_constant,
          layout.pattern,
          out_type,
          n_reads);
    }
  });

  auto& compute_encoder = metal::get_command_encoder(s);
  auto make_temporary = [&](Shape shape, Dtype dtype) {
    array a(std::move(shape), dtype, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    compute_encoder.add_temporary(a);
    return a;
  };
  array partial = out;
  if (n_splits > 1) {
    partial = make_temporary(
        row ? Shape{int(outer), int(n_splits)}
            : Shape{int(n_splits), int(out.size())},
        out_type);
  }

  // Encode dummy values for no kept axes
  int ndim = layout.shape.size();
  if (ndim == 0) {
    layout.shape.push_back(0);
    layout.strides.push_back(0);
  }

  compute_encoder.set_compute_pipeline_state(kernel);
  int cnt = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (!is_constant(i)) {
      compute_encoder.set_input_array(inputs[i], cnt++);
    }
  }
  compute_encoder.set_output_array(partial, cnt++);
  compute_encoder.set_vector_bytes(layout.shape, cnt++);
  compute_encoder.set_vector_bytes(layout.strides, cnt++);
  compute_encoder.set_bytes(ndim, cnt++);
  compute_encoder.set_bytes(size, cnt++);
  compute_encoder.set_bytes(block_size, cnt++);
  if (small) {
    int64_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
    compute_encoder.dispatch_threads(
        MTL::Size(outer, 1, 1), MTL::Size(std::min(outer, max_threads), 1, 1));
  } else if (row) {
    int64_t max_threads = kernel->maxTotalThreadsPerThreadgroup() / 32 * 32;
    int64_t threads = ceil_div(block_size, n_reads);
    threads = std::clamp(
        ceil_div(threads, 32) * 32,
        int64_t(32),
        std::min(max_threads, row_max_threads));
    compute_encoder.dispatch_threadgroups(
        MTL::Size(n_splits, outer, 1), MTL::Size(threads, 1, 1));
  } else {
    // Keep BM a power of 2 within the kernel limit
    while (col_lanes * col_bm > kernel->maxTotalThreadsPerThreadgroup()) {
      col_bm /= 2;
    }
    compute_encoder.set_vector_bytes(layout.reduce_strides, cnt++);
    compute_encoder.set_bytes(layout.cols, cnt++);
    compute_encoder.dispatch_threadgroups(
        MTL::Size(col_groups, outer, n_splits),
        MTL::Size(col_lanes, col_bm, 1));
  }
  if (n_splits == 1) {
    return;
  }

  // Reduce the partial results into a flat view of the output, or into a
  // temporary of the accumulation type which is then cast to the output
  array result = out;
  if (out_type != out.dtype()) {
    result = make_temporary(out.shape(), out_type);
  }
  array flat({int(out.size())}, out_type, nullptr, {});
  flat.copy_shared_buffer(result, {1}, result.flags(), result.size());
  auto op_name = reduce_op_name(tape.back());
  if (row && outer == 1) {
    all_reduce_dispatch(partial, flat, op_name, compute_encoder, d, s);
  } else {
    std::vector<int> partial_axes = {row ? 1 : 0};
    auto plan = get_reduction_plan(partial, partial_axes);
    if (row) {
      row_reduce_general_dispatch(
          partial, flat, op_name, plan, partial_axes, compute_encoder, d, s);
    } else {
      strided_reduce_general_dispatch(
          partial, flat, op_name, plan, partial_axes, compute_encoder, d, s);
    }
  }
  if (out_type != out.dtype()) {
    copy_gpu_inplace(result, out, CopyType::Vector, s);
  }
}

} // namespace mlx::core
