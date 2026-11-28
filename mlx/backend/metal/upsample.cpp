// Copyright © 2026 Apple Inc.
#include <algorithm>
#include <cmath>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

array ensure_row_contiguous(
    const array& x,
    metal::CommandEncoder& encoder,
    const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return x_copy;
}

// Sizes are passed as function constants so the kernel bakes them in as
// compile-time constants; hash_name keys the pipeline cache per shape.
MTL::ComputePipelineState* get_upsample_kernel(
    metal::Device& d,
    const std::string& base_name,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out) {
  metal::MTLFCList func_consts = {
      {&C, MTL::DataType::DataTypeInt, 0},
      {&H_in, MTL::DataType::DataTypeInt, 1},
      {&W_in, MTL::DataType::DataTypeInt, 2},
      {&H_out, MTL::DataType::DataTypeInt, 3},
      {&W_out, MTL::DataType::DataTypeInt, 4},
  };
  std::string hash_name;
  hash_name.reserve(64);
  concatenate(
      hash_name,
      base_name,
      "_C_",
      C,
      "_Hi_",
      H_in,
      "_Wi_",
      W_in,
      "_Ho_",
      H_out,
      "_Wo_",
      W_out);
  return d.get_kernel(base_name, hash_name, func_consts);
}

array make_host_array(Shape shape, Dtype dtype) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

// Per output index o: its two neighbour input indices l(o)/r(o) and the
// blend weight toward r(o). Matches the CPU fallback's float32 arithmetic.
struct AxisTables {
  array l;
  array r;
  array w;
};

AxisTables build_axis_tables(int N, int M, float step, float start) {
  AxisTables t{
      make_host_array({M}, uint32),
      make_host_array({M}, uint32),
      make_host_array({M}, float32)};
  auto* lp = t.l.data<uint32_t>();
  auto* rp = t.r.data<uint32_t>();
  auto* wp = t.w.data<float>();
  for (int o = 0; o < M; ++o) {
    float src = static_cast<float>(o) * step;
    src = src - start;
    src = std::clamp(src, 0.0f, static_cast<float>(N - 1));
    float l = std::floor(src);
    float r = std::ceil(src);
    lp[o] = static_cast<uint32_t>(l);
    rp[o] = static_cast<uint32_t>(r);
    wp[o] = src - l;
  }
  return t;
}

// Per input index i: the half-open range [lo(i), hi(i)) of output indices
// that read it, found with a two-pointer sweep over l and r (O(N + M)).
struct RangeTables {
  array lo;
  array hi;
};

RangeTables
build_range_tables(int N, int M, const uint32_t* l, const uint32_t* r) {
  RangeTables t{make_host_array({N}, uint32), make_host_array({N}, uint32)};
  auto* lop = t.lo.data<uint32_t>();
  auto* hip = t.hi.data<uint32_t>();
  int o = 0;
  for (int i = 0; i < N; ++i) {
    while (o < M && r[o] < static_cast<uint32_t>(i)) {
      ++o;
    }
    lop[i] = o;
  }
  o = 0;
  for (int i = 0; i < N; ++i) {
    while (o < M && l[o] <= static_cast<uint32_t>(i)) {
      ++o;
    }
    hip[i] = o;
  }
  return t;
}

} // namespace

bool UpsampleBilinear::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void UpsampleBilinear::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto in = ensure_row_contiguous(inputs[0], compute_encoder, s);
  out.set_data(allocator::malloc(out.nbytes()));

  int B = in.shape(0);
  int H_in = in.shape(1);
  int W_in = in.shape(2);
  int C = in.shape(3);
  int H_out = out_h_;
  int W_out = out_w_;

  auto y = build_axis_tables(H_in, H_out, step_h_, start_h_);
  auto x = build_axis_tables(W_in, W_out, step_w_, start_w_);

  auto kernel = get_upsample_kernel(
      d, "upsample_bilinear_" + type_to_name(in), C, H_in, W_in, H_out, W_out);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(y.l, 2);
  compute_encoder.set_input_array(y.r, 3);
  compute_encoder.set_input_array(y.w, 4);
  compute_encoder.set_input_array(x.l, 5);
  compute_encoder.set_input_array(x.r, 6);
  compute_encoder.set_input_array(x.w, 7);
  compute_encoder.add_temporaries({y.l, y.r, y.w, x.l, x.r, x.w});

  MTL::Size group_dims(std::min(C, 32), 8, 1);
  MTL::Size grid_dims(C, W_out, B * H_out);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void UpsampleBilinearVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto g = ensure_row_contiguous(inputs[0], compute_encoder, s);
  out.set_data(allocator::malloc(out.nbytes()));

  int B = g.shape(0);
  int H_out = g.shape(1);
  int W_out = g.shape(2);
  int C = g.shape(3);
  int H_in = in_h_;
  int W_in = in_w_;

  auto y = build_axis_tables(H_in, H_out, step_h_, start_h_);
  auto x = build_axis_tables(W_in, W_out, step_w_, start_w_);
  auto y_range = build_range_tables(
      H_in, H_out, y.l.data<uint32_t>(), y.r.data<uint32_t>());
  auto x_range = build_range_tables(
      W_in, W_out, x.l.data<uint32_t>(), x.r.data<uint32_t>());

  auto kernel = get_upsample_kernel(
      d,
      "upsample_bilinear_vjp_" + type_to_name(g),
      C,
      H_in,
      W_in,
      H_out,
      W_out);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(g, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(y.l, 2);
  compute_encoder.set_input_array(y.r, 3);
  compute_encoder.set_input_array(y.w, 4);
  compute_encoder.set_input_array(x.l, 5);
  compute_encoder.set_input_array(x.r, 6);
  compute_encoder.set_input_array(x.w, 7);
  compute_encoder.set_input_array(y_range.lo, 8);
  compute_encoder.set_input_array(y_range.hi, 9);
  compute_encoder.set_input_array(x_range.lo, 10);
  compute_encoder.set_input_array(x_range.hi, 11);
  compute_encoder.add_temporaries(
      {y.l,
       y.r,
       y.w,
       x.l,
       x.r,
       x.w,
       y_range.lo,
       y_range.hi,
       x_range.lo,
       x_range.hi});

  MTL::Size group_dims(std::min(C, 32), 8, 1);
  MTL::Size grid_dims(C, W_in, B * H_in);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
