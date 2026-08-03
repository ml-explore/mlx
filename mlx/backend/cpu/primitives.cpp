// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/arange.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/threefry.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void reshape(const array& in, array& out) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_cpu_inplace(in, out, CopyType::General, out.primitive().stream());
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

static std::pair<array, bool> compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    Stream stream) {
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(allocator::malloc(offset.itemsize()));
  }

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);
  auto compute_offset =
      [strides, axes, offset = offset.data<int64_t>()](const auto* indices) {
        int64_t offset_ = 0;
        for (int i = 0; i < axes.size(); ++i) {
          offset_ += indices[i] * strides[axes[i]];
        }
        offset[0] = offset_;
      };
  switch (indices.dtype()) {
    case int8:
    case uint8:
      encoder.dispatch(compute_offset, indices.data<uint8_t>());
      break;
    case int16:
    case uint16:
      encoder.dispatch(compute_offset, indices.data<uint16_t>());
      break;
    case int32:
    case uint32:
      encoder.dispatch(compute_offset, indices.data<uint32_t>());
      break;
    case int64:
    case uint64:
      encoder.dispatch(compute_offset, indices.data<uint64_t>());
      break;
    default:
      throw std::runtime_error("Invalid indices type.");
  }
  return {offset, donate};
}

void AsStrided::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Broadcast::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void BroadcastAxes::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Copy::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void CustomTransforms::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void Depends::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void ExpandDims::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void NumberOfElements::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Slice::eval_cpu(const std::vector<array>& inputs, array& out) {
  slice(inputs[0], out, start_indices_, strides_);
}
void Split::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void Squeeze::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void StopGradient::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Transpose::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Arange::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc(out.nbytes()));
  switch (out.dtype()) {
    case bool_:
      throw std::runtime_error("Bool type unsupported for arange.");
      break;
    case uint8:
      arange<uint8_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case uint16:
      arange<uint16_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case uint32:
      arange<uint32_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case uint64:
      arange<uint64_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case int8:
      arange<int8_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case int16:
      arange<int16_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case int32:
      arange<int32_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case int64:
      arange<int64_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case float16:
      arange<float16_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case float32:
      arange<float>(start_, start_ + step_, out, out.size(), stream());
      break;
    case float64:
      arange<double>(start_, start_ + step_, out, out.size(), stream());
      break;
    case bfloat16:
      arange<bfloat16_t>(start_, start_ + step_, out, out.size(), stream());
      break;
    case complex64:
      arange<complex64_t>(start_, start_ + step_, out, out.size(), stream());
      break;
  }
}

void AsType::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  CopyType ctype = in.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_cpu(in, out, ctype, stream());
}

void Concatenate::eval_cpu(const std::vector<array>& inputs, array& out) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis_));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis_] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_cpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, stream());
  }
}

void Contiguous::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  constexpr size_t extra_bytes = 16384;
  if (in.buffer_size() <= out.nbytes() + extra_bytes &&
      (in.flags().row_contiguous ||
       (allow_col_major_ && in.flags().col_contiguous))) {
    out.copy_shared_buffer(in);
  } else {
    copy_cpu(in, out, CopyType::General, stream());
  }
}

void Flatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void Unflatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void Full::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  assert(in.dtype() == out.dtype());
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_cpu(in, out, ctype, stream());
}

void Pad::eval_cpu(const std::vector<array>& inputs, array& out) {
  // Inputs must be base input array and scalar val array
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  // Padding value must be a scalar
  assert(val.size() == 1);

  // Padding value, input and output must be of the same type
  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  // Fill output with val
  copy_cpu(val, out, CopyType::Scalar, stream());

  // Find offset for start of input values
  size_t data_offset = 0;
  for (int i = 0; i < axes_.size(); i++) {
    auto ax = axes_[i] < 0 ? out.ndim() + axes_[i] : axes_[i];
    data_offset += out.strides()[ax] * low_pad_size_[i];
  }

  // Extract slice from output where input will be pasted
  array out_slice(in.shape(), out.dtype(), nullptr, {});
  out_slice.copy_shared_buffer(
      out, out.strides(), out.flags(), out_slice.size(), data_offset);

  // Copy input values into the slice
  copy_cpu_inplace(in, out_slice, CopyType::GeneralGeneral, stream());
}

void RandomBits::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc(out.nbytes()));

  auto kptr = inputs[0].data<uint32_t>();
  auto cptr = out.data<char>();
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(inputs[0]);
  encoder.set_output_array(out);
  encoder.dispatch([kptr,
                    cptr,
                    bytes_per_key,
                    num_keys,
                    kshape = keys.shape(),
                    kstrides = keys.strides()]() mutable {
    auto copy_remaining = [&](char* cptr, size_t loc, uint32_t v) {
      if (4 * loc + 4 <= bytes_per_key) {
        reinterpret_cast<uint32_t*>(cptr)[loc] = v;
      } else {
        std::copy(
            reinterpret_cast<char*>(&v),
            reinterpret_cast<char*>(&v) + bytes_per_key - 4 * loc,
            cptr + 4 * loc);
      }
    };

    size_t out_skip = (bytes_per_key + 4 - 1) / 4;
    auto half_size = out_skip / 2;
    bool even = out_skip % 2 == 0;
    for (int i = 0; i < num_keys; ++i, cptr += bytes_per_key) {
      auto ptr = reinterpret_cast<uint32_t*>(cptr);
      // Get ith key
      auto kidx = 2 * i;
      auto k1_elem = elem_to_loc(kidx, kshape, kstrides);
      auto k2_elem = elem_to_loc(kidx + 1, kshape, kstrides);
      auto key = std::make_pair(kptr[k1_elem], kptr[k2_elem]);

      std::pair<uintptr_t, uintptr_t> count{0, half_size + !even};
      for (; count.first + 1 < half_size; count.first++, count.second++) {
        std::tie(ptr[count.first], ptr[count.second]) =
            random::threefry2x32_hash(key, count);
      }
      if (count.first < half_size) {
        auto rb = random::threefry2x32_hash(key, count);
        ptr[count.first++] = rb.first;
        copy_remaining(cptr, count.second, rb.second);
      }
      if (!even) {
        count.second = 0;
        copy_remaining(
            cptr, half_size, random::threefry2x32_hash(key, count).first);
      }
    }
  });
}

void RandomInt::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  auto& keys = inputs[0];
  auto& low_in = inputs[1];
  auto& high_in = inputs[2];

  size_t n = out.size();
  size_t num_keys = keys.size() / 2;
  out.set_data(allocator::malloc(out.nbytes()));

  auto kptr = keys.data<uint32_t>();
  const char* lptr = low_in.data<char>();
  const char* hptr = high_in.data<char>();
  char* optr = out.data<char>();

  auto kshape = keys.shape();
  auto kstrides = keys.strides();
  auto lshape = low_in.shape();
  auto lstrides = low_in.strides();
  auto hshape = high_in.shape();
  auto hstrides = high_in.strides();
  auto oshape = out.shape();
  auto ostrides = out.strides();

  auto dtype = out.dtype();
  size_t out_itemsize = out.itemsize();
  size_t bounds_itemsize = low_in.itemsize();
  auto bounds_dtype_val = low_in.dtype().val();
  bool signed_output = issubdtype(dtype, signedinteger);

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(keys);
  encoder.set_input_array(low_in);
  encoder.set_input_array(high_in);
  encoder.set_output_array(out);

  encoder.dispatch(
      [kptr, lptr, hptr, optr,
       kshape, kstrides,
       lshape, lstrides,
       hshape, hstrides,
       oshape, ostrides,
       dtype, n, num_keys,
       out_itemsize, bounds_itemsize,
       bounds_dtype_val, signed_output]() mutable {
        /* Read bound at oidx as uint64. Bounds are always int64 or uint64. */
        auto get_bound = [&](const char* ptr, const Shape& s,
                             const Strides& st, uint64_t oidx) -> uint64_t {
          auto byte_loc = elem_to_loc(static_cast<int>(oidx), s, st) * bounds_itemsize;
          if (bounds_dtype_val == Dtype::Val::int64) {
            return static_cast<uint64_t>(
                *reinterpret_cast<const int64_t*>(ptr + byte_loc));
          } else {
            return *reinterpret_cast<const uint64_t*>(ptr + byte_loc);
          }
        };

        /* Store a value at oidx. For uint64 output we must NOT pass
         * through int64_t as that truncates the top bit. */
        auto store_signed = [&](uint64_t oidx, int64_t val) {
          auto byte_loc = elem_to_loc(oidx, oshape, ostrides) * out_itemsize;
          switch (dtype.val()) {
            case Dtype::Val::int8:
              reinterpret_cast<int8_t*>(optr + byte_loc)[0] =
                  static_cast<int8_t>(val);
              break;
            case Dtype::Val::int16:
              reinterpret_cast<int16_t*>(optr + byte_loc)[0] =
                  static_cast<int16_t>(val);
              break;
            case Dtype::Val::int32:
              reinterpret_cast<int32_t*>(optr + byte_loc)[0] =
                  static_cast<int32_t>(val);
              break;
            case Dtype::Val::int64:
              reinterpret_cast<int64_t*>(optr + byte_loc)[0] = val;
              break;
            default:
              break;
          }
        };
        auto store_unsigned = [&](uint64_t oidx, uint64_t val) {
          auto byte_loc = elem_to_loc(oidx, oshape, ostrides) * out_itemsize;
          switch (dtype.val()) {
            case Dtype::Val::bool_:
              reinterpret_cast<bool*>(optr + byte_loc)[0] =
                  static_cast<bool>(val);
              break;
            case Dtype::Val::uint8:
              reinterpret_cast<uint8_t*>(optr + byte_loc)[0] =
                  static_cast<uint8_t>(val);
              break;
            case Dtype::Val::uint16:
              reinterpret_cast<uint16_t*>(optr + byte_loc)[0] =
                  static_cast<uint16_t>(val);
              break;
            case Dtype::Val::uint32:
              reinterpret_cast<uint32_t*>(optr + byte_loc)[0] =
                  static_cast<uint32_t>(val);
              break;
            case Dtype::Val::uint64:
              reinterpret_cast<uint64_t*>(optr + byte_loc)[0] = val;
              break;
            default:
              break;
          }
        };

        for (uint64_t oidx = 0; oidx < n; ++oidx) {
          /* Map output element to key index. */
          size_t elems_per_key = n / num_keys;
          size_t key_idx = oidx / elems_per_key;

          auto k1_elem = elem_to_loc(2 * key_idx, kshape, kstrides);
          auto k2_elem = elem_to_loc(2 * key_idx + 1, kshape, kstrides);
          auto key =
              std::make_pair(kptr[k1_elem], kptr[k2_elem]);

          uint64_t lo_u = get_bound(lptr, lshape, lstrides, oidx);
          uint64_t hi_u = get_bound(hptr, hshape, hstrides, oidx);

          /* Compute width. For signed output, reinterpret bounds as int64,
           * then compute width as the unsigned distance. This works because
           * in two's complement, reinterpret-casting int64 to uint64 and
           * subtracting gives the correct element count even when the
           * signed subtraction would overflow.
           * Example: INT64_MIN..INT64_MAX:
           *   lo_s = -2^63, hi_s = 2^63-1
           *   lo_u = 2^63, hi_u = 2^63-1
           *   width = hi_u - lo_u = (2^63-1) - 2^63 (mod 2^64) = 2^64-1
           *   which equals INT64_MAX - INT64_MIN + 1 (the full count). */

          if (signed_output) {
            int64_t lo_s = static_cast<int64_t>(lo_u);
            int64_t hi_s = static_cast<int64_t>(hi_u);
            if (hi_s <= lo_s) {
              store_signed(oidx, lo_s);
              continue;
            }
            uint64_t width = hi_u - lo_u;
            if (width == 1) {
              store_signed(oidx, lo_s);
              continue;
            }

            /* Generate random offset in [0, width). */
            uint64_t result = 0;
            auto count = std::make_pair<uint32_t, uint32_t>(
                static_cast<uint32_t>(oidx),
                static_cast<uint32_t>(oidx >> 32));

            if (width <= UINT32_MAX) {
              uint32_t uwidth = static_cast<uint32_t>(width);
              uint32_t remainder = -uwidth % uwidth;
              while (true) {
                auto rb = random::threefry2x32_hash(key, count);
                count.first++;
                count.second++;
                if (rb.first >= remainder) {
                  result = rb.first % uwidth;
                  break;
                }
              }
            } else {
              uint64_t uwidth = width;
              uint64_t remainder = -uwidth % uwidth;
              while (true) {
                auto rb = random::threefry2x32_hash(key, count);
                count.first++;
                count.second++;
                uint64_t sample =
                    static_cast<uint64_t>(rb.first) |
                    (static_cast<uint64_t>(rb.second) << 32);
                if (sample >= remainder) {
                  result = sample % uwidth;
                  break;
                }
              }
            }

            /* Form result in uint64 to avoid signed overflow.
             * lo_u + result wraps correctly in two's complement. */
            uint64_t out_u = lo_u + result;
            store_signed(oidx, static_cast<int64_t>(out_u));
          } else {
            /* Unsigned output. */
            if (hi_u <= lo_u) {
              store_unsigned(oidx, lo_u);
              continue;
            }
            uint64_t width = hi_u - lo_u;
            if (width == 1) {
              store_unsigned(oidx, lo_u);
              continue;
            }

            /* Generate random offset in [0, width). */
            uint64_t result = 0;
            auto count = std::make_pair<uint32_t, uint32_t>(
                static_cast<uint32_t>(oidx),
                static_cast<uint32_t>(oidx >> 32));

            if (dtype.val() == Dtype::Val::bool_) {
              auto rb = random::threefry2x32_hash(key, count);
              result = rb.first & 1;
            } else if (width <= UINT32_MAX) {
              uint32_t uwidth = static_cast<uint32_t>(width);
              uint32_t remainder = -uwidth % uwidth;
              while (true) {
                auto rb = random::threefry2x32_hash(key, count);
                count.first++;
                count.second++;
                if (rb.first >= remainder) {
                  result = rb.first % uwidth;
                  break;
                }
              }
            } else {
              uint64_t uwidth = width;
              uint64_t remainder = -uwidth % uwidth;
              while (true) {
                auto rb = random::threefry2x32_hash(key, count);
                count.first++;
                count.second++;
                uint64_t sample =
                    static_cast<uint64_t>(rb.first) |
                    (static_cast<uint64_t>(rb.second) << 32);
                if (sample >= remainder) {
                  result = sample % uwidth;
                  break;
                }
              }
            }

            store_unsigned(oidx, lo_u + result);
          }
        }
      });
}

void Reshape::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void DynamicSlice::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto [in_offset, donated] =
      compute_dynamic_offset(inputs[1], in.strides(), axes_, stream());
  copy_cpu_inplace(
      /* const array& src = */ in,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ out.shape(),
      /* const Strides& i_strides = */ in.strides(),
      /* const Strides& o_strides = */ out.strides(),
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ 0,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      stream(),
      /* const std::optional<array>& dynamic_i_offset = */ in_offset,
      /* const std::optional<array>& dynamic_o_offset = */ std::nullopt);
  if (!donated) {
    cpu::get_command_encoder(stream()).add_temporary(std::move(in_offset));
  }
}

void DynamicSliceUpdate::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  // Copy or move src to dst
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_cpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  auto [out_offset, donated] =
      compute_dynamic_offset(inputs[2], out.strides(), axes_, stream());
  copy_cpu_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd.strides(),
      /* const std::vector<stride_t>& o_strides = */ out.strides(),
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ 0,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      stream(),
      /* const std::optional<array>& dynamic_i_offset = */ std::nullopt,
      /* const std::optional<array>& dynamic_o_offset = */ out_offset);
  if (!donated) {
    cpu::get_command_encoder(stream()).add_temporary(std::move(out_offset));
  }
}

void View::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  auto ibytes = size_of(in.dtype());
  auto obytes = size_of(out.dtype());
  // Conditions for buffer copying (disjunction):
  // - type size is the same
  // - type size is smaller and the last axis is contiguous
  // - the entire array is row contiguous
  if (ibytes == obytes || (obytes < ibytes && in.strides().back() == 1) ||
      in.flags().row_contiguous) {
    auto strides = in.strides();
    for (int i = 0; i < static_cast<int>(strides.size()) - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    out.copy_shared_buffer(
        in, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(
        in.shape(), in.dtype() == bool_ ? uint8 : in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    if (in.dtype() == bool_) {
      auto in_tmp = array(in.shape(), uint8, nullptr, {});
      in_tmp.copy_shared_buffer(in);
      copy_cpu_inplace(in_tmp, tmp, CopyType::General, stream());
    } else {
      copy_cpu_inplace(in, tmp, CopyType::General, stream());
    }

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.copy_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

} // namespace mlx::core
