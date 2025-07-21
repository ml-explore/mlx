// Copyright © 2024 Apple Inc.

#include <cstring>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename mask_t>
inline void mask_matrix(
    T* data,
    const mask_t* mask,
    int block_size,
    const int X,
    const int Y,
    const int64_t X_data_str,
    const int64_t Y_data_str,
    const int64_t X_mask_str,
    const int64_t Y_mask_str,
    const size_t mask_offset) {
  int tX = (X + block_size - 1) / block_size;
  int tY = (Y + block_size - 1) / block_size;

  for (int i = 0; i < tX; i++) {
    for (int j = 0; j < tY; j++) {
      mask_t do_mask = mask[mask_offset + i * X_mask_str + j * Y_mask_str];
      if (do_mask != 1) {
        int loc_x = i * block_size;
        int loc_y = j * block_size;
        T* data_block = data + loc_x * X_data_str + loc_y * Y_data_str;

        int size_x = std::min(block_size, X - loc_x);
        int size_y = std::min(block_size, Y - loc_y);
        for (int ii = 0; ii < size_x; ii++) {
          for (int jj = 0; jj < size_y; jj++) {
            if constexpr (std::is_same_v<mask_t, bool>) {
              data_block[ii * X_data_str + jj * Y_data_str] = T(0.);
            } else {
              data_block[ii * X_data_str + jj * Y_data_str] *= do_mask;
            }
          }
        }
      }
    }
  }
}

template <typename T>
inline void segmented_mm(
    const T* a,
    const T* b,
    const uint32_t* segments,
    T* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides,
    size_t num_segments,
    const Shape& segments_shape,
    const Strides& segments_strides) {
  int ndim = a_shape.size();
  Shape a_copy = a_shape;
  Shape b_copy = b_shape;
  int32_t M = a_copy[ndim - 2];
  int32_t N = b_copy[ndim - 1];
  for (int i = 0; i < num_segments; i++) {
    uint32_t k_start =
        segments[elem_to_loc(2 * i, segments_shape, segments_strides)];
    uint32_t k_end =
        segments[elem_to_loc(2 * i + 1, segments_shape, segments_strides)];
    if (k_end <= k_start) {
      std::fill_n(out + i * M * N, M * N, T(0));
      continue;
    }
    a_copy[ndim - 1] = k_end - k_start;
    b_copy[ndim - 2] = k_end - k_start;
    matmul<T>(
        a + k_start * a_strides[ndim - 1],
        b + k_start * b_strides[ndim - 2],
        out + i * M * N,
        a_transposed,
        b_transposed,
        lda,
        ldb,
        N,
        1.0,
        0.0,
        1,
        a_copy,
        a_strides,
        b_copy,
        b_strides);
  }
}

} // namespace

void BlockMaskedMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[BlockMaskedMM::eval] Currently only supports float32.");
  }
  out.set_data(allocator::malloc(out.nbytes()));

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  auto check_transpose =
      [s = stream()](const array& arr, bool do_copy, bool expand_all = false) {
        auto stx = arr.strides()[arr.ndim() - 2];
        auto sty = arr.strides()[arr.ndim() - 1];
        if (!expand_all && stx == arr.shape(-1) && sty == 1) {
          if (do_copy) {
            array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
            copy_cpu(arr, arr_copy, CopyType::Vector, s);
            return std::make_tuple(false, stx, arr_copy, true);
          }
          return std::make_tuple(false, stx, arr, false);
        } else if (!expand_all && stx == 1 && sty == arr.shape(-2)) {
          if (do_copy) {
            array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
            copy_cpu(arr, arr_copy, CopyType::Vector, s);
            return std::make_tuple(true, sty, arr_copy, true);
          }
          return std::make_tuple(true, sty, arr, false);
        } else {
          int64_t stx = arr.shape(-1);
          array arr_copy = contiguous_copy_cpu(arr, s);
          return std::make_tuple(false, stx, arr_copy, true);
        }
      };

  bool has_op_mask = inputs.size() > 3;
  bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;
  auto [a_transposed, lda, a, a_copied] =
      check_transpose(a_pre, has_op_mask, inputs.back().dtype() != bool_);
  auto [b_transposed, ldb, b, b_copied] =
      check_transpose(b_pre, has_op_mask, inputs.back().dtype() != bool_);

  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  if (M == 0 || N == 0) {
    return;
  }

  auto& encoder = cpu::get_command_encoder(stream());
  if (K == 0) {
    encoder.set_output_array(out);
    encoder.dispatch([out_ptr = out.data<void>(), nbytes = out.nbytes()]() {
      std::memset(out_ptr, 0, nbytes);
    });
    return;
  }

  auto mask_array = [](const void* mask,
                       float* data,
                       int block_size,
                       int batch_idx,
                       int X,
                       int Y,
                       size_t X_data_str,
                       size_t Y_data_str,
                       const Shape& mask_shape,
                       const Strides& mask_strides,
                       bool is_bool) {
    auto ndim = mask_shape.size();
    auto mask_offset = elem_to_loc(
        mask_shape[ndim - 1] * mask_shape[ndim - 2] * batch_idx,
        mask_shape,
        mask_strides);

    auto X_mask_str = mask_strides[ndim - 2];
    auto Y_mask_str = mask_strides[ndim - 1];

    if (is_bool) {
      return mask_matrix(
          data,
          static_cast<const bool*>(mask),
          block_size,
          X,
          Y,
          X_data_str,
          Y_data_str,
          X_mask_str,
          Y_mask_str,
          mask_offset);
    } else {
      return mask_matrix(
          data,
          static_cast<const float*>(mask),
          block_size,
          X,
          Y,
          X_data_str,
          Y_data_str,
          X_mask_str,
          Y_mask_str,
          mask_offset);
    }
  };

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  const void* a_mask_ptr;
  const void* b_mask_ptr;
  const void* out_mask_ptr;
  Shape a_mask_shape;
  Shape b_mask_shape;
  Shape out_mask_shape;
  Strides a_mask_strides;
  Strides b_mask_strides;
  Strides out_mask_strides;
  bool a_mask_bool;
  bool b_mask_bool;
  bool out_mask_bool;
  if (has_op_mask) {
    auto& a_mask = inputs[inputs.size() - 2];
    auto& b_mask = inputs[inputs.size() - 1];
    a_mask_ptr = a_mask.data<void>();
    b_mask_ptr = b_mask.data<void>();
    a_mask_shape = a_mask.shape();
    b_mask_shape = b_mask.shape();
    a_mask_strides = a_mask.strides();
    b_mask_strides = b_mask.strides();
    a_mask_bool = (a_mask.dtype() == bool_);
    b_mask_bool = (b_mask.dtype() == bool_);
    encoder.set_input_array(a_mask);
    encoder.set_input_array(b_mask);
  }
  if (has_out_mask) {
    auto& out_mask = inputs[2];
    out_mask_ptr = out_mask.data<void>();
    out_mask_bool = (out_mask.dtype() == bool_);
    encoder.set_input_array(out_mask);
    out_mask_shape = out_mask.shape();
    out_mask_strides = out_mask.strides();
  }
  encoder.set_output_array(out);
  auto a_ptr = a.data<float>();
  auto b_ptr = b.data<float>();
  auto out_ptr = out.data<float>();
  size_t num_matrices = out.size() / (M * size_t(N));
  auto ldc = out.shape(-1);

  encoder.dispatch([a_ptr,
                    b_ptr,
                    out_ptr,
                    a_mask_ptr,
                    b_mask_ptr,
                    out_mask_ptr,
                    has_op_mask,
                    has_out_mask,
                    block_size = block_size_,
                    num_matrices,
                    M,
                    N,
                    K,
                    a_transposed = a_transposed,
                    b_transposed = b_transposed,
                    lda = lda,
                    ldb = ldb,
                    ldc,
                    a_shape = a.shape(),
                    a_strides = a.strides(),
                    b_shape = b.shape(),
                    b_strides = b.strides(),
                    a_mask_shape = std::move(a_mask_shape),
                    b_mask_shape = std::move(b_mask_shape),
                    out_mask_shape = std::move(out_mask_shape),
                    a_mask_strides = std::move(a_mask_strides),
                    b_mask_strides = std::move(b_mask_strides),
                    out_mask_strides = std::move(out_mask_strides),
                    mask_array,
                    a_mask_bool,
                    b_mask_bool,
                    out_mask_bool]() {
    for (int i = 0; i < num_matrices; ++i) {
      // Adjust pointer
      float* ai = a_ptr + elem_to_loc(M * K * i, a_shape, a_strides);
      float* bi = b_ptr + elem_to_loc(K * N * i, b_shape, b_strides);
      float* ci = out_ptr + M * N * i;

      // Zero out blocks in a and b if needed
      if (has_op_mask) {
        mask_array(
            a_mask_ptr,
            ai,
            block_size,
            i,
            M,
            K,
            a_transposed ? 1 : lda,
            a_transposed ? lda : 1,
            a_mask_shape,
            a_mask_strides,
            a_mask_bool);

        mask_array(
            b_mask_ptr,
            bi,
            block_size,
            i,
            K,
            N,
            b_transposed ? 1 : ldb,
            b_transposed ? ldb : 1,
            b_mask_shape,
            b_mask_strides,
            b_mask_bool);
      }

      // Do matmul
      cblas_sgemm(
          CblasRowMajor,
          a_transposed ? CblasTrans : CblasNoTrans, // transA
          b_transposed ? CblasTrans : CblasNoTrans, // transB
          M,
          N,
          K,
          1.0, // alpha
          ai,
          lda,
          bi,
          ldb,
          0.0, // beta
          ci,
          ldc);

      // Zero out blocks in out
      if (has_out_mask) {
        mask_array(
            out_mask_ptr,
            ci,
            block_size,
            i,
            M,
            N,
            N,
            1,
            out_mask_shape,
            out_mask_strides,
            out_mask_bool);
      }
    }
  });
  if (a_copied) {
    encoder.add_temporary(a);
  }
  if (b_copied) {
    encoder.add_temporary(b);
  }
}

void GatherMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[GatherMM::eval] Currently only supports float32.");
  }
  out.set_data(allocator::malloc(out.nbytes()));

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  std::vector<array> temps;
  auto check_transpose = [s = stream(), &temps](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      temps.push_back(array(arr.shape(), arr.dtype(), nullptr, {}));
      copy_cpu(arr, temps.back(), CopyType::General, s);
      int64_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, temps.back());
    }
  };

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);

  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  if (M == 0 || N == 0) {
    return;
  }

  auto& encoder = cpu::get_command_encoder(stream());
  if (K == 0) {
    encoder.set_output_array(out);
    encoder.dispatch([out_ptr = out.data<float>(), size = out.size()]() {
      std::fill_n(out_ptr, size, 0);
    });
    return;
  }

  // Get batch dims
  auto batch_size_out = out.size() / (M * N);
  size_t matrix_stride_out = M * N;

  auto get_batch_dims = [](const auto& v) {
    return decltype(v){v.begin(), v.end() - 2};
  };

  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  auto batch_shape = get_batch_dims(out.shape());
  int batch_ndim = batch_shape.size();

  auto batch_shape_A = get_batch_dims(a.shape());
  auto batch_strides_A = get_batch_dims(a.strides());
  auto batch_shape_B = get_batch_dims(b.shape());
  auto batch_strides_B = get_batch_dims(b.strides());

  const uint32_t* lhs_indices_ptr = lhs_indices.data<uint32_t>();
  const uint32_t* rhs_indices_ptr = rhs_indices.data<uint32_t>();
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);
  auto ldc = out.shape(-1);

  encoder.dispatch([a_ptr = a.data<float>(),
                    b_ptr = b.data<float>(),
                    out_ptr = out.data<float>(),
                    M,
                    N,
                    K,
                    lda = lda,
                    ldb = ldb,
                    a_transposed = a_transposed,
                    b_transposed = b_transposed,
                    ldc,
                    lhs_indices_ptr,
                    rhs_indices_ptr,
                    lhs_indices_shape = lhs_indices.shape(),
                    lhs_indices_strides = lhs_indices.strides(),
                    rhs_indices_shape = rhs_indices.shape(),
                    rhs_indices_strides = rhs_indices.strides(),
                    batch_size_out,
                    matrix_stride_out,
                    batch_shape_A = std::move(batch_shape_A),
                    batch_shape_B = std::move(batch_shape_B),
                    batch_strides_A = std::move(batch_strides_A),
                    batch_strides_B = std::move(batch_strides_B)]() {
    for (int i = 0; i < batch_size_out; i++) {
      // Get index
      uint32_t indx_A = lhs_indices_ptr[elem_to_loc(
          i, lhs_indices_shape, lhs_indices_strides)];
      uint32_t indx_B = rhs_indices_ptr[elem_to_loc(
          i, rhs_indices_shape, rhs_indices_strides)];

      cblas_sgemm(
          CblasRowMajor,
          a_transposed ? CblasTrans : CblasNoTrans, // transA
          b_transposed ? CblasTrans : CblasNoTrans, // transB
          M,
          N,
          K,
          1.0f, // alpha
          a_ptr + elem_to_loc(indx_A, batch_shape_A, batch_strides_A),
          lda,
          b_ptr + elem_to_loc(indx_B, batch_shape_B, batch_strides_B),
          ldb,
          0.0f, // beta
          out_ptr + matrix_stride_out * i,
          ldc);
    }
  });
  encoder.add_temporaries(std::move(temps));
}

void SegmentedMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = cpu::get_command_encoder(stream());
  auto check_transpose = [&s, &encoder](const array& x) {
    auto stx = x.strides()[x.ndim() - 2];
    auto sty = x.strides()[x.ndim() - 1];
    if (stx == x.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, x);
    } else if (stx == 1 && sty == x.shape(-2)) {
      return std::make_tuple(true, sty, x);
    } else {
      array xc(x.shape(), x.dtype(), nullptr, {});
      copy_cpu(x, xc, CopyType::General, s);
      encoder.add_temporary(xc);
      int64_t stx = x.shape(-1);
      return std::make_tuple(false, stx, xc);
    }
  };

  auto [a_transposed, lda, a] = check_transpose(inputs[0]);
  auto [b_transposed, ldb, b] = check_transpose(inputs[1]);
  auto& segments = inputs[2];

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(segments);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    segments = array::unsafe_weak_copy(segments),
                    out_ptr = out.data<void>(),
                    a_transposed = a_transposed,
                    b_transposed = b_transposed,
                    lda = lda,
                    ldb = ldb]() {
    switch (a.dtype()) {
      case float64:
        segmented_mm<double>(
            a.data<double>(),
            b.data<double>(),
            segments.data<uint32_t>(),
            static_cast<double*>(out_ptr),
            a_transposed,
            b_transposed,
            lda,
            ldb,
            a.shape(),
            a.strides(),
            b.shape(),
            b.strides(),
            segments.size() / 2,
            segments.shape(),
            segments.strides());
        break;
      case float32:
        segmented_mm<float>(
            a.data<float>(),
            b.data<float>(),
            segments.data<uint32_t>(),
            static_cast<float*>(out_ptr),
            a_transposed,
            b_transposed,
            lda,
            ldb,
            a.shape(),
            a.strides(),
            b.shape(),
            b.strides(),
            segments.size() / 2,
            segments.shape(),
            segments.strides());
        break;
      case float16:
        segmented_mm<float16_t>(
            a.data<float16_t>(),
            b.data<float16_t>(),
            segments.data<uint32_t>(),
            static_cast<float16_t*>(out_ptr),
            a_transposed,
            b_transposed,
            lda,
            ldb,
            a.shape(),
            a.strides(),
            b.shape(),
            b.strides(),
            segments.size() / 2,
            segments.shape(),
            segments.strides());
        break;
      case bfloat16:
        segmented_mm<bfloat16_t>(
            a.data<bfloat16_t>(),
            b.data<bfloat16_t>(),
            segments.data<uint32_t>(),
            static_cast<bfloat16_t*>(out_ptr),
            a_transposed,
            b_transposed,
            lda,
            ldb,
            a.shape(),
            a.strides(),
            b.shape(),
            b.strides(),
            segments.size() / 2,
            segments.shape(),
            segments.strides());
        break;
      default:
        throw std::invalid_argument(
            "Segmented mm supports only real float types.");
    }
  });
}

} // namespace mlx::core
