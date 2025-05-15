// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
void eig_impl(
    array& a,
    array& vectors,
    array& values,
    bool compute_eigenvectors,
    Stream stream) {
  using OT = std::complex<T>;
  auto a_ptr = a.data<T>();
  auto eig_ptr = values.data<OT>();

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(values);
  OT* vec_ptr = nullptr;
  if (compute_eigenvectors) {
    encoder.set_output_array(vectors);
    vec_ptr = vectors.data<OT>();
  }
  encoder.dispatch([a_ptr,
                    vec_ptr,
                    eig_ptr,
                    compute_eigenvectors,
                    N = vectors.shape(-1),
                    size = vectors.size()]() mutable {
    // Work query
    char jobr = 'N';
    char jobl = compute_eigenvectors ? 'V' : 'N';
    int n_vecs_r = 1;
    int n_vecs_l = compute_eigenvectors ? N : 1;
    int lwork = -1;
    int info;
    {
      T work;
      int iwork;
      geev<T>(
          &jobl,
          &jobr,
          &N,
          nullptr,
          &N,
          nullptr,
          nullptr,
          nullptr,
          &n_vecs_l,
          nullptr,
          &n_vecs_r,
          &work,
          &lwork,
          &info);
      lwork = static_cast<int>(work);
    }

    auto eig_tmp_data = array::Data{allocator::malloc(sizeof(T) * N * 2)};
    auto vec_tmp_data =
        array::Data{allocator::malloc(vec_ptr ? sizeof(T) * N * N * 2 : 0)};
    auto eig_tmp = static_cast<T*>(eig_tmp_data.buffer.raw_ptr());
    auto vec_tmp = static_cast<T*>(vec_tmp_data.buffer.raw_ptr());
    auto work_buf = array::Data{allocator::malloc(sizeof(T) * lwork)};
    for (size_t i = 0; i < size / (N * N); ++i) {
      geev<T>(
          &jobl,
          &jobr,
          &N,
          a_ptr,
          &N,
          eig_tmp,
          eig_tmp + N,
          vec_tmp,
          &n_vecs_l,
          nullptr,
          &n_vecs_r,
          static_cast<T*>(work_buf.buffer.raw_ptr()),
          &lwork,
          &info);
      for (int i = 0; i < N; ++i) {
        eig_ptr[i] = {eig_tmp[i], eig_tmp[N + i]};
      }
      if (vec_ptr) {
        for (int i = 0; i < N; ++i) {
          if (eig_ptr[i].imag() != 0) {
            // This vector and the next are a pair
            for (int j = 0; j < N; ++j) {
              vec_ptr[i * N + j] = {
                  vec_tmp[i * N + j], -vec_tmp[(i + 1) * N + j]};
              vec_ptr[(i + 1) * N + j] = {
                  vec_tmp[i * N + j], vec_tmp[(i + 1) * N + j]};
            }
            i += 1;
          } else {
            for (int j = 0; j < N; ++j) {
              vec_ptr[i * N + j] = {vec_tmp[i * N + j], 0};
            }
          }
        }
        vec_ptr += N * N;
      }
      a_ptr += N * N;
      eig_ptr += N;
      if (info != 0) {
        std::stringstream msg;
        msg << "[Eig::eval_cpu] Eigenvalue decomposition failed with error code "
            << info;
        throw std::runtime_error(msg.str());
      }
    }
  });
  encoder.add_temporary(a);
}

} // namespace

void Eig::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& a = inputs[0];
  auto& values = outputs[0];

  auto vectors = compute_eigenvectors_
      ? outputs[1]
      : array(a.shape(), complex64, nullptr, {});

  auto a_copy = array(a.shape(), a.dtype(), nullptr, {});
  copy(
      a,
      a_copy,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream());

  values.set_data(allocator::malloc(values.nbytes()));

  if (compute_eigenvectors_) {
    // Set the strides and flags so the eigenvectors
    // are in the columns of the output
    auto flags = vectors.flags();
    auto strides = vectors.strides();
    auto ndim = a.ndim();
    std::swap(strides[ndim - 1], strides[ndim - 2]);

    if (a.size() > 1) {
      flags.row_contiguous = false;
      if (ndim > 2) {
        flags.col_contiguous = false;
      } else {
        flags.col_contiguous = true;
      }
    }
    vectors.set_data(
        allocator::malloc(vectors.nbytes()), vectors.size(), strides, flags);
  }
  switch (a.dtype()) {
    case float32:
      eig_impl<float>(a_copy, vectors, values, compute_eigenvectors_, stream());
      break;
    default:
      throw std::runtime_error("[Eig::eval_cpu] only supports float32.");
  }
}

} // namespace mlx::core
