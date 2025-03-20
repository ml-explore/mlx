// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void qrf_impl(const array& a, array& q, array& r, Stream stream) {
  const int M = a.shape(-2);
  const int N = a.shape(-1);
  const int lda = M;
  size_t num_matrices = a.size() / (M * N);

  // Copy A to inplace input and make it col-contiguous
  array in(a.shape(), a.dtype(), nullptr, {});
  auto flags = in.flags();

  // Copy the input to be column contiguous
  flags.col_contiguous = num_matrices == 1;
  flags.row_contiguous = false;
  auto strides = in.strides();
  strides[in.ndim() - 2] = 1;
  strides[in.ndim() - 1] = M;
  in.set_data(allocator::malloc(in.nbytes()), in.nbytes(), strides, flags);
  copy_inplace(a, in, CopyType::GeneralGeneral, stream);
  auto& encoder = cpu::get_command_encoder(stream);
  q.set_data(allocator::malloc(q.nbytes()));
  r.set_data(allocator::malloc(r.nbytes()));

  auto in_ptr = in.data<T>();
  auto r_ptr = r.data<T>();
  auto q_ptr = q.data<T>();

  encoder.set_input_array(in);
  encoder.set_output_array(q);
  encoder.set_output_array(r);
  encoder.dispatch([in_ptr, q_ptr, r_ptr, M, N, lda, num_matrices]() {
    int num_reflectors = std::min(M, N);
    auto tau = allocator::malloc(sizeof(T) * num_matrices * num_reflectors);

    T optimal_work;
    int lwork = -1;
    int info;

    // Compute workspace size
    geqrf<T>(&M, &N, nullptr, &lda, nullptr, &optimal_work, &lwork, &info);

    // Update workspace size
    lwork = optimal_work;
    auto work = allocator::malloc(sizeof(T) * lwork);

    // Loop over matrices
    for (int i = 0; i < num_matrices; ++i) {
      // Solve
      geqrf<T>(
          &M,
          &N,
          in_ptr + M * N * i,
          &lda,
          static_cast<T*>(tau.raw_ptr()) + num_reflectors * i,
          static_cast<T*>(work.raw_ptr()),
          &lwork,
          &info);
    }
    allocator::free(work);

    for (int i = 0; i < num_matrices; ++i) {
      /// num_reflectors x N
      for (int j = 0; j < num_reflectors; ++j) {
        for (int k = 0; k < j; ++k) {
          r_ptr[i * N * num_reflectors + j * N + k] = 0;
        }
        for (int k = j; k < N; ++k) {
          r_ptr[i * N * num_reflectors + j * N + k] =
              in_ptr[i * N * M + j + k * M];
        }
      }
    }

    // Get work size
    lwork = -1;
    orgqr<T>(
        &M,
        &num_reflectors,
        &num_reflectors,
        nullptr,
        &lda,
        nullptr,
        &optimal_work,
        &lwork,
        &info);
    lwork = optimal_work;
    work = allocator::malloc(sizeof(T) * lwork);

    // Loop over matrices
    for (int i = 0; i < num_matrices; ++i) {
      // Compute Q
      orgqr<T>(
          &M,
          &num_reflectors,
          &num_reflectors,
          in_ptr + M * N * i,
          &lda,
          static_cast<T*>(tau.raw_ptr()) + num_reflectors * i,
          static_cast<T*>(work.raw_ptr()),
          &lwork,
          &info);
    }

    for (int i = 0; i < num_matrices; ++i) {
      // M x num_reflectors
      for (int j = 0; j < M; ++j) {
        for (int k = 0; k < num_reflectors; ++k) {
          q_ptr[i * M * num_reflectors + j * num_reflectors + k] =
              in_ptr[i * N * M + j + k * M];
        }
      }
    }

    // Cleanup
    allocator::free(work);
    allocator::free(tau);
  });
  encoder.add_temporary(in);
}

void QRF::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  switch (inputs[0].dtype()) {
    case float32:
      qrf_impl<float>(inputs[0], outputs[0], outputs[1], stream());
      break;
    case float64:
      qrf_impl<double>(inputs[0], outputs[0], outputs[1], stream());
      break;
    default:
      throw std::runtime_error(
          "[QRF::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
