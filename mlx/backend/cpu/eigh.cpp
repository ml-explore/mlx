// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, class Enable = void>
struct EighWork {};

template <typename T>
struct EighWork<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using R = T;

  char jobz;
  char uplo;
  int N;
  int lwork;
  int liwork;
  int info;
  std::vector<array::Data> buffers;

  EighWork(char jobz_, char uplo_, int N_)
      : jobz(jobz_), uplo(uplo_), N(N_), lwork(-1), liwork(-1) {
    T work;
    int iwork;
    syevd<T>(
        &jobz,
        &uplo,
        &N,
        nullptr,
        &N,
        nullptr,
        &work,
        &lwork,
        &iwork,
        &liwork,
        &info);
    lwork = static_cast<int>(work);
    liwork = iwork;
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
    buffers.emplace_back(allocator::malloc(sizeof(int) * liwork));
  }

  void run(T* vectors, T* values) {
    syevd<T>(
        &jobz,
        &uplo,
        &N,
        vectors,
        &N,
        values,
        static_cast<T*>(buffers[0].buffer.raw_ptr()),
        &lwork,
        static_cast<int*>(buffers[1].buffer.raw_ptr()),
        &liwork,
        &info);
  }
};

template <>
struct EighWork<std::complex<float>> {
  using T = std::complex<float>;
  using R = float;

  char jobz;
  char uplo;
  int N;
  int lwork;
  int lrwork;
  int liwork;
  int info;
  std::vector<array::Data> buffers;

  EighWork(char jobz_, char uplo_, int N_)
      : jobz(jobz_), uplo(uplo_), N(N_), lwork(-1), lrwork(-1), liwork(-1) {
    T work;
    R rwork;
    int iwork;
    heevd<T>(
        &jobz,
        &uplo,
        &N,
        nullptr,
        &N,
        nullptr,
        &work,
        &lwork,
        &rwork,
        &lrwork,
        &iwork,
        &liwork,
        &info);
    lwork = static_cast<int>(work.real());
    lrwork = static_cast<int>(rwork);
    liwork = iwork;
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
    buffers.emplace_back(allocator::malloc(sizeof(R) * lrwork));
    buffers.emplace_back(allocator::malloc(sizeof(int) * liwork));
  }

  void run(T* vectors, R* values) {
    heevd<T>(
        &jobz,
        &uplo,
        &N,
        vectors,
        &N,
        values,
        static_cast<T*>(buffers[0].buffer.raw_ptr()),
        &lwork,
        static_cast<R*>(buffers[1].buffer.raw_ptr()),
        &lrwork,
        static_cast<int*>(buffers[2].buffer.raw_ptr()),
        &liwork,
        &info);
    if (jobz == 'V') {
      // We have pre-transposed the vectors but we also must conjugate them
      // when they are complex.
      //
      // We could vectorize this but it is so fast in comparison to heevd that
      // it doesn't really matter.
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          *vectors = std::conj(*vectors);
          vectors++;
        }
      }
    }
  }
};

template <typename T>
void eigh_impl(
    array& vectors,
    array& values,
    const std::string& uplo,
    bool compute_eigenvectors,
    Stream stream) {
  using R = typename EighWork<T>::R;

  auto vec_ptr = vectors.data<T>();
  auto eig_ptr = values.data<R>();
  char jobz = compute_eigenvectors ? 'V' : 'N';

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(vectors);
  encoder.set_output_array(values);
  encoder.dispatch([vec_ptr,
                    eig_ptr,
                    jobz,
                    uplo = uplo[0],
                    N = vectors.shape(-1),
                    size = vectors.size()]() mutable {
    // Work query
    EighWork<T> work(jobz, uplo, N);

    // Work loop
    for (size_t i = 0; i < size / (N * N); ++i) {
      work.run(vec_ptr, eig_ptr);
      vec_ptr += N * N;
      eig_ptr += N;
      if (work.info != 0) {
        std::stringstream msg;
        msg << "[Eigh::eval_cpu] Eigenvalue decomposition failed with error code "
            << work.info;
        throw std::runtime_error(msg.str());
      }
    }
  });
  if (!compute_eigenvectors) {
    encoder.add_temporary(vectors);
  }
}

} // namespace

void Eigh::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& a = inputs[0];
  auto& values = outputs[0];

  auto vectors = compute_eigenvectors_
      ? outputs[1]
      : array(a.shape(), a.dtype(), nullptr, {});

  values.set_data(allocator::malloc(values.nbytes()));

  copy_cpu(
      a,
      vectors,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream());

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
    vectors.copy_shared_buffer(vectors, strides, flags, vectors.data_size());
  }
  switch (a.dtype()) {
    case float32:
      eigh_impl<float>(vectors, values, uplo_, compute_eigenvectors_, stream());
      break;
    case float64:
      eigh_impl<double>(
          vectors, values, uplo_, compute_eigenvectors_, stream());
      break;
    case complex64:
      eigh_impl<std::complex<float>>(
          vectors, values, uplo_, compute_eigenvectors_, stream());
      break;
    default:
      throw std::runtime_error(
          "[Eigh::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
