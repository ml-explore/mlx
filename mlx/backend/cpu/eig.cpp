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
complex64_t to_complex(T r, T i) {
  return {static_cast<float>(r), static_cast<float>(i)};
}

template <typename T, class Enable = void>
struct EigWork {};

template <typename T>
struct EigWork<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using O = complex64_t;

  char jobl;
  char jobr;
  int N;
  int lwork;
  int info;
  std::vector<array::Data> buffers;

  EigWork(char jobl_, char jobr_, int N_, bool compute_eigenvectors)
      : jobl(jobl_), jobr(jobr_), N(N_), lwork(-1) {
    T work;
    int n_vecs_l = compute_eigenvectors ? N_ : 1;
    int n_vecs_r = 1;
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

    buffers.emplace_back(allocator::malloc(sizeof(T) * N * 2));
    if (compute_eigenvectors) {
      buffers.emplace_back(allocator::malloc(sizeof(T) * N * N * 2));
    }
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
  }

  void run(T* a, O* values, O* vectors) {
    auto eig_tmp = static_cast<T*>(buffers[0].buffer.raw_ptr());
    T* vec_tmp = nullptr;
    if (vectors) {
      vec_tmp = static_cast<T*>(buffers[1].buffer.raw_ptr());
    }
    auto work = static_cast<T*>(buffers.back().buffer.raw_ptr());

    int n_vecs_l = vectors ? N : 1;
    int n_vecs_r = 1;
    geev<T>(
        &jobl,
        &jobr,
        &N,
        a,
        &N,
        eig_tmp,
        eig_tmp + N,
        vectors ? vec_tmp : nullptr,
        &n_vecs_l,
        nullptr,
        &n_vecs_r,
        work,
        &lwork,
        &info);

    for (int i = 0; i < N; ++i) {
      values[i] = to_complex(eig_tmp[i], eig_tmp[N + i]);
    }

    if (vectors) {
      for (int i = 0; i < N; ++i) {
        if (values[i].imag() != 0) {
          for (int j = 0; j < N; ++j) {
            vectors[i * N + j] =
                to_complex(vec_tmp[i * N + j], -vec_tmp[(i + 1) * N + j]);
            vectors[(i + 1) * N + j] =
                to_complex(vec_tmp[i * N + j], vec_tmp[(i + 1) * N + j]);
          }
          i += 1;
        } else {
          for (int j = 0; j < N; ++j) {
            vectors[i * N + j] = to_complex(vec_tmp[i * N + j], T(0.0));
          }
        }
      }
    }
  }
};

template <>
struct EigWork<std::complex<float>> {
  using T = std::complex<float>;
  using R = float;
  using O = T;

  char jobl;
  char jobr;
  int N;
  int lwork;
  int lrwork;
  int info;
  std::vector<array::Data> buffers;

  EigWork(char jobl_, char jobr_, int N_, bool compute_eigenvectors)
      : jobl(jobl_), jobr(jobr_), N(N_), lwork(-1), lrwork(2 * N_) {
    T work;
    R rwork;
    int n_vecs_l = compute_eigenvectors ? N_ : 1;
    int n_vecs_r = 1;
    geev<T>(
        &jobl,
        &jobr,
        &N,
        nullptr,
        &N,
        nullptr,
        nullptr,
        &n_vecs_l,
        nullptr,
        &n_vecs_r,
        &work,
        &lwork,
        &rwork,
        &info);
    lwork = static_cast<int>(work.real());
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
    buffers.emplace_back(allocator::malloc(sizeof(R) * lrwork));
  }

  void run(T* a, T* values, T* vectors) {
    int n_vecs_l = vectors ? N : 1;
    int n_vecs_r = 1;
    geev<T>(
        &jobl,
        &jobr,
        &N,
        a,
        &N,
        values,
        vectors,
        &n_vecs_l,
        nullptr,
        &n_vecs_r,
        static_cast<T*>(buffers[0].buffer.raw_ptr()),
        &lwork,
        static_cast<R*>(buffers[1].buffer.raw_ptr()),
        &info);
  }
};

template <typename T>
void eig_impl(
    array& a,
    array& vectors,
    array& values,
    bool compute_eigenvectors,
    Stream stream) {
  auto a_ptr = a.data<T>();
  auto val_ptr = values.data<complex64_t>();

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(values);
  complex64_t* vec_ptr = nullptr;
  if (compute_eigenvectors) {
    encoder.set_output_array(vectors);
    vec_ptr = vectors.data<complex64_t>();
  }
  encoder.dispatch([a_ptr,
                    val_ptr,
                    vec_ptr,
                    compute_eigenvectors,
                    N = vectors.shape(-1),
                    size = vectors.size()]() mutable {
    char jobr = 'N';
    char jobl = compute_eigenvectors ? 'V' : 'N';

    EigWork<T> work(jobl, jobr, N, compute_eigenvectors);

    for (size_t i = 0; i < size / (N * N); ++i) {
      work.run(a_ptr, val_ptr, vec_ptr);
      a_ptr += N * N;
      val_ptr += N;
      if (vec_ptr) {
        vec_ptr += N * N;
      }
      if (work.info != 0) {
        std::stringstream msg;
        msg << "[Eig::eval_cpu] Eigenvalue decomposition failed with error code "
            << work.info;
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
  copy_cpu(
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
    case float64:
      eig_impl<double>(
          a_copy, vectors, values, compute_eigenvectors_, stream());
      break;
    case complex64:
      eig_impl<std::complex<float>>(
          a_copy, vectors, values, compute_eigenvectors_, stream());
      break;
    default:
      throw std::runtime_error(
          "[Eig::eval_cpu] only supports float32, float64, or complex64.");
  }
}

} // namespace mlx::core
