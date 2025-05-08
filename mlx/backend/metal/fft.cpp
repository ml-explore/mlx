// Copyright © 2024 Apple Inc.
#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

#include "mlx/3rdparty/pocketfft.h"
#include "mlx/backend/common/transpose.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/metal/binary.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace mlx::core {

using MTLFC = std::tuple<const void*, MTL::DataType, NS::UInteger>;

#define MAX_STOCKHAM_FFT_SIZE 4096
#define MAX_RADER_FFT_SIZE 2048
#define MAX_BLUESTEIN_FFT_SIZE 2048
// Threadgroup memory batching improves throughput for small n
#define MIN_THREADGROUP_MEM_SIZE 256
// For strided reads/writes, coalesce at least this many complex64s
#define MIN_COALESCE_WIDTH 4

inline constexpr std::array<int, 9> supported_radices() {
  // Ordered by preference in decomposition.
  return {13, 11, 8, 7, 6, 5, 4, 3, 2};
}

std::vector<int> prime_factors(int n) {
  int z = 2;
  std::vector<int> factors;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

int next_fast_n(int n) {
  return next_power_of_2(n);
}

std::vector<int> stockham_decompose(int n) {
  auto radices = supported_radices();
  std::vector<int> steps(radices.size(), 0);
  int orig_n = n;

  for (int i = 0; i < radices.size(); i++) {
    int radix = radices[i];

    // Manually tuned radices for powers of 2
    if (is_power_of_2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }

    while (n % radix == 0) {
      steps[i] += 1;
      n /= radix;
      if (n == 1) {
        return steps;
      }
    }
  }

  return {};
}

struct FourStepParams {
  bool required = false;
  bool first_step = true;
  int n1 = 0;
  int n2 = 0;
};

// Forward Declaration
void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    const FourStepParams four_step_params,
    bool inplace,
    metal::Device& d,
    const Stream& s);

struct OldFFTPlan {
  int n = 0;
  // Number of steps for each radix in the Stockham decomposition
  std::vector<int> stockham;
  // Number of steps for each radix in the Rader decomposition
  std::vector<int> rader;
  // Rader factor, 1 if no rader factors
  int rader_n = 1;
  int bluestein_n = -1;
  // Four step FFT
  bool four_step = false;
  int n1 = 0;
  int n2 = 0;
};

class FFTPlan {
 public:
  enum FFTType {
    UNSUPPORTED,
    NOOP,
    STOCKHAM,
    RADER,
    BLUESTEIN,
    SMALL_FOUR_STEP,
    LARGE_FOUR_STEP
  };

  FFTPlan(int n) : n_(n) {
    // NOOP
    if (n == 1) {
      type_ = NOOP;
    }

    // Four step fft
    else if (n > MAX_STOCKHAM_FFT_SIZE && is_power_of_2(n)) {
      if (n <= 1 << 20) {
        type_ = SMALL_FOUR_STEP;
        n2_ = n > 65536 ? 1024 : 64;
        n1_ = n / n2_;
        steps1_ = stockham_decompose(n1_);
        steps2_ = stockham_decompose(n2_);
      } else {
        type_ = LARGE_FOUR_STEP;
      }
    }

    // Bluestein fft
    else if (n > MAX_STOCKHAM_FFT_SIZE) {
      type_ = BLUESTEIN;
      bluestein_n_ = next_fast_n(2 * n - 1);
    }

    // Stockham fft
    else if (auto steps = stockham_decompose(n); steps.size() > 0) {
      type_ = STOCKHAM;
      steps_ = steps;
    }

    // throw for now but we have rader and bluestein to do
    else {
      type_ = UNSUPPORTED;
    }
  }

  FFTType type() const {
    return type_;
  }

  int size() const {
    return n_;
  }

  const std::vector<int>& steps() const {
    return steps_;
  }

  int first_size() const {
    return n1_;
  }

  const std::vector<int>& first_steps() const {
    return steps1_;
  }

  int second_size() const {
    return n2_;
  }

  const std::vector<int>& second_steps() const {
    return steps2_;
  }

 private:
  int n_;
  FFTType type_;
  std::vector<int> steps_;
  int n1_;
  std::vector<int> steps1_;
  int n2_;
  std::vector<int> steps2_;
  int bluestein_n_;
};

std::vector<int> plan_stockham_fft(int n) {
  auto radices = supported_radices();
  std::vector<int> plan(radices.size(), 0);
  int orig_n = n;
  if (n == 1) {
    return plan;
  }
  for (int i = 0; i < radices.size(); i++) {
    int radix = radices[i];
    // Manually tuned radices for powers of 2
    if (is_power_of_2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return plan;
      }
    }
  }
  throw std::runtime_error("Unplannable");
}

OldFFTPlan plan_fft(int n) {
  auto radices = supported_radices();

  OldFFTPlan plan;
  plan.n = n;
  plan.rader = std::vector<int>(radices.size(), 0);

  // Four Step FFT when N is too large for shared mem.
  if (n > MAX_STOCKHAM_FFT_SIZE && is_power_of_2(n)) {
    // For power's of two we have a fast, no transpose four step implementation.
    plan.four_step = true;
    // Rough heuristic for choosing faster powers of two when we can
    plan.n2 = n > 65536 ? 1024 : 64;
    plan.n1 = n / plan.n2;
    return plan;
  }

  if (n > MAX_STOCKHAM_FFT_SIZE) {
    // Otherwise we use a multi-upload Bluestein's
    plan.four_step = true;
    plan.bluestein_n = next_fast_n(2 * n - 1);
    return plan;
  }

  int remaining_n = n;
  auto factors = prime_factors(n);
  for (int factor : factors) {
    // Make sure the factor is a supported radix
    if (std::find(radices.begin(), radices.end(), factor) == radices.end()) {
      // We only support a single Rader factor currently
      // TODO(alexbarron) investigate weirdness with large
      // Rader sizes -- possibly a compiler issue?
      if (plan.rader_n > 1 || n > MAX_RADER_FFT_SIZE) {
        plan.four_step = n > MAX_BLUESTEIN_FFT_SIZE;
        plan.bluestein_n = next_fast_n(2 * n - 1);
        plan.stockham = plan_stockham_fft(plan.bluestein_n);
        plan.rader = std::vector<int>(radices.size(), 0);
        return plan;
      }
      // See if we can use Rader's algorithm to Stockham decompose n - 1
      auto rader_factors = prime_factors(factor - 1);
      int last_factor = -1;
      for (int rf : rader_factors) {
        // We don't nest Rader's algorithm so if `factor - 1`
        // isn't Stockham decomposable we give up and do Bluestein's.
        if (std::find(radices.begin(), radices.end(), rf) == radices.end()) {
          plan.four_step = n > MAX_BLUESTEIN_FFT_SIZE;
          plan.bluestein_n = next_fast_n(2 * n - 1);
          plan.stockham = plan_stockham_fft(plan.bluestein_n);
          plan.rader = std::vector<int>(radices.size(), 0);
          return plan;
        }
      }
      plan.rader = plan_stockham_fft(factor - 1);
      plan.rader_n = factor;
      remaining_n /= factor;
    }
  }

  plan.stockham = plan_stockham_fft(remaining_n);
  return plan;
}

int compute_elems_per_thread(OldFFTPlan plan) {
  // Heuristics for selecting an efficient number
  // of threads to use for a particular mixed-radix FFT.
  auto n = plan.n;

  std::vector<int> steps;
  auto radices = supported_radices();
  steps.insert(steps.end(), plan.stockham.begin(), plan.stockham.end());
  steps.insert(steps.end(), plan.rader.begin(), plan.rader.end());
  std::set<int> used_radices;
  for (int i = 0; i < steps.size(); i++) {
    int radix = radices[i % radices.size()];
    if (steps[i] > 0) {
      used_radices.insert(radix);
    }
  }

  // Manual tuning for 7/11/13
  if (used_radices.find(7) != used_radices.end() &&
      (used_radices.find(11) != used_radices.end() ||
       used_radices.find(13) != used_radices.end())) {
    return 7;
  } else if (
      used_radices.find(11) != used_radices.end() &&
      used_radices.find(13) != used_radices.end()) {
    return 11;
  }

  // TODO(alexbarron) Some really weird stuff is going on
  // for certain `elems_per_thread` on large composite n.
  // Possibly a compiler issue?
  if (n == 3159)
    return 13;
  if (n == 3645)
    return 5;
  if (n == 3969)
    return 7;
  if (n == 1982)
    return 5;

  if (used_radices.size() == 1) {
    return *(used_radices.begin());
  }
  if (used_radices.size() == 2) {
    if (used_radices.find(11) != used_radices.end() ||
        used_radices.find(13) != used_radices.end()) {
      return std::accumulate(used_radices.begin(), used_radices.end(), 0) / 2;
    }
    std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
    return radix_vec[1];
  }
  // In all other cases use the second smallest radix.
  std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
  return radix_vec[1];
}

// Rader
int mod_exp(int x, int y, int n) {
  int out = 1;
  while (y) {
    if (y & 1) {
      out = out * x % n;
    }
    y >>= 1;
    x = x * x % n;
  }
  return out;
}

int primitive_root(int n) {
  auto factors = prime_factors(n - 1);

  for (int r = 2; r < n - 1; r++) {
    bool found = true;
    for (int factor : factors) {
      if (mod_exp(r, (n - 1) / factor, n) == 1) {
        found = false;
        break;
      }
    }
    if (found) {
      return r;
    }
  }
  return -1;
}

std::tuple<array, array, array> compute_raders_constants(
    int rader_n,
    const Stream& s) {
  int proot = primitive_root(rader_n);
  // Fermat's little theorem
  int inv = mod_exp(proot, rader_n - 2, rader_n);
  std::vector<short> g_q(rader_n - 1);
  std::vector<short> g_minus_q(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    g_q[i] = mod_exp(proot, i, rader_n);
    g_minus_q[i] = mod_exp(inv, i, rader_n);
  }
  array g_q_arr(g_q.begin(), {rader_n - 1});
  array g_minus_q_arr(g_minus_q.begin(), {rader_n - 1});

  std::vector<std::complex<float>> b_q(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    float pi_i = (float)g_minus_q[i] * -2.0 * M_PI / rader_n;
    b_q[i] = std::exp(std::complex<float>(0, pi_i));
  }

  array b_q_fft({rader_n - 1}, complex64, nullptr, {});
  b_q_fft.set_data(allocator::malloc(b_q_fft.nbytes()));
  auto b_q_fft_ptr =
      reinterpret_cast<std::complex<float>*>(b_q_fft.data<complex64_t>());
  std::ptrdiff_t item_size = b_q_fft.itemsize();
  size_t fft_size = rader_n - 1;
  // This FFT is always small (<4096, batch 1) so save some overhead
  // and do it on the CPU
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ b_q.data(),
      /* data_out= */ b_q_fft_ptr,
      /* scale= */ 1.0f);
  return std::make_tuple(b_q_fft, g_q_arr, g_minus_q_arr);
}

// Bluestein
std::pair<array, array> compute_bluestein_constants(int n, int bluestein_n) {
  // We need to calculate the Bluestein twiddle factors
  // in double precision for the overall numerical stability
  // of Bluestein's FFT algorithm to be acceptable.
  //
  // Metal doesn't support float64, so instead we
  // manually implement the required operations on cpu.
  //
  // In numpy:
  // w_k = np.exp(-1j * np.pi / N * (np.arange(-N + 1, N) ** 2))
  // w_q = np.fft.fft(1/w_k)
  // return w_k, w_q
  int length = 2 * n - 1;

  std::vector<std::complex<float>> w_k_vec(n);
  std::vector<std::complex<float>> w_q_vec(bluestein_n, 0);

  for (int i = -n + 1; i < n; i++) {
    double theta = pow(i, 2) * M_PI / (double)n;
    w_q_vec[i + n - 1] = std::exp(std::complex<double>(0, theta));
    if (i >= 0) {
      w_k_vec[i] = std::exp(std::complex<double>(0, -theta));
    }
  }

  array w_k({n}, complex64, nullptr, {});
  w_k.set_data(allocator::malloc(w_k.nbytes()));
  std::copy(w_k_vec.begin(), w_k_vec.end(), w_k.data<complex64_t>());

  array w_q({bluestein_n}, complex64, nullptr, {});
  w_q.set_data(allocator::malloc(w_q.nbytes()));
  auto w_q_ptr =
      reinterpret_cast<std::complex<float>*>(w_q.data<complex64_t>());

  std::ptrdiff_t item_size = w_q.itemsize();
  size_t fft_size = bluestein_n;
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ w_q_vec.data(),
      /* data_out= */ w_q_ptr,
      /* scale= */ 1.0f);
  return std::make_tuple(w_k, w_q);
}

void multi_upload_bluestein_fft(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    OldFFTPlan& plan,
    std::vector<array>& copies,
    const Stream& s) {
  auto& d = metal::device(s.device);

  // TODO(alexbarron) Implement fused kernels for mutli upload bluestein's
  // algorithm
  int n = inverse ? out.shape(axis) : in.shape(axis);
  auto [w_k, w_q] = compute_bluestein_constants(n, plan.bluestein_n);
  copies.push_back(w_k);
  copies.push_back(w_q);

  auto temp_shape = inverse ? out.shape() : in.shape();
  array temp(temp_shape, complex64, nullptr, {});
  array temp1(temp_shape, complex64, nullptr, {});

  if (real && !inverse) {
    // Convert float32->complex64
    copy_gpu(in, temp, CopyType::General, s);
    copies.push_back(temp);
  } else if (real && inverse) {
    int back_offset = n % 2 == 0 ? 2 : 1;
    auto slice_shape = in.shape();
    slice_shape[axis] -= back_offset;
    array slice_temp(slice_shape, complex64, nullptr, {});
    array conj_temp(in.shape(), complex64, nullptr, {});
    copies.push_back(conj_temp);

    Shape rstarts(in.ndim(), 0);
    Shape rstrides(in.ndim(), 1);
    rstarts[axis] = in.shape(axis) - back_offset;
    rstrides[axis] = -1;
    unary_op_gpu({in}, conj_temp, "Conjugate", s);
    slice_gpu(in, slice_temp, rstarts, rstrides, s);
    concatenate_gpu({conj_temp, slice_temp}, temp, (int)axis, s);
    copies.push_back(temp);
  } else if (inverse) {
    unary_op_gpu({in}, temp, "Conjugate", s);
    copies.push_back(temp);
  } else {
    temp.copy_shared_buffer(in);
  }

  Strides b_strides(in.ndim(), 0);
  b_strides[axis] = 1;
  array w_k_broadcast(temp.shape(), complex64, nullptr, {});
  w_k_broadcast.copy_shared_buffer(w_k, b_strides, {}, w_k.data_size());
  binary_op_gpu({temp, w_k_broadcast}, temp1, "Multiply", s);

  std::vector<std::pair<int, int>> pads;
  auto padded_shape = out.shape();
  padded_shape[axis] = plan.bluestein_n;
  array pad_temp(padded_shape, complex64, nullptr, {});
  auto zero = array(complex64_t{0.0f, 0.0f});
  copies.push_back(zero);
  pad_gpu(temp1, zero, pad_temp, {(int)axis}, {0}, s);
  copies.push_back(pad_temp);

  array pad_temp1(padded_shape, complex64, nullptr, {});
  fft_op(
      pad_temp,
      pad_temp1,
      axis,
      /*inverse=*/false,
      /*real=*/false,
      FourStepParams(),
      /*inplace=*/false,
      d,
      s);
  copies.push_back(pad_temp1);

  array w_q_broadcast(pad_temp1.shape(), complex64, nullptr, {});
  w_q_broadcast.copy_shared_buffer(w_q, b_strides, {}, w_q.data_size());
  binary_op_gpu_inplace({pad_temp1, w_q_broadcast}, pad_temp, "Multiply", s);

  fft_op(
      pad_temp,
      pad_temp1,
      axis,
      /* inverse= */ true,
      /* real= */ false,
      FourStepParams(),
      /*inplace=*/true,
      d,
      s);

  int offset = plan.bluestein_n - (2 * n - 1);
  Shape starts(in.ndim(), 0);
  Shape strides(in.ndim(), 1);
  starts[axis] = plan.bluestein_n - offset - n;

  array temp2(temp_shape, complex64, nullptr, {});
  slice_gpu(pad_temp1, temp2, starts, strides, s);

  binary_op_gpu_inplace({temp2, w_k_broadcast}, temp1, "Multiply", s);

  if (real && !inverse) {
    Shape rstarts(in.ndim(), 0);
    Shape rstrides(in.ndim(), 1);
    slice_gpu(temp1, out, rstarts, strides, s);
  } else if (real && inverse) {
    Strides b_strides(in.ndim(), 0);
    auto inv_n = array({1.0f / n}, {1}, float32);
    array temp_float(out.shape(), out.dtype(), nullptr, {});
    copies.push_back(temp_float);
    copies.push_back(inv_n);
    copies.push_back(temp1);

    copy_gpu(temp1, temp_float, CopyType::General, s);
    binary_op_gpu({temp_float, inv_n}, out, "Multiply", s);
  } else if (inverse) {
    auto inv_n = array({1.0f / n}, {1}, complex64);
    array temp3(temp_shape, complex64, nullptr, {});
    unary_op_gpu({temp1}, temp3, "Conjugate", s);
    binary_op_gpu({temp3, inv_n}, out, "Multiply", s);
    copies.push_back(inv_n);
    copies.push_back(temp1);
    copies.push_back(temp3);
  } else {
    out.copy_shared_buffer(temp1);
  }
}

void four_step_fft(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    OldFFTPlan& plan,
    std::vector<array>& copies,
    const Stream& s,
    bool in_place) {
  auto& d = metal::device(s.device);

  if (plan.bluestein_n == -1) {
    // Fast no transpose implementation for powers of 2.
    FourStepParams four_step_params = {
        /* required= */ true, /* first_step= */ true, plan.n1, plan.n2};
    auto temp_shape = (real && inverse) ? out.shape() : in.shape();
    array temp(temp_shape, complex64, nullptr, {});
    fft_op(
        in,
        temp,
        axis,
        inverse,
        real,
        four_step_params,
        /*inplace=*/false,
        d,
        s);
    four_step_params.first_step = false;
    fft_op(
        temp,
        out,
        axis,
        inverse,
        real,
        four_step_params,
        /*inplace=*/in_place,
        d,
        s);
    copies.push_back(temp);
  } else {
    multi_upload_bluestein_fft(in, out, axis, inverse, real, plan, copies, s);
  }
}

void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    const FourStepParams four_step_params,
    bool inplace,
    metal::Device& d,
    const Stream& s) {
  size_t n = out.dtype() == float32 ? out.shape(axis) : in.shape(axis);
  if (n == 1) {
    out.copy_shared_buffer(in);
    return;
  }

  if (four_step_params.required) {
    // Four Step FFT decomposes into two FFTs: n1 on columns, n2 on rows
    n = four_step_params.first_step ? four_step_params.n1 : four_step_params.n2;
  }

  // Make sure that the array is contiguous and has stride 1 in the FFT dim
  std::vector<array> copies;
  auto check_input = [&axis, &copies, &s](const array& x) {
    // TODO: Pass the strides to the kernel so
    // we can avoid the copy when x is not contiguous.
    bool no_copy = x.strides()[axis] == 1 &&
        (x.flags().row_contiguous || x.flags().col_contiguous);
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      Strides strides;
      int64_t cur_stride = x.shape(axis);
      for (int a = 0; a < x.ndim(); a++) {
        if (a == axis) {
          strides.push_back(1);
        } else {
          strides.push_back(cur_stride);
          cur_stride *= x.shape(a);
        }
      }

      auto flags = x.flags();
      auto [data_size, is_row_contiguous, is_col_contiguous] =
          check_contiguity(x.shape(), strides);

      flags.col_contiguous = is_col_contiguous;
      flags.row_contiguous = is_row_contiguous;
      flags.contiguous = data_size == x_copy.size();

      x_copy.set_data(allocator::malloc(x.nbytes()), data_size, strides, flags);
      copy_gpu_inplace(x, x_copy, CopyType::GeneralGeneral, s);
      copies.push_back(x_copy);
      return x_copy;
    }
  };
  const array& in_contiguous = check_input(in);

  // real to complex: n -> (n/2)+1
  // complex to real: (n/2)+1 -> n
  auto out_strides = in_contiguous.strides();
  size_t out_data_size = in_contiguous.data_size();
  if (in.shape(axis) != out.shape(axis)) {
    for (int i = 0; i < out_strides.size(); i++) {
      if (out_strides[i] != 1) {
        out_strides[i] = out_strides[i] / in.shape(axis) * out.shape(axis);
      }
    }
    out_data_size = out_data_size / in.shape(axis) * out.shape(axis);
  }

  auto plan = plan_fft(n);
  if (plan.four_step) {
    four_step_fft(in, out, axis, inverse, real, plan, copies, s, inplace);
    d.add_temporaries(std::move(copies), s.index);
    return;
  }

  // TODO: allow donation here
  if (!inplace) {
    out.set_data(
        allocator::malloc(out.nbytes()),
        out_data_size,
        out_strides,
        in_contiguous.flags());
  }

  auto radices = supported_radices();
  int fft_size = plan.bluestein_n > 0 ? plan.bluestein_n : n;

  // Setup function constants
  bool power_of_2 = is_power_of_2(fft_size);

  auto make_int = [](int* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeInt, i);
  };
  auto make_bool = [](bool* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeBool, i);
  };

  std::vector<MTLFC> func_consts = {
      make_bool(&inverse, 0), make_bool(&power_of_2, 1)};

  // Start of radix/rader step constants
  int index = 4;
  for (int i = 0; i < plan.stockham.size(); i++) {
    func_consts.push_back(make_int(&plan.stockham[i], index));
    index += 1;
  }
  for (int i = 0; i < plan.rader.size(); i++) {
    func_consts.push_back(make_int(&plan.rader[i], index));
    index += 1;
  }
  int elems_per_thread = compute_elems_per_thread(plan);
  func_consts.push_back(make_int(&elems_per_thread, 2));

  int rader_m = n / plan.rader_n;
  func_consts.push_back(make_int(&rader_m, 3));

  // The overall number of FFTs we're going to compute for this input
  int size = out.dtype() == float32 ? out.size() : in.size();
  if (real && inverse && four_step_params.required) {
    size = out.size();
  }
  int total_batch_size = size / n;
  int threads_per_fft = (fft_size + elems_per_thread - 1) / elems_per_thread;

  // We batch among threadgroups for improved efficiency when n is small
  int threadgroup_batch_size = std::max(MIN_THREADGROUP_MEM_SIZE / fft_size, 1);
  if (four_step_params.required) {
    // Require a threadgroup batch size of at least 4 for four step FFT
    // so we can coalesce the memory accesses.
    threadgroup_batch_size =
        std::max(threadgroup_batch_size, MIN_COALESCE_WIDTH);
  }
  int threadgroup_mem_size = next_power_of_2(threadgroup_batch_size * fft_size);
  // FFTs up to 2^20 are currently supported
  assert(threadgroup_mem_size <= MAX_STOCKHAM_FFT_SIZE);

  // ceil divide
  int batch_size =
      (total_batch_size + threadgroup_batch_size - 1) / threadgroup_batch_size;

  if (real && !four_step_params.required) {
    // We can perform 2 RFFTs at once so the batch size is halved.
    batch_size = (batch_size + 2 - 1) / 2;
  }
  int out_buffer_size = out.size();

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto in_type_str = in.dtype() == float32 ? "float" : "float2";
  auto out_type_str = out.dtype() == float32 ? "float" : "float2";
  // Only required by four step
  int step = -1;
  {
    std::ostringstream kname;
    std::string inv_string = inverse ? "true" : "false";
    std::string real_string = real ? "true" : "false";
    std::string func_name;
    if (plan.bluestein_n > 0) {
      kname << "bluestein_fft_mem_" << threadgroup_mem_size << "_"
            << in_type_str << "_" << out_type_str;
      func_name = "bluestein_fft";
    } else if (plan.rader_n > 1) {
      kname << "rader_fft_mem_" << threadgroup_mem_size << "_" << in_type_str
            << "_" << out_type_str;
      func_name = "rader_fft";
    } else if (four_step_params.required) {
      step = four_step_params.first_step ? 0 : 1;
      kname << "four_step_mem_" << threadgroup_mem_size << "_" << in_type_str
            << "_" << out_type_str << "_" << step << "_" << real_string;
      func_name = "four_step_fft";
    } else {
      kname << "fft_mem_" << threadgroup_mem_size << "_" << in_type_str << "_"
            << out_type_str;
      func_name = "fft";
    }
    std::string base_name = kname.str();
    // We use a specialized kernel for each FFT size
    kname << "_n" << fft_size << "_inv_" << inverse;
    std::string hash_name = kname.str();
    auto template_def = func_name == "four_step_fft" ? get_template_definition(
                                                           base_name,
                                                           func_name,
                                                           threadgroup_mem_size,
                                                           in_type_str,
                                                           out_type_str,
                                                           step,
                                                           real)
                                                     : get_template_definition(
                                                           base_name,
                                                           func_name,
                                                           threadgroup_mem_size,
                                                           in_type_str,
                                                           out_type_str);
    auto kernel =
        get_fft_kernel(d, base_name, hash_name, func_consts, template_def);

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);

    if (plan.bluestein_n > 0) {
      // Precomputed twiddle factors for Bluestein's
      auto [w_k, w_q] = compute_bluestein_constants(n, plan.bluestein_n);
      copies.push_back(w_q);
      copies.push_back(w_k);

      compute_encoder.set_input_array(w_q, 2); // w_q
      compute_encoder.set_input_array(w_k, 3); // w_k
      compute_encoder.set_bytes(n, 4);
      compute_encoder.set_bytes(plan.bluestein_n, 5);
      compute_encoder.set_bytes(total_batch_size, 6);
    } else if (plan.rader_n > 1) {
      auto [b_q, g_q, g_minus_q] = compute_raders_constants(plan.rader_n, s);
      copies.push_back(b_q);
      copies.push_back(g_q);
      copies.push_back(g_minus_q);

      compute_encoder.set_input_array(b_q, 2);
      compute_encoder.set_input_array(g_q, 3);
      compute_encoder.set_input_array(g_minus_q, 4);
      compute_encoder.set_bytes(n, 5);
      compute_encoder.set_bytes(total_batch_size, 6);
      compute_encoder.set_bytes(plan.rader_n, 7);
    } else if (four_step_params.required) {
      compute_encoder.set_bytes(four_step_params.n1, 2);
      compute_encoder.set_bytes(four_step_params.n2, 3);
      compute_encoder.set_bytes(total_batch_size, 4);
    } else {
      compute_encoder.set_bytes(n, 2);
      compute_encoder.set_bytes(total_batch_size, 3);
    }

    auto group_dims = MTL::Size(1, threadgroup_batch_size, threads_per_fft);
    auto grid_dims =
        MTL::Size(batch_size, threadgroup_batch_size, threads_per_fft);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

inline int compute_elems_per_thread(int n, const std::vector<int>& steps) {
  auto radices = supported_radices();
  std::set<int> used_radices;
  for (int i = 0; i < steps.size(); i++) {
    if (steps[i] > 0) {
      used_radices.insert(radices[i % radices.size()]);
    }
  }

  // Manual tuning for 7/11/13
  if (used_radices.find(7) != used_radices.end() &&
      (used_radices.find(11) != used_radices.end() ||
       used_radices.find(13) != used_radices.end())) {
    return 7;
  } else if (
      used_radices.find(11) != used_radices.end() &&
      used_radices.find(13) != used_radices.end()) {
    return 11;
  }

  // TODO(alexbarron) Some really weird stuff is going on
  // for certain `elems_per_thread` on large composite n.
  // Possibly a compiler issue?
  if (n == 3159)
    return 13;
  if (n == 3645)
    return 5;
  if (n == 3969)
    return 7;
  if (n == 1982)
    return 5;

  if (used_radices.size() == 1) {
    return *(used_radices.begin());
  }
  if (used_radices.size() == 2 &&
      (used_radices.find(11) != used_radices.end() ||
       used_radices.find(13) != used_radices.end())) {
    return std::accumulate(used_radices.begin(), used_radices.end(), 0) / 2;
  }

  // In all other cases use the second smallest radix.
  return *(++used_radices.begin());
}

inline array ensure_fastest_moving_axis(
    const array& x,
    int axis,
    metal::Device& d,
    const Stream& s) {
  // The axis is already with a stride of 1 so check that we have no overlaps
  // and broadcasting and avoid the copy.
  if (x.strides(axis) == 1) {
    // This is a fairly strict test perhaps consider relaxing it in the future.
    if (x.flags().row_contiguous || x.flags().col_contiguous) {
      return x;
    }
  }

  // To make it the fastest moving axis simply transpose it, then copy it and
  // then transpose it back.

  // Transpose it
  std::vector<int> axes(x.ndim(), 0);
  for (int ax = 0; ax < axes.size(); ax++) {
    axes[ax] = (ax < axis) ? ax : ax + 1;
  }
  axes.back() = axis;
  Shape xtshape;
  xtshape.reserve(axes.size());
  for (auto ax : axes) {
    xtshape.push_back(x.shape(ax));
  }
  array xt(xtshape, x.dtype(), nullptr, {});
  transpose(x, xt, axes);

  // Copy it
  array xtc(xt.shape(), x.dtype(), nullptr, {});
  copy_gpu(
      xt,
      xtc,
      xt.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      s);
  d.add_temporary(xtc, s.index);

  // Transpose it
  for (int ax = 0; ax < axes.size(); ax++) {
    axes[ax] = (ax < axis) ? ax : ((ax == axis) ? axes.size() - 1 : ax - 1);
  }
  array y(x.shape(), x.dtype(), nullptr, {});
  transpose(xtc, y, axes);

  return y;
}

inline void prepare_output_array(const array& in, array& out, int axis) {
  // Prepare the output array such that it matches the input in terms of
  // stride ordering. Namely we might have moved `axis` around in the `in`
  // array. We must do the same in `out`. The difference is that we don't have
  // to copy anything because `out` contains garbage at the moment.

  if (in.flags().row_contiguous && out.flags().row_contiguous) {
    return;
  }

  std::vector<int> axes(out.ndim(), 0);
  for (int ax = 0; ax < axes.size(); ax++) {
    axes[ax] = (ax < axis) ? ax : ax + 1;
  }
  axes.back() = axis;
  as_transposed(out, axes);
}

void fft_stockham_inplace(
    const FFTPlan& plan,
    const array& in_,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    metal::Device& d,
    const Stream& s) {
  // Prepare the input and output arrays such that `axis` has stride 1.
  // Possibly copy the input but never the output as it doesn't have anything
  // useful in it yet.
  array in = ensure_fastest_moving_axis(in_, axis, d, s);
  prepare_output_array(in, out, axis);

  // Prepare the arguments for stockham fft
  int n = plan.size();
  bool power_of_2 = is_power_of_2(n);
  int total_batch_size =
      out.dtype() == float32 ? out.size() / n : in.size() / n;
  auto& steps = plan.steps();
  int elems_per_thread = compute_elems_per_thread(n, steps);
  int threads_per_fft = ceildiv(n, elems_per_thread);
  int tg_batch_size = std::max(MIN_THREADGROUP_MEM_SIZE / n, 1);
  int tg_mem_size = next_power_of_2(tg_batch_size * n);
  int batch_size = ceildiv(total_batch_size, tg_batch_size);
  batch_size = real ? ceildiv(batch_size, 2) : batch_size; // 2 RFFTs at once
  std::vector<MTLFC> func_consts = {
      {&inverse, MTL::DataType::DataTypeBool, 0},
      {&power_of_2, MTL::DataType::DataTypeBool, 1},
      {&elems_per_thread, MTL::DataType::DataTypeInt, 2}};
  for (int i = 0; i < steps.size(); i++) {
    func_consts.emplace_back(&steps[i], MTL::DataType::DataTypeInt, 4 + i);
  }

  // Get the kernel
  auto in_type = in.dtype() == float32 ? "float" : "float2";
  auto out_type = out.dtype() == float32 ? "float" : "float2";
  std::string hash_name;
  std::string kname;
  kname.reserve(64);
  hash_name.reserve(64);
  concatenate(kname, "fft_mem_", tg_mem_size, "_", in_type, "_", out_type);
  concatenate(hash_name, kname, "_n", n, "_inv_", inverse);
  auto template_def =
      get_template_definition(kname, "fft", tg_mem_size, in_type, out_type);
  auto kernel = get_fft_kernel(d, kname, hash_name, func_consts, template_def);

  // Launch it
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(n, 2);
  compute_encoder.set_bytes(total_batch_size, 3);

  MTL::Size group_dims(1, tg_batch_size, threads_per_fft);
  MTL::Size grid_dims(batch_size, tg_batch_size, threads_per_fft);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void fft_four_step_inplace(
    const FFTPlan& plan,
    const array& in_,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    metal::Device& d,
    const Stream& s) {
  // Prepare the input and output arrays such that `axis` has stride 1.
  // Possibly copy the input but never the output as it doesn't have anything
  // useful in it yet.
  array in = ensure_fastest_moving_axis(in_, axis, d, s);
  prepare_output_array(in, out, axis);

  // Also prepare the intermediate array for the four-step fft which is
  // implemented with 2 kernel calls.
  array intermediate(
      (real && inverse) ? out.shape() : in.shape(), complex64, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  prepare_output_array(in, intermediate, axis);
  d.add_temporary(intermediate, s.index);

  // Make the two calls
  for (int step = 0; step < 2; step++) {
    // Create the parameters
    int n1 = plan.first_size();
    int n2 = plan.second_size();
    int n = (step == 0) ? n1 : n2;
    bool power_of_2 = true;
    int total_batch_size =
        out.dtype() == float32 ? out.size() / n : in.size() / n;
    auto& steps = (step == 0) ? plan.first_steps() : plan.second_steps();
    int elems_per_thread = compute_elems_per_thread(n, steps);
    int threads_per_fft = ceildiv(n, elems_per_thread);
    int tg_batch_size =
        std::max(MIN_THREADGROUP_MEM_SIZE / n, MIN_COALESCE_WIDTH);
    int tg_mem_size = next_power_of_2(tg_batch_size * n);
    int batch_size = ceildiv(total_batch_size, tg_batch_size);
    std::vector<MTLFC> func_consts = {
        {&inverse, MTL::DataType::DataTypeBool, 0},
        {&power_of_2, MTL::DataType::DataTypeBool, 1},
        {&elems_per_thread, MTL::DataType::DataTypeInt, 2}};
    for (int i = 0; i < steps.size(); i++) {
      func_consts.emplace_back(&steps[i], MTL::DataType::DataTypeInt, 4 + i);
    }

    // Get the kernel
    auto in_type = in.dtype() == float32 ? "float" : "float2";
    auto out_type = out.dtype() == float32 ? "float" : "float2";
    std::string hash_name;
    std::string kname;
    kname.reserve(64);
    hash_name.reserve(64);
    concatenate(
        kname,
        "four_step_mem_",
        tg_mem_size,
        "_",
        in_type,
        "_",
        out_type,
        "_",
        step,
        (real ? "_true" : "_false"));
    concatenate(hash_name, kname, "_n", n, "_inv_", inverse);
    auto template_def = get_template_definition(
        kname, "four_step_fft", tg_mem_size, in_type, out_type, step, real);
    auto kernel =
        get_fft_kernel(d, kname, hash_name, func_consts, template_def);

    // Launch it
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array((step == 0) ? in : intermediate, 0);
    compute_encoder.set_output_array((step == 0) ? intermediate : out, 1);
    compute_encoder.set_bytes(n1, 2);
    compute_encoder.set_bytes(n2, 3);
    compute_encoder.set_bytes(total_batch_size, 4);

    MTL::Size group_dims(1, tg_batch_size, threads_per_fft);
    MTL::Size grid_dims(batch_size, tg_batch_size, threads_per_fft);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void fft_op_inplace(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    metal::Device& d,
    const Stream& s) {
  // Get the FFT size and plan it
  auto plan =
      FFTPlan(out.dtype() == float32 ? out.shape(axis) : in.shape(axis));

  switch (plan.type()) {
    case FFTPlan::NOOP:
      std::cout << "--------------> 1-size FFT <-----------------" << std::endl;
      break;
    case FFTPlan::STOCKHAM:
      return fft_stockham_inplace(plan, in, out, axis, inverse, real, d, s);
    case FFTPlan::SMALL_FOUR_STEP:
      return fft_four_step_inplace(plan, in, out, axis, inverse, real, d, s);
    case FFTPlan::UNSUPPORTED: {
      std::string msg;
      concatenate(msg, "FFT of size ", plan.size(), " not supported");
      throw std::runtime_error(msg);
    }
    default:
      std::cout << "----- NYI ----" << std::endl;
      break;
  }
}

void nd_fft_op_inplace(
    const array& in,
    array& out,
    const std::vector<size_t>& axes,
    bool inverse,
    bool real,
    metal::Device& d,
    const Stream& s) {
  // We are going to make and possibly reuse some intermediate arrays that will
  // hold the intermediate fft results.
  auto shape = inverse ? in.shape() : out.shape();
  std::vector<array> intermediates;
  intermediates.reserve(2);

  // Utility to return either in or one of the intermediates.
  auto get_input_array = [&](int step) -> const array& {
    // The first step so use the input array
    if (step == 0) {
      return in;
    }

    return intermediates[(step - 1) % 2];
  };

  // Utility to return either out or one of the intermediates. It also informs
  // us if we should allocate memory for that output or there is already some
  // allocated.
  auto get_output_array = [&](int step) -> array& {
    // It is the final step so return the output array
    if (step == axes.size() - 1) {
      return out;
    }

    // We already have made an array that we can use so go ahead and use it and
    // don't reallocate the memory.
    if (step % 2 < intermediates.size()) {
      return intermediates[step % 2];
    }

    array x(shape, complex64, nullptr, {});
    x.set_data(allocator::malloc(x.nbytes()));
    intermediates.emplace_back(std::move(x));
    d.add_temporary(intermediates.back(), s.index);

    return intermediates.back();
  };

  // Perform ND FFT on GPU as a series of 1D FFTs
  for (int step = 0; step < axes.size(); step++) {
    auto x = get_input_array(step);
    auto y = get_output_array(step);
    auto step_axis = axes[inverse ? step : axes.size() - step - 1];
    auto step_real = real && (inverse ? step == axes.size() - 1 : step == 0);
    fft_op_inplace(x, y, step_axis, inverse, step_real, d, s);
  }
}

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  // The FFT ops above have the *_inplace suffix. This means that the memory
  // needs to be already allocated in the output array. Similar to
  // copy_gpu_inplace and so on.
  //
  // Even though we allocate the memory, we do not necessarily want the
  // contiguous strides so the *_inplace ops may change the strides and flags
  // of the array but will not reallocate the memory.

  out.set_data(allocator::malloc(out.nbytes()));

  if (axes_.size() > 1) {
    nd_fft_op_inplace(in, out, axes_, inverse_, real_, d, s);
  } else {
    fft_op_inplace(in, out, axes_[0], inverse_, real_, d, s);
  }
}

} // namespace mlx::core
