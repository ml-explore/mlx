// Copyright © 2023 Apple Inc.
#include <cassert>
#include <map>
#include <numeric>
#include <set>

#include <vecLib/vDSP.h>
#include <vecLib/vForce.h>

#include "mlx/3rdparty/pocketfft.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/ops.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/metal/binary.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/slicing.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mlx::core {

using MTLFC = std::tuple<const void*, MTL::DataType, NS::UInteger>;

#define MAX_STOCKHAM_FFT_SIZE 4096
#define MAX_RADER_FFT_SIZE 2048
#define MAX_BLUESTEIN_FFT_SIZE 2048
// Threadgroup memory batching improves throughput for small n
#define MIN_THREADGROUP_MEM_SIZE 256
// For strided reads/writes, coalesce at least this many complex64s
#define MIN_COALESCE_WIDTH 4

inline const std::vector<int> supported_radices() {
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
    const Stream& s);

struct FFTPlan {
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

int next_fast_n(int n) {
  return next_power_of_2(n);
}

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

FFTPlan plan_fft(int n) {
  auto radices = supported_radices();
  std::set<int> radices_set(radices.begin(), radices.end());

  FFTPlan plan;
  plan.n = n;
  plan.rader = std::vector<int>(radices.size(), 0);
  auto factors = prime_factors(n);
  int remaining_n = n;

  // Four Step FFT when N is too large for shared mem.
  if (n > MAX_STOCKHAM_FFT_SIZE && is_power_of_2(n)) {
    // For power's of two we have a fast, no transpose four step implementation.
    plan.four_step = true;
    // Rough heuristic for choosing faster powers of two when we can
    plan.n2 = n > 65536 ? 1024 : 64;
    plan.n1 = n / plan.n2;
    return plan;
  } else if (n > MAX_STOCKHAM_FFT_SIZE) {
    // Otherwise we use a multi-upload Bluestein's
    plan.four_step = true;
    plan.bluestein_n = next_fast_n(2 * n - 1);
    return plan;
  }

  for (int factor : factors) {
    // Make sure the factor is a supported radix
    if (radices_set.find(factor) == radices_set.end()) {
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
        if (radices_set.find(rf) == radices_set.end()) {
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

int compute_elems_per_thread(FFTPlan plan) {
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

void compute_bluestein_constants(
    int n,
    int bluestein_n,
    array& w_q,
    array& w_k) {
  // We need to calculate the Bluestein twiddle factors
  // in double precision for the overall numerical stability
  // of Bluestein's FFT algorithm to be acceptable.
  //
  // Metal doesn't support float64, so instead we
  // manually implement the required operations using accelerate on cpu.
  //
  // In numpy:
  // w_k = np.exp(-1j * np.pi / N * (np.arange(-N + 1, N) ** 2))
  // w_q = np.fft.fft(1/w_k)
  // return w_k, w_q
  size_t fft_size = w_q.shape(0);

  int length = 2 * n - 1;

  std::vector<double> x(length);
  std::vector<double> y(length);

  std::iota(x.begin(), x.end(), -n + 1);
  vDSP_vsqD(x.data(), 1, y.data(), 1, x.size());
  double theta = (double)1.0 / (double)n;
  vDSP_vsmulD(y.data(), 1, &theta, x.data(), 1, x.size());

  std::vector<double> real_part(length);
  std::vector<double> imag_part(length);
  vvcospi(real_part.data(), x.data(), &length);
  vvsinpi(imag_part.data(), x.data(), &length);

  double minus_1 = -1.0;
  vDSP_vsmulD(x.data(), 1, &minus_1, y.data(), 1, x.size());

  // compute w_k
  std::vector<double> real_part_w_k(n);
  std::vector<double> imag_part_w_k(n);
  vvcospi(real_part_w_k.data(), y.data() + length - n, &n);
  vvsinpi(imag_part_w_k.data(), y.data() + length - n, &n);

  auto convert_float = [](double real, double imag) {
    return std::complex<float>(real, imag);
  };

  std::vector<std::complex<float>> w_k_input(n, 0.0);
  std::transform(
      real_part_w_k.begin(),
      real_part_w_k.end(),
      imag_part_w_k.begin(),
      w_k_input.begin(),
      convert_float);

  w_k.set_data(allocator::malloc_or_wait(w_k.nbytes()));

  auto w_k_ptr =
      reinterpret_cast<std::complex<float>*>(w_k.data<complex64_t>());
  memcpy(w_k_ptr, w_k_input.data(), n * w_k.itemsize());

  // convert back to float now we've done the sincos
  std::vector<std::complex<float>> fft_input(fft_size, 0.0);
  std::transform(
      real_part.begin(),
      real_part.end(),
      imag_part.begin(),
      fft_input.begin(),
      convert_float);

  w_q.set_data(allocator::malloc_or_wait(w_q.nbytes()));
  auto w_q_ptr =
      reinterpret_cast<std::complex<float>*>(w_q.data<complex64_t>());

  std::ptrdiff_t item_size = w_q.itemsize();

  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ fft_input.data(),
      /* data_out= */ w_q_ptr,
      /* scale= */ 1.0f);
}

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

  CopyType ctype =
      g_minus_q_arr.flags().contiguous ? CopyType::Vector : CopyType::General;
  array g_minus_q_float({rader_n - 1}, complex64, nullptr, {});
  copy(g_minus_q_arr, g_minus_q_float, ctype);

  array pi_i =
      array({complex64_t{0.0f, (float)(-2.0 * M_PI / rader_n)}}, complex64);
  array temp_mul({rader_n - 1}, complex64, nullptr, {});
  binary(g_minus_q_float, pi_i, temp_mul, detail::Multiply());

  array temp_exp({rader_n - 1}, complex64, nullptr, {});
  unary_fp(temp_mul, temp_exp, detail::Exp());

  array b_q_fft({rader_n - 1}, complex64, nullptr, {});
  b_q_fft.set_data(allocator::malloc_or_wait(b_q_fft.nbytes()));
  auto b_q_fft_ptr =
      reinterpret_cast<std::complex<float>*>(b_q_fft.data<complex64_t>());
  auto temp_exp_ptr =
      reinterpret_cast<std::complex<float>*>(temp_exp.data<complex64_t>());
  std::ptrdiff_t item_size = b_q_fft.itemsize();
  size_t fft_size = rader_n - 1;
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ temp_exp_ptr,
      /* data_out= */ b_q_fft_ptr,
      /* scale= */ 1.0f);
  return std::make_tuple(b_q_fft, g_q_arr, g_minus_q_arr);
}

void multi_upload_bluestein_fft(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    FFTPlan& plan,
    std::vector<array> copies,
    const Stream& s) {
  // TODO(alexbarron) Implement fused kernels for mutli upload bluestein's
  // algorithm
  int n = inverse ? out.shape(axis) : in.shape(axis);
  array w_k({n}, complex64, nullptr, {});
  array w_q({plan.bluestein_n}, complex64, nullptr, {});
  compute_bluestein_constants(n, plan.bluestein_n, w_q, w_k);

  // Broadcast w_q and w_k to the batch size
  std::vector<size_t> b_strides(in.ndim(), 0);
  b_strides[axis] = 1;
  array w_k_broadcast({}, complex64, nullptr, {});
  array w_q_broadcast({}, complex64, nullptr, {});
  w_k_broadcast.copy_shared_buffer(w_k, b_strides, {}, w_k.data_size());
  w_q_broadcast.copy_shared_buffer(w_q, b_strides, {}, w_q.data_size());

  auto temp_shape = inverse ? out.shape() : in.shape();
  array temp(temp_shape, complex64, nullptr, {});
  array temp1(temp_shape, complex64, nullptr, {});

  if (real && !inverse) {
    // Convert float32->complex64
    copy_gpu(in, temp, CopyType::General, s);
  } else if (real && inverse) {
    int back_offset = n % 2 == 0 ? 2 : 1;
    auto slice_shape = in.shape();
    slice_shape[axis] -= back_offset;
    array slice_temp(slice_shape, complex64, nullptr, {});
    array conj_temp(in.shape(), complex64, nullptr, {});
    copies.push_back(slice_temp);
    copies.push_back(conj_temp);

    std::vector<int> rstarts(in.ndim(), 0);
    std::vector<int> rstrides(in.ndim(), 1);
    rstarts[axis] = in.shape(axis) - back_offset;
    rstrides[axis] = -1;
    unary_op_gpu({in}, conj_temp, "conj", s);
    slice_gpu(in, slice_temp, rstarts, rstrides, s);
    concatenate_gpu({conj_temp, slice_temp}, temp, (int)axis, s);
  } else if (inverse) {
    unary_op_gpu({in}, temp, "conj", s);
  } else {
    temp.copy_shared_buffer(in);
  }

  binary_op_gpu({temp, w_k_broadcast}, temp1, "mul", s);

  std::vector<std::pair<int, int>> pads;
  auto padded_shape = out.shape();
  padded_shape[axis] = plan.bluestein_n;
  array pad_temp(padded_shape, complex64, nullptr, {});
  pad_gpu(temp1, array(complex64_t{0.0f, 0.0f}), pad_temp, {(int)axis}, {0}, s);

  array pad_temp1(padded_shape, complex64, nullptr, {});
  fft_op(
      pad_temp,
      pad_temp1,
      axis,
      /*inverse=*/false,
      /*real=*/false,
      FourStepParams(),
      /*inplace=*/false,
      s);

  binary_op_gpu_inplace({pad_temp1, w_q_broadcast}, pad_temp, "mul", s);

  fft_op(
      pad_temp,
      pad_temp1,
      axis,
      /* inverse= */ true,
      /* real= */ false,
      FourStepParams(),
      /*inplace=*/true,
      s);

  int offset = plan.bluestein_n - (2 * n - 1);
  std::vector<int> starts(in.ndim(), 0);
  std::vector<int> strides(in.ndim(), 1);
  starts[axis] = plan.bluestein_n - offset - n;
  slice_gpu(pad_temp1, temp, starts, strides, s);

  binary_op_gpu_inplace({temp, w_k_broadcast}, temp1, "mul", s);

  if (real && !inverse) {
    std::vector<int> rstarts(in.ndim(), 0);
    std::vector<int> rstrides(in.ndim(), 1);
    slice_gpu(temp1, out, rstarts, strides, s);
  } else if (real && inverse) {
    std::vector<size_t> b_strides(in.ndim(), 0);
    auto inv_n = array({1.0f / n}, {1}, float32);
    array temp_float(out.shape(), out.dtype(), nullptr, {});
    copies.push_back(temp_float);
    copies.push_back(inv_n);

    copy_gpu(temp1, temp_float, CopyType::General, s);
    binary_op_gpu({temp_float, inv_n}, out, "mul", s);
  } else if (inverse) {
    auto inv_n = array({1.0f / n}, {1}, complex64);
    unary_op_gpu({temp1}, temp, "conj", s);
    binary_op_gpu({temp, inv_n}, out, "mul", s);
    copies.push_back(inv_n);
  } else {
    out.copy_shared_buffer(temp1);
  }

  copies.push_back(w_k);
  copies.push_back(w_q);
  copies.push_back(w_k_broadcast);
  copies.push_back(w_q_broadcast);
  copies.push_back(temp);
  copies.push_back(temp1);
  copies.push_back(pad_temp);
  copies.push_back(pad_temp1);
}

void four_step_fft(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    FFTPlan& plan,
    std::vector<array> copies,
    const Stream& s) {
  auto& d = metal::device(s.device);

  if (plan.bluestein_n == -1) {
    // Fast no transpose implementation for powers of 2.
    FourStepParams four_step_params = {
        /* required= */ true, /* first_step= */ true, plan.n1, plan.n2};
    auto temp_shape = (real && inverse) ? out.shape() : in.shape();
    array temp(temp_shape, complex64, nullptr, {});
    fft_op(
        in, temp, axis, inverse, real, four_step_params, /*inplace=*/false, s);
    four_step_params.first_step = false;
    fft_op(
        temp, out, axis, inverse, real, four_step_params, /*inplace=*/false, s);
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
    const Stream& s) {
  auto& d = metal::device(s.device);

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
      std::vector<size_t> strides;
      size_t cur_stride = x.shape(axis);
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

      flags.col_contiguous = is_row_contiguous;
      flags.row_contiguous = is_col_contiguous;
      flags.contiguous = data_size == x_copy.size();

      x_copy.set_data(
          allocator::malloc_or_wait(x.nbytes()), data_size, strides, flags);
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
    four_step_fft(in, out, axis, inverse, real, plan, copies, s);
    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }

  // TODO: allow donation here
  if (!inplace) {
    out.set_data(
        allocator::malloc_or_wait(out.nbytes()),
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
    if (plan.bluestein_n > 0) {
      kname << "bluestein_fft_mem_" << threadgroup_mem_size << "_"
            << in_type_str << "_" << out_type_str;
    } else if (plan.rader_n > 1) {
      kname << "rader_fft_mem_" << threadgroup_mem_size << "_" << in_type_str
            << "_" << out_type_str;
    } else if (four_step_params.required) {
      step = four_step_params.first_step ? 0 : 1;
      kname << "four_step_mem_" << threadgroup_mem_size << "_" << in_type_str
            << "_" << out_type_str << "_" << step << "_" << real_string;
    } else {
      kname << "fft_mem_" << threadgroup_mem_size << "_" << in_type_str << "_"
            << out_type_str;
    }
    std::string base_name = kname.str();
    // We use a specialized kernel for each FFT size
    kname << "_n" << fft_size << "_inv_" << inverse;
    std::string hash_name = kname.str();
    auto kernel = get_fft_kernel(
        d,
        base_name,
        hash_name,
        threadgroup_mem_size,
        in_type_str,
        out_type_str,
        step,
        real,
        func_consts);

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);

    if (plan.bluestein_n > 0) {
      // Precomputed twiddle factors for Bluestein's
      array w_q({plan.bluestein_n}, complex64, nullptr, {});
      array w_k({(int)n}, complex64, nullptr, {});
      compute_bluestein_constants(n, plan.bluestein_n, w_q, w_k);
      copies.push_back(w_q);
      copies.push_back(w_k);

      compute_encoder.set_input_array(w_q, 2); // w_q
      compute_encoder.set_input_array(w_k, 3); // w_k
      compute_encoder->setBytes(&n, sizeof(int), 4);
      compute_encoder->setBytes(&plan.bluestein_n, sizeof(int), 5);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 6);
    } else if (plan.rader_n > 1) {
      auto [b_q, g_q, g_minus_q] = compute_raders_constants(plan.rader_n, s);
      copies.push_back(b_q);
      copies.push_back(g_q);
      copies.push_back(g_minus_q);

      compute_encoder.set_input_array(b_q, 2);
      compute_encoder.set_input_array(g_q, 3);
      compute_encoder.set_input_array(g_minus_q, 4);
      compute_encoder->setBytes(&n, sizeof(int), 5);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 6);
      compute_encoder->setBytes(&plan.rader_n, sizeof(int), 7);
    } else if (four_step_params.required) {
      compute_encoder->setBytes(&four_step_params.n1, sizeof(int), 2);
      compute_encoder->setBytes(&four_step_params.n2, sizeof(int), 3);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 4);
    } else {
      compute_encoder->setBytes(&n, sizeof(int), 2);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 3);
    }

    auto group_dims = MTL::Size(1, threadgroup_batch_size, threads_per_fft);
    auto grid_dims =
        MTL::Size(batch_size, threadgroup_batch_size, threads_per_fft);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    bool inplace,
    const Stream& s) {
  fft_op(in, out, axis, inverse, real, FourStepParams(), inplace, s);
}

void nd_fft_op(
    const array& in,
    array& out,
    const std::vector<size_t>& axes,
    bool inverse,
    bool real,
    const Stream& s) {
  // Perform ND FFT on GPU as a series of 1D FFTs
  auto temp_shape = inverse ? in.shape() : out.shape();
  array temp1(temp_shape, complex64, nullptr, {});
  array temp2(temp_shape, complex64, nullptr, {});
  std::vector<array> temp_arrs = {temp1, temp2};
  for (int i = axes.size() - 1; i >= 0; i--) {
    int reverse_index = axes.size() - i - 1;
    // For 5D and above, we don't want to reallocate our two temporary arrays
    bool inplace = reverse_index >= 3 && i != 0;
    // Opposite order for fft vs ifft
    int index = inverse ? reverse_index : i;
    size_t axis = axes[index];
    // Mirror np.fft.(i)rfftn and perform a real transform
    // only on the final axis.
    bool step_real = (real && index == axes.size() - 1);
    int step_shape = inverse ? out.shape(axis) : in.shape(axis);
    const array& in_arr = i == axes.size() - 1 ? in : temp_arrs[1 - i % 2];
    array& out_arr = i == 0 ? out : temp_arrs[i % 2];
    fft_op(in_arr, out_arr, axis, inverse, step_real, inplace, s);
  }

  std::vector<array> copies = {temp1, temp2};
  auto& d = metal::device(s.device);
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& in = inputs[0];

  if (axes_.size() > 1) {
    nd_fft_op(in, out, axes_, inverse_, real_, s);
  } else {
    fft_op(in, out, axes_[0], inverse_, real_, /*inplace=*/false, s);
  }
}

} // namespace mlx::core
