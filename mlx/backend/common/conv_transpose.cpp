// Copyright © 2023 Apple Inc.

#include <cassert>

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif

#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

///////////////////////////////////////////////////////////////////////////////
// Naive reference transpose conv
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void slow_conv_1D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  const T* start_wt_ptr = wt.data<T>();

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();

  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = in.shape(1); // Input spatial dim
  const int oH = out.shape(1); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int C = wt.shape(2); // In channels
  const int wH = wt.shape(1); // Weight spatial dim

  const size_t in_stride_N = in.strides()[0];
  const size_t in_stride_H = in.strides()[1];
  const size_t in_stride_C = in.strides()[2];

  const size_t wt_stride_O = wt.strides()[0];
  const size_t wt_stride_H = wt.strides()[1];
  const size_t wt_stride_C = wt.strides()[2];

  const size_t out_stride_N = out.strides()[0];
  const size_t out_stride_H = out.strides()[1];
  const size_t out_stride_O = out.strides()[2];

  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < oH; ++oh) {
      for (int o = 0; o < O; ++o) {
        const T* filter_wt_ptr = start_wt_ptr + o * wt_stride_O;
        float r = 0.;

        for (int wh = 0; wh < wH; ++wh) {
          const T* wt_ptr = filter_wt_ptr + wh * wt_stride_H;

          int ih = oh * wt_strides[0] - padding[0] + wh * wt_dilation[0];

          if (ih >= 0 && ih < iH) {
            for (int c = 0; c < C; ++c) {
              r += static_cast<float>(
                       in_ptr[ih * in_stride_H + c * in_stride_C]) *
                  static_cast<float>(wt_ptr[c * wt_stride_C]);
            } // c

          } // ih check
        } // wh

        out_ptr[oh * out_stride_H + o * out_stride_O] = static_cast<T>(r);
      } // o
    } // oh

    in_ptr += in_stride_N;
    out_ptr += out_stride_N;

  } // n
}

template <typename T>
void slow_conv_transpose_2D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  const T* st_wt_ptr = wt.data<T>();
  const T* st_in_ptr = in.data<T>();
  T* st_out_ptr = out.data<T>();

  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = in.shape(1); // Input spatial dim
  const int iW = in.shape(2); // Input spatial dim
  const int oH = out.shape(1); // Output spatial dim
  const int oW = out.shape(2); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int C = wt.shape(3); // In channels
  const int wH = wt.shape(1); // Weight spatial dim
  const int wW = wt.shape(2); // Weight spatial dim

  const size_t in_stride_N = in.strides()[0];
  const size_t in_stride_H = in.strides()[1];
  const size_t in_stride_W = in.strides()[2];
  const size_t in_stride_C = in.strides()[3];

  const size_t wt_stride_O = wt.strides()[0];
  const size_t wt_stride_H = wt.strides()[1];
  const size_t wt_stride_W = wt.strides()[2];
  const size_t wt_stride_C = wt.strides()[3];

  const size_t out_stride_N = out.strides()[0];
  const size_t out_stride_H = out.strides()[1];
  const size_t out_stride_W = out.strides()[2];
  const size_t out_stride_O = out.strides()[3];

  auto pt_conv_transpose_no_checks =
      [&](const T* in_ptr, const T* wt_ptr, T* out_ptr, int oh, int ow) {
        out_ptr += oh * out_stride_H + ow * out_stride_W;
        int ih_base = oh * wt_strides[0] - padding[0];
        int iw_base = ow * wt_strides[1] - padding[1];

        for (int o = 0; o < O; ++o) {
          float r = 0.;

          for (int wh = 0; wh < wH; ++wh) {
            for (int ww = 0; ww < wW; ++ww) {
              int ih = ih_base + wh * wt_dilation[0];
              int iw = iw_base + ww * wt_dilation[1];

              const T* wt_ptr_pt = wt_ptr + wh * wt_stride_H + ww * wt_stride_W;
              const T* in_ptr_pt = in_ptr + ih * in_stride_H + iw * in_stride_W;

              for (int c = 0; c < C; ++c) {
                r += static_cast<float>(in_ptr_pt[0]) *
                    static_cast<float>(wt_ptr_pt[0]);

                out_ptr[0] = static_cast<T>(r);
                in_ptr_pt += in_stride_C;
                wt_ptr_pt += wt_stride_C;
              } // c

            } // ww
          } // wh

          out_ptr += out_stride_O;
          wt_ptr += wt_stride_O;
        } // o
      };

  auto pt_conv_transpose_all_checks =
      [&](const T* in_ptr, const T* wt_ptr, T* out_ptr, int oh, int ow) {
        out_ptr += oh * out_stride_H + ow * out_stride_W;
        int ih_base = oh * wt_strides[0] - padding[0];
        int iw_base = ow * wt_strides[1] - padding[1];

        for (int o = 0; o < O; ++o) {
          float r = 0.;

          for (int wh = 0; wh < wH; ++wh) {
            for (int ww = 0; ww < wW; ++ww) {
              int ih = ih_base + wh * wt_dilation[0];
              int iw = iw_base + ww * wt_dilation[1];

              if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                const T* wt_ptr_pt =
                    wt_ptr + wh * wt_stride_H + ww * wt_stride_W;
                const T* in_ptr_pt =
                    in_ptr + ih * in_stride_H + iw * in_stride_W;

                for (int c = 0; c < C; ++c) {
                  r += static_cast<float>(in_ptr_pt[0]) *
                      static_cast<float>(wt_ptr_pt[0]);
                  in_ptr_pt += in_stride_C;
                  wt_ptr_pt += wt_stride_C;
                } // c

              } // ih, iw check
            } // ww
          } // wh

          out_ptr[0] = static_cast<T>(r);
          out_ptr += out_stride_O;
          wt_ptr += wt_stride_O;
        } // o
      };

  int oH_border_0 = 0;
  int oH_border_1 = (padding[0] + wt_strides[0] + 1) / wt_strides[0];
  int oH_border_2 = (iH + padding[0] - wH * wt_dilation[0]) / wt_strides[0];
  int oH_border_3 = oH;

  int oW_border_0 = 0;
  int oW_border_1 = (padding[1] + wt_strides[0] + 1) / wt_strides[1];
  int oW_border_2 = (iW + padding[1] - wW * wt_dilation[1]) / wt_strides[1];
  int oW_border_3 = oW;

  for (int n = 0; n < N; ++n) {
    // Case 1: oh might put us out of bounds
    for (int oh = oH_border_0; oh < oH_border_1; ++oh) {
      for (int ow = 0; ow < oW; ++ow) {
        pt_conv_transpose_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow
    } // oh

    // Case 2: oh in bounds
    for (int oh = oH_border_1; oh < oH_border_2; ++oh) {
      // Case a: ow might put us out of bounds
      for (int ow = oW_border_0; ow < oW_border_1; ++ow) {
        pt_conv_transpose_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

      // Case b: ow in bounds
      for (int ow = oW_border_1; ow < oW_border_2; ++ow) {
        pt_conv_transpose_no_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

      // Case c: ow might put us out of bounds
      for (int ow = oW_border_2; ow < oW_border_3; ++ow) {
        pt_conv_transpose_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

    } // oh

    // Case 3: oh might put us out of bounds
    for (int oh = oH_border_2; oh < oH_border_3; ++oh) {
      for (int ow = 0; ow < oW; ++ow) {
        pt_conv_transpose_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow
    } // oh

    st_in_ptr += in_stride_N;
    st_out_ptr += out_stride_N;

  } // n
}

// void dispatch_slow_conv_1D(
//     const array& in,
//     const array& wt,
//     array out,
//     const std::vector<int>& padding,
//     const std::vector<int>& wt_strides,
//     const std::vector<int>& wt_dilation) {
//   if (in.dtype() == float32) {
//     return slow_conv_1D<float>(in, wt, out, padding, wt_strides,
//     wt_dilation);
//   } else if (in.dtype() == float16) {
//     return slow_conv_1D<float16_t>(
//         in, wt, out, padding, wt_strides, wt_dilation);
//   } else if (in.dtype() == bfloat16) {
//     return slow_conv_1D<bfloat16_t>(
//         in, wt, out, padding, wt_strides, wt_dilation);
//   } else {
//     throw std::invalid_argument(
//         "[Convolution::eval] got unsupported data type.");
//   }
// }

void dispatch_slow_conv_transpose_2D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  if (in.dtype() == float32) {
    return slow_conv_transpose_2D<float>(
        in, wt, out, padding, wt_strides, wt_dilation);
  } else if (in.dtype() == float16) {
    return slow_conv_transpose_2D<float16_t>(
        in, wt, out, padding, wt_strides, wt_dilation);
  } else if (in.dtype() == bfloat16) {
    return slow_conv_transpose_2D<bfloat16_t>(
        in, wt, out, padding, wt_strides, wt_dilation);
  } else {
    throw std::invalid_argument(
        "[TransposeConvolution::eval] got unsupported data type.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// ConvTranspose routing
///////////////////////////////////////////////////////////////////////////////

// void conv_1D_cpu(
//     const array& in,
//     const array& wt,
//     array out,
//     const std::vector<int>& padding,
//     const std::vector<int>& wt_strides,
//     const std::vector<int>& wt_dilation) {
//   if (wt_dilation[0] == 1) {
//     return explicit_gemm_conv_1D_cpu(
//         in, wt, out, padding, wt_strides, wt_dilation);
//   }

//   return dispatch_slow_conv_1D(in, wt, out, padding, wt_strides,
//   wt_dilation);
// }

void conv_transpose_2D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  return dispatch_slow_conv_transpose_2D(
      in, wt, out, padding, wt_strides, wt_dilation);
}

} // namespace

void TransposeConvolution::eval(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& in = inputs[0];
  auto& wt = inputs[1];

  // 2D convolution
  if (in.ndim() == (2 + 2)) {
    return conv_transpose_2D_cpu(
        in, wt, out, padding_, kernel_strides_, kernel_dilation_);
  }
  // 1D convolution
  // else if (in.ndim() == (1 + 2)) {
  //   return conv_1D_cpu(
  //       in, wt, out, padding_, kernel_strides_, kernel_dilation_);
  // }
  // Throw error
  else {
    std::ostringstream msg;
    msg << "[TransposeConvolution::eval] Transpose Convolution currently only supports"
        << " 2D convolutions. Got inputs with " << in.ndim() - 2
        << " spatial dimensions";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace mlx::core
