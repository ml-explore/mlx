// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <numeric>

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

///////////////////////////////////////////////////////////////////////////////
// Naive reference conv
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void slow_conv_1D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const T* start_wt_ptr = wt.data<T>();

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();

  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = 1 + in_dilation[0] * (in.shape(1) - 1); // Input spatial dim
  const int C = in.shape(2); // Input channels
  const int oH = out.shape(1); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int wH = wt.shape(1); // Weight spatial dim

  const int groups = C / wt.shape(2);
  const int C_per_group = wt.shape(2);
  const int O_per_group = O / groups;

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
      for (int g = 0; g < groups; ++g) {
        for (int o = g * O_per_group; o < (g + 1) * O_per_group; ++o) {
          const T* filter_wt_ptr = start_wt_ptr + o * wt_stride_O;
          float r = 0.;

          for (int wh = 0; wh < wH; ++wh) {
            const T* wt_ptr = filter_wt_ptr + wh * wt_stride_H;

            int wh_flip = flip ? (wH - wh - 1) : wh;
            int ih = oh * wt_strides[0] - padding[0] + wh_flip * wt_dilation[0];

            auto ih_div = std::div(ih, in_dilation[0]);

            if (ih >= 0 && ih < iH && ih_div.rem == 0) {
              for (int c = g * C_per_group; c < (g + 1) * C_per_group; ++c) {
                r += static_cast<float>(
                         in_ptr[ih_div.quot * in_stride_H + c * in_stride_C]) *
                    static_cast<float>(wt_ptr[(c % C_per_group) * wt_stride_C]);
              } // c

            } // ih check
          } // wh

          out_ptr[oh * out_stride_H + o * out_stride_O] = static_cast<T>(r);
        } // o
      } // g
    } // oh

    in_ptr += in_stride_N;
    out_ptr += out_stride_N;
  } // n
}

template <typename T>
void slow_conv_2D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const T* st_wt_ptr = wt.data<T>();
  const T* st_in_ptr = in.data<T>();
  T* st_out_ptr = out.data<T>();

  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = 1 + in_dilation[0] * (in.shape(1) - 1); // Input spatial dim
  const int iW = 1 + in_dilation[1] * (in.shape(2) - 1); // Input spatial dim
  const int C = in.shape(3); // In channels
  const int oH = out.shape(1); // Output spatial dim
  const int oW = out.shape(2); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int wH = wt.shape(1); // Weight spatial dim
  const int wW = wt.shape(2); // Weight spatial dim

  const int groups = C / wt.shape(3);
  const int C_per_group = wt.shape(3);
  const int O_per_group = O / groups;

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

  bool is_idil_one = in_dilation[0] == 1 && in_dilation[1] == 1;

  auto pt_conv_no_checks =
      [&](const T* in_ptr, const T* wt_ptr, T* out_ptr, int oh, int ow) {
        out_ptr += oh * out_stride_H + ow * out_stride_W;
        int ih_base = oh * wt_strides[0] - padding[0];
        int iw_base = ow * wt_strides[1] - padding[1];

        for (int g = 0; g < groups; ++g) {
          for (int o = g * O_per_group; o < (g + 1) * O_per_group; ++o) {
            float r = 0.;

            for (int wh = 0; wh < wH; ++wh) {
              for (int ww = 0; ww < wW; ++ww) {
                int wh_flip = flip ? wH - wh - 1 : wh;
                int ww_flip = flip ? wW - ww - 1 : ww;
                int ih = ih_base + wh_flip * wt_dilation[0];
                int iw = iw_base + ww_flip * wt_dilation[1];

                const T* wt_ptr_pt =
                    wt_ptr + wh * wt_stride_H + ww * wt_stride_W;
                const T* in_ptr_pt =
                    in_ptr + ih * in_stride_H + iw * in_stride_W;

                for (int c = g * C_per_group; c < (g + 1) * C_per_group; ++c) {
                  r += static_cast<float>(in_ptr_pt[c * in_stride_C]) *
                      static_cast<float>(
                           wt_ptr_pt[(c % C_per_group) * wt_stride_C]);
                } // c
              } // ww
            } // wh

            out_ptr[0] = static_cast<T>(r);
            out_ptr += out_stride_O;
            wt_ptr += wt_stride_O;
          } // o
        } // g
      };

  int jump_h = flip ? -wt_dilation[0] : wt_dilation[0];
  int jump_w = flip ? -wt_dilation[1] : wt_dilation[1];

  int init_h = (flip ? (wH - 1) * wt_dilation[0] : 0);
  int init_w = (flip ? (wW - 1) * wt_dilation[1] : 0);

  int f_wgt_jump_h = std::lcm(in_dilation[0], wt_dilation[0]) / wt_dilation[0];
  int f_wgt_jump_w = std::lcm(in_dilation[1], wt_dilation[1]) / wt_dilation[1];

  int f_out_jump_h = std::lcm(in_dilation[0], wt_strides[0]) / wt_strides[0];
  int f_out_jump_w = std::lcm(in_dilation[1], wt_strides[1]) / wt_strides[1];

  std::vector<int> base_h(f_out_jump_h);
  std::vector<int> base_w(f_out_jump_w);

  for (int i = 0; i < f_out_jump_h; ++i) {
    int ih_loop = i * wt_strides[0] - padding[0] + init_h;

    int wh_base = 0;
    while (wh_base < wH && ih_loop % in_dilation[0] != 0) {
      wh_base++;
      ih_loop += jump_h;
    }

    base_h[i] = wh_base;
  }

  for (int j = 0; j < f_out_jump_w; ++j) {
    int iw_loop = j * wt_strides[1] - padding[1] + init_w;

    int ww_base = 0;
    while (ww_base < wW && iw_loop % in_dilation[1] != 0) {
      ww_base++;
      iw_loop += jump_w;
    }

    base_w[j] = ww_base;
  }

  auto pt_conv_all_checks =
      [&](const T* in_ptr, const T* wt_ptr, T* out_ptr, int oh, int ow) {
        out_ptr += oh * out_stride_H + ow * out_stride_W;

        int ih_base = oh * wt_strides[0] - padding[0];
        int iw_base = ow * wt_strides[1] - padding[1];

        int wh_base = base_h[oh % f_out_jump_h];
        int ww_base = base_w[ow % f_out_jump_w];

        for (int g = 0; g < groups; ++g) {
          for (int o = g * O_per_group; o < (g + 1) * O_per_group; ++o) {
            float r = 0.;

            for (int wh = wh_base; wh < wH; wh += f_wgt_jump_h) {
              for (int ww = ww_base; ww < wW; ww += f_wgt_jump_w) {
                int wh_flip = flip ? wH - wh - 1 : wh;
                int ww_flip = flip ? wW - ww - 1 : ww;
                int ih = ih_base + wh_flip * wt_dilation[0];
                int iw = iw_base + ww_flip * wt_dilation[1];

                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                  const T* wt_ptr_pt =
                      wt_ptr + wh * wt_stride_H + ww * wt_stride_W;

                  int ih_dil = !is_idil_one ? (ih / in_dilation[0]) : ih;
                  int iw_dil = !is_idil_one ? (iw / in_dilation[1]) : iw;

                  const T* in_ptr_pt =
                      in_ptr + ih_dil * in_stride_H + iw_dil * in_stride_W;

                  for (int c = g * C_per_group; c < (g + 1) * C_per_group;
                       ++c) {
                    r += static_cast<float>(in_ptr_pt[c * in_stride_C]) *
                        static_cast<float>(
                             wt_ptr_pt[(c % C_per_group) * wt_stride_C]);
                  } // c

                } // ih, iw check
              } // ww
            } // wh

            out_ptr[0] = static_cast<T>(r);
            out_ptr += out_stride_O;
            wt_ptr += wt_stride_O;
          } // o
        } // g
      };

  int oH_border_0 = 0;
  int oH_border_1 =
      is_idil_one ? ((padding[0] + wt_strides[0] - 1) / wt_strides[0]) : oH;
  int oH_border_2 = std::max(
      oH_border_1, (iH + padding[0] - wH * wt_dilation[0]) / wt_strides[0]);
  int oH_border_3 = oH;

  int oW_border_0 = 0;
  int oW_border_1 =
      is_idil_one ? ((padding[1] + wt_strides[1] - 1) / wt_strides[1]) : oW;
  int oW_border_2 = std::max(
      oW_border_1, (iW + padding[1] - wW * wt_dilation[1]) / wt_strides[1]);
  int oW_border_3 = oW;

  for (int n = 0; n < N; ++n) {
    // Case 1: oh might put us out of bounds
    for (int oh = oH_border_0; oh < oH_border_1; ++oh) {
      for (int ow = 0; ow < oW; ++ow) {
        pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow
    } // oh

    // Case 2: oh in bounds
    for (int oh = oH_border_1; oh < oH_border_2; ++oh) {
      // Case a: ow might put us out of bounds
      for (int ow = oW_border_0; ow < oW_border_1; ++ow) {
        pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

      // Case b: ow in bounds
      for (int ow = oW_border_1; ow < oW_border_2; ++ow) {
        pt_conv_no_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

      // Case c: ow might put us out of bounds
      for (int ow = oW_border_2; ow < oW_border_3; ++ow) {
        pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow

    } // oh

    // Case 3: oh might put us out of bounds
    for (int oh = oH_border_2; oh < oH_border_3; ++oh) {
      for (int ow = 0; ow < oW; ++ow) {
        pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, oh, ow);
      } // ow
    } // oh

    st_in_ptr += in_stride_N;
    st_out_ptr += out_stride_N;

  } // n
}

template <typename T>
void slow_conv_3D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const T* st_wt_ptr = wt.data<T>();
  const T* st_in_ptr = in.data<T>();
  T* st_out_ptr = out.data<T>();

  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iD = 1 + in_dilation[0] * (in.shape(1) - 1); // Input spatial dim
  const int iH = 1 + in_dilation[1] * (in.shape(2) - 1); // Input spatial dim
  const int iW = 1 + in_dilation[2] * (in.shape(3) - 1); // Input spatial dim
  const int oD = out.shape(1); // Output spatial dim
  const int oH = out.shape(2); // Output spatial dim
  const int oW = out.shape(3); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int C = wt.shape(4); // In channels
  const int wD = wt.shape(1); // Weight spatial dim
  const int wH = wt.shape(2); // Weight spatial dim
  const int wW = wt.shape(3); // Weight spatial dim

  const size_t in_stride_N = in.strides()[0];
  const size_t in_stride_D = in.strides()[1];
  const size_t in_stride_H = in.strides()[2];
  const size_t in_stride_W = in.strides()[3];
  const size_t in_stride_C = in.strides()[4];

  const size_t wt_stride_O = wt.strides()[0];
  const size_t wt_stride_D = wt.strides()[1];
  const size_t wt_stride_H = wt.strides()[2];
  const size_t wt_stride_W = wt.strides()[3];
  const size_t wt_stride_C = wt.strides()[4];

  const size_t out_stride_N = out.strides()[0];
  const size_t out_stride_D = out.strides()[1];
  const size_t out_stride_H = out.strides()[2];
  const size_t out_stride_W = out.strides()[3];
  const size_t out_stride_O = out.strides()[4];

  bool is_idil_one =
      in_dilation[0] == 1 && in_dilation[1] == 1 && in_dilation[2] == 1;

  auto pt_conv_no_checks = [&](const T* in_ptr,
                               const T* wt_ptr,
                               T* out_ptr,
                               int od,
                               int oh,
                               int ow) {
    out_ptr += od * out_stride_D + oh * out_stride_H + ow * out_stride_W;
    int id_base = od * wt_strides[0] - padding[0];
    int ih_base = oh * wt_strides[1] - padding[1];
    int iw_base = ow * wt_strides[2] - padding[2];

    for (int o = 0; o < O; ++o) {
      float r = 0.;

      for (int wd = 0; wd < wD; ++wd) {
        for (int wh = 0; wh < wH; ++wh) {
          for (int ww = 0; ww < wW; ++ww) {
            int wd_flip = flip ? wD - wd - 1 : wd;
            int wh_flip = flip ? wH - wh - 1 : wh;
            int ww_flip = flip ? wW - ww - 1 : ww;
            int id = id_base + wd_flip * wt_dilation[0];
            int ih = ih_base + wh_flip * wt_dilation[1];
            int iw = iw_base + ww_flip * wt_dilation[2];

            const T* wt_ptr_pt =
                wt_ptr + wd * wt_stride_D + wh * wt_stride_H + ww * wt_stride_W;
            const T* in_ptr_pt =
                in_ptr + id * in_stride_D + ih * in_stride_H + iw * in_stride_W;

            for (int c = 0; c < C; ++c) {
              r += static_cast<float>(in_ptr_pt[0]) *
                  static_cast<float>(wt_ptr_pt[0]);
              in_ptr_pt += in_stride_C;
              wt_ptr_pt += wt_stride_C;
            } // c

          } // ww
        } // wh
      } // wd

      out_ptr[0] = static_cast<T>(r);
      out_ptr += out_stride_O;
      wt_ptr += wt_stride_O;
    } // o
  };

  int jump_d = flip ? -wt_dilation[0] : wt_dilation[0];
  int jump_h = flip ? -wt_dilation[1] : wt_dilation[1];
  int jump_w = flip ? -wt_dilation[2] : wt_dilation[2];

  int init_d = (flip ? (wD - 1) * wt_dilation[0] : 0);
  int init_h = (flip ? (wH - 1) * wt_dilation[1] : 0);
  int init_w = (flip ? (wW - 1) * wt_dilation[2] : 0);

  int f_wgt_jump_d = std::lcm(in_dilation[0], wt_dilation[0]) / wt_dilation[0];
  int f_wgt_jump_h = std::lcm(in_dilation[1], wt_dilation[1]) / wt_dilation[1];
  int f_wgt_jump_w = std::lcm(in_dilation[2], wt_dilation[2]) / wt_dilation[2];

  int f_out_jump_d = std::lcm(in_dilation[0], wt_strides[0]) / wt_strides[0];
  int f_out_jump_h = std::lcm(in_dilation[1], wt_strides[1]) / wt_strides[1];
  int f_out_jump_w = std::lcm(in_dilation[2], wt_strides[2]) / wt_strides[2];

  std::vector<int> base_d(f_out_jump_d);
  std::vector<int> base_h(f_out_jump_h);
  std::vector<int> base_w(f_out_jump_w);

  for (int i = 0; i < f_out_jump_d; ++i) {
    int id_loop = i * wt_strides[0] - padding[0] + init_d;

    int wd_base = 0;
    while (wd_base < wD && id_loop % in_dilation[0] != 0) {
      wd_base++;
      id_loop += jump_d;
    }

    base_d[i] = wd_base;
  }

  for (int i = 0; i < f_out_jump_h; ++i) {
    int ih_loop = i * wt_strides[1] - padding[1] + init_h;

    int wh_base = 0;
    while (wh_base < wH && ih_loop % in_dilation[1] != 0) {
      wh_base++;
      ih_loop += jump_h;
    }

    base_h[i] = wh_base;
  }

  for (int j = 0; j < f_out_jump_w; ++j) {
    int iw_loop = j * wt_strides[2] - padding[2] + init_w;

    int ww_base = 0;
    while (ww_base < wW && iw_loop % in_dilation[2] != 0) {
      ww_base++;
      iw_loop += jump_w;
    }

    base_w[j] = ww_base;
  }

  auto pt_conv_all_checks = [&](const T* in_ptr,
                                const T* wt_ptr,
                                T* out_ptr,
                                int od,
                                int oh,
                                int ow) {
    out_ptr += od * out_stride_D + oh * out_stride_H + ow * out_stride_W;

    int id_base = od * wt_strides[0] - padding[0];
    int ih_base = oh * wt_strides[1] - padding[1];
    int iw_base = ow * wt_strides[2] - padding[2];

    int wd_base = base_d[od % f_out_jump_d];
    int wh_base = base_h[oh % f_out_jump_h];
    int ww_base = base_w[ow % f_out_jump_w];

    for (int o = 0; o < O; ++o) {
      float r = 0.;

      for (int wd = wd_base; wd < wD; wd += f_wgt_jump_d) {
        for (int wh = wh_base; wh < wH; wh += f_wgt_jump_h) {
          for (int ww = ww_base; ww < wW; ww += f_wgt_jump_w) {
            int wd_flip = flip ? wD - wd - 1 : wd;
            int wh_flip = flip ? wH - wh - 1 : wh;
            int ww_flip = flip ? wW - ww - 1 : ww;
            int id = id_base + wd_flip * wt_dilation[0];
            int ih = ih_base + wh_flip * wt_dilation[1];
            int iw = iw_base + ww_flip * wt_dilation[2];

            if (id >= 0 && id < iD && ih >= 0 && ih < iH && iw >= 0 &&
                iw < iW) {
              const T* wt_ptr_pt = wt_ptr + wd * wt_stride_D +
                  wh * wt_stride_H + ww * wt_stride_W;

              int id_dil = !is_idil_one ? (id / in_dilation[0]) : id;
              int ih_dil = !is_idil_one ? (ih / in_dilation[1]) : ih;
              int iw_dil = !is_idil_one ? (iw / in_dilation[2]) : iw;

              const T* in_ptr_pt = in_ptr + id_dil * in_stride_D +
                  ih_dil * in_stride_H + iw_dil * in_stride_W;

              for (int c = 0; c < C; ++c) {
                r += static_cast<float>(in_ptr_pt[0]) *
                    static_cast<float>(wt_ptr_pt[0]);
                in_ptr_pt += in_stride_C;
                wt_ptr_pt += wt_stride_C;
              } // c

            } // iD, ih, iw check
          } // ww
        } // wh
      } // wd

      out_ptr[0] = static_cast<T>(r);
      out_ptr += out_stride_O;
      wt_ptr += wt_stride_O;
    } // o
  };

  int oD_border_0 = 0;
  int oD_border_1 =
      is_idil_one ? ((padding[0] + wt_strides[0] - 1) / wt_strides[0]) : oD;
  int oD_border_2 = std::max(
      oD_border_1, (iD + padding[0] - wD * wt_dilation[0]) / wt_strides[0]);
  int oD_border_3 = oD;

  int oH_border_0 = 0;
  int oH_border_1 =
      is_idil_one ? ((padding[1] + wt_strides[1] - 1) / wt_strides[1]) : oH;
  int oH_border_2 = std::max(
      oH_border_1, (iH + padding[1] - wH * wt_dilation[1]) / wt_strides[1]);
  int oH_border_3 = oH;

  int oW_border_0 = 0;
  int oW_border_1 =
      is_idil_one ? ((padding[2] + wt_strides[2] - 1) / wt_strides[2]) : oW;
  int oW_border_2 = std::max(
      oW_border_1, (iW + padding[2] - wW * wt_dilation[2]) / wt_strides[2]);
  int oW_border_3 = oW;

  for (int n = 0; n < N; ++n) {
    // Case 1: od might put us out of bounds
    for (int od = oD_border_0; od < oD_border_1; ++od) {
      for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow
      } // oh
    } // od

    // Case 2: od in bounds
    for (int od = oD_border_1; od < oD_border_2; ++od) {
      // Case 2.1: oh might put us out of bounds
      for (int oh = oH_border_0; oh < oH_border_1; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow
      } // oh

      // Case 2.2: oh in bounds
      for (int oh = oH_border_1; oh < oH_border_2; ++oh) {
        // Case 2.2.1: ow might put us out of bounds
        for (int ow = oW_border_0; ow < oW_border_1; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow

        // Case 2.2.2: ow in bounds
        for (int ow = oW_border_1; ow < oW_border_2; ++ow) {
          pt_conv_no_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow

        // Case 2.2.3: ow might put us out of bounds
        for (int ow = oW_border_2; ow < oW_border_3; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow
      } // oh

      // Case 2.3: oh might put us out of bounds
      for (int oh = oH_border_2; oh < oH_border_3; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow
      } // oh
    } // od

    // Case 3: od might put us out of bounds
    for (int od = oD_border_2; od < oD_border_3; ++od) {
      for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
          pt_conv_all_checks(st_in_ptr, st_wt_ptr, st_out_ptr, od, oh, ow);
        } // ow
      } // oh
    } // od

    st_in_ptr += in_stride_N;
    st_out_ptr += out_stride_N;

  } // n
}

void dispatch_slow_conv_1D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  if (in.dtype() == float32) {
    return slow_conv_1D<float>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == float16) {
    return slow_conv_1D<float16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == bfloat16) {
    return slow_conv_1D<bfloat16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else {
    throw std::invalid_argument(
        "[Convolution::eval] got unsupported data type.");
  }
}

void dispatch_slow_conv_2D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  if (in.dtype() == float32) {
    return slow_conv_2D<float>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == float16) {
    return slow_conv_2D<float16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == bfloat16) {
    return slow_conv_2D<bfloat16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else {
    throw std::invalid_argument(
        "[Convolution::eval] got unsupported data type.");
  }
}

void dispatch_slow_conv_3D(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  if (in.dtype() == float32) {
    return slow_conv_3D<float>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == float16) {
    return slow_conv_3D<float16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else if (in.dtype() == bfloat16) {
    return slow_conv_3D<bfloat16_t>(
        in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
  } else {
    throw std::invalid_argument(
        "[Convolution::eval] got unsupported data type.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Explicit gemm conv
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void flip_spatial_dims_inplace(array& wt) {
  T* x = wt.data<T>();
  size_t out_channels = wt.shape(0);
  size_t in_channels = wt.shape(-1);

  // Calculate the total size of the spatial dimensions
  int spatial_size = 1;
  for (int d = 1; d < wt.ndim() - 1; ++d) {
    spatial_size *= wt.shape(d);
  }

  for (size_t i = 0; i < out_channels; i++) {
    T* top = x + i * spatial_size * in_channels;
    T* bottom =
        x + i * spatial_size * in_channels + (spatial_size - 1) * in_channels;
    for (size_t j = 0; j < spatial_size / 2; j++) {
      for (size_t k = 0; k < in_channels; k++) {
        std::swap(top[k], bottom[k]);
      }
      top += in_channels;
      bottom -= in_channels;
    }
  }
}

void explicit_gemm_conv_1D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = in.shape(1); // Input spatial dim
  const int C = in.shape(2); // Input channels
  const int oH = out.shape(1); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int wH = wt.shape(1); // Weight spatial dim

  const int groups = C / wt.shape(2);
  const int C_per_group = wt.shape(2);
  const int O_per_group = O / groups;

  auto conv_dtype = float32;

  // Pad input
  std::vector<int> padded_shape = {N, iH + 2 * padding[0], C};
  array in_padded(padded_shape, conv_dtype, nullptr, {});

  // Fill with zeros
  copy(array(0, conv_dtype), in_padded, CopyType::Scalar);

  // Pick input slice from padded
  size_t data_offset = padding[0] * in_padded.strides()[1];
  array in_padded_slice(in.shape(), in_padded.dtype(), nullptr, {});
  in_padded_slice.copy_shared_buffer(
      in_padded,
      in_padded.strides(),
      in_padded.flags(),
      in_padded_slice.size(),
      data_offset);

  // Copy input values into the slice
  copy_inplace(in, in_padded_slice, CopyType::GeneralGeneral);

  // Make strided view
  std::vector<int> strided_shape = {N, oH, wH, C};

  std::vector<size_t> strided_strides = {
      in_padded.strides()[0],
      in_padded.strides()[1] * wt_strides[0],
      in_padded.strides()[1],
      in_padded.strides()[2]};
  auto flags = in_padded.flags();
  if (groups > 1) {
    // Transpose the last two dimensions for grouped convolutions
    std::swap(strided_shape[2], strided_shape[3]);
    std::swap(strided_strides[2], strided_strides[3]);
  }

  array in_strided_view(strided_shape, in_padded.dtype(), nullptr, {});
  in_strided_view.copy_shared_buffer(
      in_padded, strided_strides, flags, in_strided_view.size(), 0);

  // Materialize strided view
  std::vector<int> strided_reshape = {N * oH, wH * C};
  array in_strided(strided_reshape, in_strided_view.dtype(), nullptr, {});
  copy(in_strided_view, in_strided, CopyType::General);

  // Check wt dtype and prepare
  auto gemm_wt = wt;
  auto gemm_out = out;

  if (groups > 1) {
    // Transpose the last two dimensions for grouped convolutions
    array wt_transpose(
        {wt.shape(0), wt.shape(2), wt.shape(1)}, wt.dtype(), nullptr, {});
    wt_transpose.copy_shared_buffer(
        wt,
        {wt.strides(0), wt.strides(2), wt.strides(1)},
        wt.flags(),
        wt.size(),
        0);
    gemm_wt = array(wt_transpose.shape(), float32, nullptr, {});
    copy(wt_transpose, gemm_wt, CopyType::General);
  } else if (wt.dtype() != float32 || !wt.flags().row_contiguous) {
    auto ctype =
        wt.flags().row_contiguous ? CopyType::Vector : CopyType::General;
    gemm_wt = array(wt.shape(), float32, nullptr, {});
    copy(wt, gemm_wt, ctype);
  }

  if (out.dtype() != float32) {
    gemm_out = array(out.shape(), float32, nullptr, {});
    gemm_out.set_data(allocator::malloc_or_wait(gemm_out.nbytes()));
  }

  for (int g = 0; g < groups; ++g) {
    // Perform gemm
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, // no trans A
        CblasTrans, // transB
        strided_reshape[0], // M
        O_per_group, // N
        C_per_group * wH, // K
        1.0f, // alpha
        in_strided.data<float>() + g * C_per_group * wH, // A
        wH * C, // lda
        gemm_wt.data<float>() + g * O_per_group * C_per_group * wH, // B
        wH * C_per_group, // ldb
        0.0f, // beta
        gemm_out.data<float>() + g * O_per_group, // C
        O // ldc
    );

    // Copy results if needed
    if (out.dtype() != float32) {
      copy(gemm_out, out, CopyType::Vector);
    }
  }
}

void explicit_gemm_conv_2D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation) {
  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const int iH = in.shape(1); // Input spatial dim
  const int iW = in.shape(2); // Input spatial dim
  const int oH = out.shape(1); // Output spatial dim
  const int oW = out.shape(2); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int C = wt.shape(3); // In channels
  const int wH = wt.shape(1); // Weight spatial dim
  const int wW = wt.shape(2); // Weight spatial dim

  auto conv_dtype = out.dtype();

  // Pad input
  std::vector<int> padded_shape = {
      N, iH + 2 * padding[0], iW + 2 * padding[1], C};
  array in_padded(padded_shape, conv_dtype, nullptr, {});

  // Fill with zeros
  copy(array(0, conv_dtype), in_padded, CopyType::Scalar);

  // Pick input slice from padded
  size_t data_offset =
      padding[0] * in_padded.strides()[1] + padding[1] * in_padded.strides()[2];
  array in_padded_slice(in.shape(), in_padded.dtype(), nullptr, {});
  in_padded_slice.copy_shared_buffer(
      in_padded,
      in_padded.strides(),
      in_padded.flags(),
      in_padded_slice.size(),
      data_offset);

  // Copy input values into the slice
  copy_inplace(in, in_padded_slice, CopyType::GeneralGeneral);

  // Make strided view
  std::vector<int> strided_shape = {N, oH, oW, wH, wW, C};

  std::vector<size_t> strided_strides = {
      in_padded.strides()[0],
      in_padded.strides()[1] * wt_strides[0],
      in_padded.strides()[2] * wt_strides[1],
      in_padded.strides()[1],
      in_padded.strides()[2],
      in_padded.strides()[3]};
  auto flags = in_padded.flags();

  array in_strided_view(strided_shape, in_padded.dtype(), nullptr, {});
  in_strided_view.copy_shared_buffer(
      in_padded, strided_strides, flags, in_strided_view.size(), 0);

  // Materialize strided view
  std::vector<int> strided_reshape = {N * oH * oW, wH * wW * C};
  array in_strided(strided_reshape, in_strided_view.dtype(), nullptr, {});
  copy(in_strided_view, in_strided, CopyType::General);

  // Check wt dtype and prepare
  auto gemm_wt = wt;
  auto gemm_out = out;

  if (wt.dtype() != float32 || !wt.flags().row_contiguous) {
    auto ctype =
        wt.flags().row_contiguous ? CopyType::Vector : CopyType::General;
    gemm_wt = array(wt.shape(), float32, nullptr, {});
    copy(wt, gemm_wt, ctype);
  }

  if (out.dtype() != float32) {
    gemm_out = array(out.shape(), float32, nullptr, {});
    gemm_out.set_data(allocator::malloc_or_wait(gemm_out.nbytes()));
  }

  // Perform gemm
  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans, // no trans A
      CblasTrans, // transB
      strided_reshape[0], // M
      O, // N
      strided_reshape[1], // K
      1.0f, // alpha
      in_strided.data<float>(),
      strided_reshape[1], // lda
      gemm_wt.data<float>(),
      strided_reshape[1], // ldb
      0.0f, // beta
      gemm_out.data<float>(),
      O // ldc
  );

  // Copy results if needed
  if (out.dtype() != float32) {
    copy(gemm_out, out, CopyType::Vector);
  }
}

void explicit_gemm_conv_ND_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const bool flip) {
  const int N = in.shape(0); // Batch size, should be the same as out.shape(0)
  const auto iDim = std::vector<int>(
      in.shape().begin() + 1, in.shape().end() - 1); // Input spatial dim
  const auto oDim = std::vector<int>(
      out.shape().begin() + 1, out.shape().end() - 1); // Output spatial dim
  const int O = wt.shape(0); // Out channels
  const int C = wt.shape(-1); // In channels
  const auto wDim = std::vector<int>(
      wt.shape().begin() + 1, wt.shape().end() - 1); // Weight spatial dim

  auto conv_dtype = float32;

  // Pad input
  std::vector<int> padded_shape(in.shape().size());
  padded_shape.front() = N;
  for (size_t i = 0; i < iDim.size(); i++) {
    padded_shape[i + 1] = iDim[i] + 2 * padding[i];
  }
  padded_shape.back() = C;
  array in_padded(padded_shape, conv_dtype, nullptr, {});

  // Fill with zeros
  copy(array(0, conv_dtype), in_padded, CopyType::Scalar);

  // Pick input slice from padded
  size_t data_offset = 0;
  for (size_t i = 0; i < padding.size(); i++) {
    data_offset += padding[i] * in_padded.strides()[i + 1];
  }
  array in_padded_slice(in.shape(), in_padded.dtype(), nullptr, {});
  in_padded_slice.copy_shared_buffer(
      in_padded,
      in_padded.strides(),
      in_padded.flags(),
      in_padded_slice.size(),
      data_offset);

  // Copy input values into the slice
  copy_inplace(in, in_padded_slice, CopyType::GeneralGeneral);

  // Make strided view
  std::vector<int> strided_shape(oDim.size() + wDim.size() + 2);
  strided_shape.front() = N;
  for (size_t i = 0; i < oDim.size(); i++) {
    strided_shape[i + 1] = oDim[i];
  }
  for (size_t i = 0; i < wDim.size(); i++) {
    strided_shape[i + 1 + oDim.size()] = wDim[i];
  }
  strided_shape.back() = C;

  std::vector<size_t> strided_strides(in.shape().size() * 2 - 2);
  strided_strides[0] = in_padded.strides()[0];
  for (size_t i = 0; i < wt_strides.size(); i++) {
    strided_strides[i + 1] = in_padded.strides()[i + 1] * wt_strides[i];
  }
  for (size_t i = 1; i < in_padded.strides().size(); i++) {
    strided_strides[i + wt_strides.size()] = in_padded.strides()[i];
  }

  auto flags = in_padded.flags();

  array in_strided_view(strided_shape, in_padded.dtype(), nullptr, {});
  in_strided_view.copy_shared_buffer(
      in_padded, strided_strides, flags, in_strided_view.size(), 0);

  // Materialize strided view
  std::vector<int> strided_reshape = {N, C};
  for (const auto& o : oDim) {
    strided_reshape[0] *= o;
  }
  for (const auto& w : wDim) {
    strided_reshape[1] *= w;
  }

  array in_strided(strided_reshape, in_strided_view.dtype(), nullptr, {});
  copy(in_strided_view, in_strided, CopyType::General);

  // Check wt dtype and prepare
  auto gemm_wt = wt;
  auto gemm_out = out;

  if (wt.dtype() != float32 || !wt.flags().row_contiguous) {
    auto ctype =
        wt.flags().row_contiguous ? CopyType::Vector : CopyType::General;
    gemm_wt = array(wt.shape(), float32, nullptr, {});
    copy(wt, gemm_wt, ctype);
  }

  if (flip) {
    auto gemm_wt_ = array(gemm_wt.shape(), float32, nullptr, {});
    copy(gemm_wt, gemm_wt_, CopyType::Vector);

    flip_spatial_dims_inplace<float>(gemm_wt_);
    gemm_wt = gemm_wt_;
  }

  if (out.dtype() != float32) {
    gemm_out = array(out.shape(), float32, nullptr, {});
    gemm_out.set_data(allocator::malloc_or_wait(gemm_out.nbytes()));
  }

  // Perform gemm
  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans, // no trans A
      CblasTrans, // transB
      strided_reshape[0], // M
      O, // N
      strided_reshape[1], // K
      1.0f, // alpha
      in_strided.data<float>(),
      strided_reshape[1], // lda
      gemm_wt.data<float>(),
      strided_reshape[1], // ldb
      0.0f, // beta
      gemm_out.data<float>(),
      O // ldc
  );

  // Copy results if needed
  if (out.dtype() != float32) {
    copy(gemm_out, out, CopyType::Vector);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Conv routing
///////////////////////////////////////////////////////////////////////////////

void conv_1D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const int groups = in.shape().back() / wt.shape().back();
  if (wt_dilation[0] == 1 && in_dilation[0] == 1 && !flip) {
    return explicit_gemm_conv_1D_cpu(
        in, wt, out, padding, wt_strides, wt_dilation);
  }
  if (wt_dilation[0] == 1 && in_dilation[0] == 1 && groups == 1) {
    return explicit_gemm_conv_ND_cpu(
        in, wt, out, padding, wt_strides, wt_dilation, flip);
  }

  return dispatch_slow_conv_1D(
      in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
}

void conv_2D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const int groups = in.shape().back() / wt.shape().back();
  if (wt_dilation[0] == 1 && wt_dilation[1] == 1 && in_dilation[0] == 1 &&
      in_dilation[1] == 1 && groups == 1) {
    return explicit_gemm_conv_ND_cpu(
        in, wt, out, padding, wt_strides, wt_dilation, flip);
  }

  return dispatch_slow_conv_2D(
      in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
}

void conv_3D_cpu(
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip) {
  const int groups = in.shape().back() / wt.shape().back();
  if (wt_dilation[0] == 1 && wt_dilation[1] == 1 && wt_dilation[2] == 1 &&
      in_dilation[0] == 1 && in_dilation[1] == 1 && in_dilation[2] == 1 &&
      groups == 1) {
    return explicit_gemm_conv_ND_cpu(
        in, wt, out, padding, wt_strides, wt_dilation, flip);
  }

  return dispatch_slow_conv_3D(
      in, wt, out, padding, wt_strides, wt_dilation, in_dilation, flip);
}

} // namespace

void Convolution::eval(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& in = inputs[0];
  auto& wt = inputs[1];

  // 3D convolution
  if (in.ndim() == (3 + 2)) {
    return conv_3D_cpu(
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        flip_);
  }
  // 2D convolution
  else if (in.ndim() == (2 + 2)) {
    return conv_2D_cpu(
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        flip_);
  }
  // 1D convolution
  else if (in.ndim() == (1 + 2)) {
    return conv_1D_cpu(
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        flip_);
  }
  // Throw error
  else {
    std::ostringstream msg;
    msg << "[Convolution::eval] Convolution currently only supports"
        << " 1D, 2D and 3D convolutions. Got inputs with " << in.ndim() - 2
        << " spatial dimensions";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace mlx::core
