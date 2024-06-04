/* Radix kernels

We provide optimized, single threaded Radix codelets
for n=2,3,4,5,6,7,8,10,11,12,13.

For n=2,3,4,5,6 we hand write the codelets.
For n=8,10,12 we combine smaller codelets.
For n=7,11,13 we use Rader's algorithm which decomposes
them into (n-1)=6,10,12 codelets. */

#include <metal_common>

#include "mlx/backend/metal/kernels/utils.h"

METAL_FUNC void radix2(thread float2* x, thread float2* y) {
  y[0] = x[0] + x[1];
  y[1] = x[0] - x[1];
}

METAL_FUNC void radix3(thread float2* x, thread float2* y) {
  float pi_2_3 = -0.8660254037844387;

  float2 a_1 = x[1] + x[2];
  float2 a_2 = x[1] - x[2];

  y[0] = x[0] + a_1;
  float2 b_1 = x[0] - 0.5 * a_1;
  float2 b_2 = pi_2_3 * a_2;

  float2 b_2_j = {-b_2.y, b_2.x};
  y[1] = b_1 + b_2_j;
  y[2] = b_1 - b_2_j;
}

METAL_FUNC void radix4(thread float2* x, thread float2* y) {
  float2 z_0 = x[0] + x[2];
  float2 z_1 = x[0] - x[2];
  float2 z_2 = x[1] + x[3];
  float2 z_3 = x[1] - x[3];
  float2 z_3_i = {z_3.y, -z_3.x};

  y[0] = z_0 + z_2;
  y[1] = z_1 + z_3_i;
  y[2] = z_0 - z_2;
  y[3] = z_1 - z_3_i;
}

METAL_FUNC void radix5(thread float2* x, thread float2* y) {
  float2 root_5_4 = 0.5590169943749475;
  float2 sin_2pi_5 = 0.9510565162951535;
  float2 sin_1pi_5 = 0.5877852522924731;

  float2 a_1 = x[1] + x[4];
  float2 a_2 = x[2] + x[3];
  float2 a_3 = x[1] - x[4];
  float2 a_4 = x[2] - x[3];

  float2 a_5 = a_1 + a_2;
  float2 a_6 = root_5_4 * (a_1 - a_2);
  float2 a_7 = x[0] - a_5 / 4;
  float2 a_8 = a_7 + a_6;
  float2 a_9 = a_7 - a_6;
  float2 a_10 = sin_2pi_5 * a_3 + sin_1pi_5 * a_4;
  float2 a_11 = sin_1pi_5 * a_3 - sin_2pi_5 * a_4;
  float2 a_10_j = {a_10.y, -a_10.x};
  float2 a_11_j = {a_11.y, -a_11.x};

  y[0] = x[0] + a_5;
  y[1] = a_8 + a_10_j;
  y[2] = a_9 + a_11_j;
  y[3] = a_9 - a_11_j;
  y[4] = a_8 - a_10_j;
}

METAL_FUNC void radix6(thread float2* x, thread float2* y) {
  float sin_pi_3 = 0.8660254037844387;
  float2 a_1 = x[2] + x[4];
  float2 a_2 = x[0] - a_1 / 2;
  float2 a_3 = sin_pi_3 * (x[2] - x[4]);
  float2 a_4 = x[5] + x[1];
  float2 a_5 = x[3] - a_4 / 2;
  float2 a_6 = sin_pi_3 * (x[5] - x[1]);
  float2 a_7 = x[0] + a_1;

  float2 a_3_i = {a_3.y, -a_3.x};
  float2 a_6_i = {a_6.y, -a_6.x};
  float2 a_8 = a_2 + a_3_i;
  float2 a_9 = a_2 - a_3_i;
  float2 a_10 = x[3] + a_4;
  float2 a_11 = a_5 + a_6_i;
  float2 a_12 = a_5 - a_6_i;

  y[0] = a_7 + a_10;
  y[1] = a_8 - a_11;
  y[2] = a_9 + a_12;
  y[3] = a_7 - a_10;
  y[4] = a_8 + a_11;
  y[5] = a_9 - a_12;
}

METAL_FUNC void radix7(thread float2* x, thread float2* y) {
  // Rader's algorithm
  float2 inv = {1 / 6.0, -1 / 6.0};

  // fft
  float2 in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
  radix6(in1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(2.44013336, -1.02261879));
  y[3] = complex_mul_conj(y[3], float2(2.37046941, -1.17510629));
  y[4] = complex_mul_conj(y[4], float2(0, -2.64575131));
  y[5] = complex_mul_conj(y[5], float2(2.37046941, 1.17510629));
  y[6] = complex_mul_conj(y[6], float2(-2.44013336, -1.02261879));

  // ifft
  radix6(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[5] = x[2] * inv + x[0];
  y[4] = x[3] * inv + x[0];
  y[6] = x[4] * inv + x[0];
  y[2] = x[5] * inv + x[0];
  y[3] = x[6] * inv + x[0];
}

METAL_FUNC void radix8(thread float2* x, thread float2* y) {
  float cos_pi_4 = 0.7071067811865476;
  float2 w_0 = {cos_pi_4, -cos_pi_4};
  float2 w_1 = {-cos_pi_4, -cos_pi_4};
  float2 temp[8] = {x[0], x[2], x[4], x[6], x[1], x[3], x[5], x[7]};
  radix4(temp, x);
  radix4(temp + 4, x + 4);

  y[0] = x[0] + x[4];
  y[4] = x[0] - x[4];
  float2 x_5 = complex_mul(x[5], w_0);
  y[1] = x[1] + x_5;
  y[5] = x[1] - x_5;
  float2 x_6 = {x[6].y, -x[6].x};
  y[2] = x[2] + x_6;
  y[6] = x[2] - x_6;
  float2 x_7 = complex_mul(x[7], w_1);
  y[3] = x[3] + x_7;
  y[7] = x[3] - x_7;
}

template <bool raders_perm>
METAL_FUNC void radix10(thread float2* x, thread float2* y) {
  float2 w[4];
  w[0] = {0.8090169943749475, -0.5877852522924731};
  w[1] = {0.30901699437494745, -0.9510565162951535};
  w[2] = {-w[1].x, w[1].y};
  w[3] = {-w[0].x, w[0].y};

  if (raders_perm) {
    float2 temp[10] = {
        x[0], x[3], x[4], x[8], x[2], x[1], x[7], x[9], x[6], x[5]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  } else {
    float2 temp[10] = {
        x[0], x[2], x[4], x[6], x[8], x[1], x[3], x[5], x[7], x[9]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  }

  y[0] = x[0] + x[5];
  y[5] = x[0] - x[5];
  for (int t = 1; t < 5; t++) {
    float2 a = complex_mul(x[t + 5], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 5] = x[t] - a;
  }
}

METAL_FUNC void radix11(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 inv = {1 / 10.0, -1 / 10.0};

  // fft
  radix10<true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(0.955301878, -3.17606649));
  y[3] = complex_mul_conj(y[3], float2(2.63610556, 2.01269656));
  y[4] = complex_mul_conj(y[4], float2(2.54127802, 2.13117479));
  y[5] = complex_mul_conj(y[5], float2(2.07016210, 2.59122150));
  y[6] = complex_mul_conj(y[6], float2(0, -3.31662479));
  y[7] = complex_mul_conj(y[7], float2(2.07016210, -2.59122150));
  y[8] = complex_mul_conj(y[8], float2(-2.54127802, 2.13117479));
  y[9] = complex_mul_conj(y[9], float2(2.63610556, -2.01269656));
  y[10] = complex_mul_conj(y[10], float2(-0.955301878, -3.17606649));

  // ifft
  radix10<false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[6] = x[2] * inv + x[0];
  y[3] = x[3] * inv + x[0];
  y[7] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[10] = x[6] * inv + x[0];
  y[5] = x[7] * inv + x[0];
  y[8] = x[8] * inv + x[0];
  y[4] = x[9] * inv + x[0];
  y[2] = x[10] * inv + x[0];
}

template <bool raders_perm>
METAL_FUNC void radix12(thread float2* x, thread float2* y) {
  float2 w[6];
  float sin_pi_3 = 0.8660254037844387;
  w[0] = {sin_pi_3, -0.5};
  w[1] = {0.5, -sin_pi_3};
  w[2] = {0, -1};
  w[3] = {-0.5, -sin_pi_3};
  w[4] = {-sin_pi_3, -0.5};

  if (raders_perm) {
    float2 temp[12] = {
        x[0],
        x[3],
        x[2],
        x[11],
        x[8],
        x[9],
        x[1],
        x[7],
        x[5],
        x[10],
        x[4],
        x[6]};
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  } else {
    float2 temp[12] = {
        x[0],
        x[2],
        x[4],
        x[6],
        x[8],
        x[10],
        x[1],
        x[3],
        x[5],
        x[7],
        x[9],
        x[11]};
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  }

  y[0] = x[0] + x[6];
  y[6] = x[0] - x[6];
  for (int t = 1; t < 6; t++) {
    float2 a = complex_mul(x[t + 6], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 6] = x[t] - a;
  }
}

METAL_FUNC void radix13(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 inv = {1 / 12.0, -1 / 12.0};

  // fft
  radix12<true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(3.07497206, -1.88269669));
  y[3] = complex_mul_conj(y[3], float2(3.09912468, 1.84266823));
  y[4] = complex_mul_conj(y[4], float2(3.45084438, -1.04483161));
  y[5] = complex_mul_conj(y[5], float2(0.91083583, 3.48860690));
  y[6] = complex_mul_conj(y[6], float2(-3.60286363, 0.139189267));
  y[7] = complex_mul_conj(y[7], float2(3.60555128, 0));
  y[8] = complex_mul_conj(y[8], float2(3.60286363, 0.139189267));
  y[9] = complex_mul_conj(y[9], float2(0.91083583, -3.48860690));
  y[10] = complex_mul_conj(y[10], float2(-3.45084438, -1.04483161));
  y[11] = complex_mul_conj(y[11], float2(3.09912468, -1.84266823));
  y[12] = complex_mul_conj(y[12], float2(-3.07497206, -1.88269669));

  // ifft
  radix12<false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[7] = x[2] * inv + x[0];
  y[10] = x[3] * inv + x[0];
  y[5] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[11] = x[6] * inv + x[0];
  y[12] = x[7] * inv + x[0];
  y[6] = x[8] * inv + x[0];
  y[3] = x[9] * inv + x[0];
  y[8] = x[10] * inv + x[0];
  y[4] = x[11] * inv + x[0];
  y[2] = x[12] * inv + x[0];
}