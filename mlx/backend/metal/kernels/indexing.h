// Copyright © 2023-2024 Apple Inc.

#include <metal_stdlib>

using namespace metal;

/////////////////////////////////////////////////////////////////////
// Indexing utils
/////////////////////////////////////////////////////////////////////

template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant size_t* strides;
  const int ndim;
};

template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, size_t size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}

#define IDX_ARG_N(idx_t, n) const device idx_t *idx##n [[buffer(n)]],

#define IDX_ARG_0(idx_t)
#define IDX_ARG_1(idx_t) IDX_ARG_0(idx_t) IDX_ARG_N(idx_t, 21)
#define IDX_ARG_2(idx_t) IDX_ARG_1(idx_t) IDX_ARG_N(idx_t, 22)
#define IDX_ARG_3(idx_t) IDX_ARG_2(idx_t) IDX_ARG_N(idx_t, 23)
#define IDX_ARG_4(idx_t) IDX_ARG_3(idx_t) IDX_ARG_N(idx_t, 24)
#define IDX_ARG_5(idx_t) IDX_ARG_4(idx_t) IDX_ARG_N(idx_t, 25)
#define IDX_ARG_6(idx_t) IDX_ARG_5(idx_t) IDX_ARG_N(idx_t, 26)
#define IDX_ARG_7(idx_t) IDX_ARG_6(idx_t) IDX_ARG_N(idx_t, 27)
#define IDX_ARG_8(idx_t) IDX_ARG_7(idx_t) IDX_ARG_N(idx_t, 28)
#define IDX_ARG_9(idx_t) IDX_ARG_8(idx_t) IDX_ARG_N(idx_t, 29)
#define IDX_ARG_10(idx_t) IDX_ARG_9(idx_t) IDX_ARG_N(idx_t, 30)

#define IDX_ARR_N(n) idx##n,

#define IDX_ARR_0()
#define IDX_ARR_1() IDX_ARR_0() IDX_ARR_N(21)
#define IDX_ARR_2() IDX_ARR_1() IDX_ARR_N(22)
#define IDX_ARR_3() IDX_ARR_2() IDX_ARR_N(23)
#define IDX_ARR_4() IDX_ARR_3() IDX_ARR_N(24)
#define IDX_ARR_5() IDX_ARR_4() IDX_ARR_N(25)
#define IDX_ARR_6() IDX_ARR_5() IDX_ARR_N(26)
#define IDX_ARR_7() IDX_ARR_6() IDX_ARR_N(27)
#define IDX_ARR_8() IDX_ARR_7() IDX_ARR_N(28)
#define IDX_ARR_9() IDX_ARR_8() IDX_ARR_N(29)
#define IDX_ARR_10() IDX_ARR_9() IDX_ARR_N(30)