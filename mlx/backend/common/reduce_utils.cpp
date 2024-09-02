// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/reduce.h"

namespace mlx::core {

std::pair<std::vector<int>, std::vector<size_t>> shapes_without_reduction_axes(
    const array& x,
    const std::vector<int>& axes) {
  std::vector<int> shape = x.shape();
  std::vector<size_t> strides = x.strides();

  for (int i = axes.size() - 1; i >= 0; i--) {
    int a = axes[i];
    shape.erase(shape.begin() + a);
    strides.erase(strides.begin() + a);
  }

  return std::make_pair(shape, strides);
}

ReductionPlan get_reduction_plan(const array& x, const std::vector<int>& axes) {
  // The data is all there and we are reducing over everything
  if (x.size() == x.data_size() && axes.size() == x.ndim() &&
      x.flags().contiguous) {
    return ContiguousAllReduce;
  }

  // Row contiguous input so the output is row contiguous
  if (x.flags().row_contiguous) {
    // Merge consecutive axes
    std::vector<int> shape = {x.shape(axes[0])};
    std::vector<size_t> strides = {x.strides()[axes[0]]};
    for (int i = 1; i < axes.size(); i++) {
      if (axes[i] - 1 == axes[i - 1] && x.shape(axes[i]) > 1) {
        shape.back() *= x.shape(axes[i]);
        strides.back() = x.strides()[axes[i]];
      } else {
        shape.push_back(x.shape(axes[i]));
        strides.push_back(x.strides()[axes[i]]);
      }
    }

    // Remove singleton axes from the plan
    for (int i = shape.size() - 1; i >= 0; i--) {
      if (shape[i] == 1) {
        shape.erase(shape.begin() + i);
        strides.erase(strides.begin() + i);
      }
    }

    if (strides.back() == 1) {
      return ReductionPlan(ContiguousReduce, shape, strides);
    } else if (strides.back() > 1) {
      return ReductionPlan(ContiguousStridedReduce, shape, strides);
    }
  }

  // Let's check if we can optimize our access patterns
  //
  // 1. We have a reduction axis with stride 1. Simply call
  //    GeneralContiguousReduce and be done with it.
  // 2. We have transpositions and we are not reducing over the axis with
  //    stride 1. However, we are reducing over an axis where everything is
  //    contiguous in memory to the right of that axis. We can call strided
  //    reduce and be done with it.
  // 2. We have weird transpositions and expands. Copy the strides to the
  //    output, then call strided reduce.

  // Sort reduction axes by stride in order to merge them and figure out if we
  // have a contiguous reduction.
  std::vector<std::pair<int, size_t>> reductions;
  for (auto a : axes) {
    if (x.shape(a) > 1) {
      reductions.push_back(std::make_pair(x.shape(a), x.strides()[a]));
    }
  }
  std::sort(reductions.begin(), reductions.end(), [](auto a, auto b) {
    bool a_is_zero = a.second == 0;
    bool b_is_zero = b.second == 0;
    return (a_is_zero != b_is_zero) ? a.second < b.second : a.second > b.second;
  });
  // Extract the two smallest and try to merge them in case the contiguous
  // reduction can be bigger than just the last axis.
  for (int i = reductions.size() - 1; i >= 1; i--) {
    auto a = reductions[i];
    auto b = reductions[i - 1];

    // b.stride = a.shape * a.stride then a and b are contiguous
    if (b.second == a.first * a.second) {
      reductions.erase(reductions.begin() + i);
      reductions[i - 1] = std::make_pair(a.first * b.first, a.second);
    }
  }

  std::vector<int> shape;
  std::vector<size_t> strides;
  for (auto r : reductions) {
    shape.push_back(r.first);
    strides.push_back(r.second);
  }

  // We can call the contiguous reduction op for every weird way the input is
  // structured in the rest of the axes.
  if (strides.back() == 1) {
    return ReductionPlan(GeneralContiguousReduce, shape, strides);
  }

  // Delegate to the general strided reduction op if the axes after
  // strides.back() are contiguous.
  if (strides.back() > 1) {
    int size = 1;
    bool have_expand = false;
    for (int i = x.ndim() - 1; i >= 0; i--) {
      if (axes.back() == i) {
        continue;
      }

      size_t stride_i = x.strides()[i];
      int shape_i = x.shape(i);
      if (stride_i == 0) {
        if (shape_i == 1) {
          continue;
        }

        have_expand = true;
        break;
      }

      if (stride_i != size && shape_i != 1) {
        break;
      }
      size *= shape_i;
    }
    // In the case of an expanded dimension we are being conservative and
    // require the smallest reduction stride to be smaller than the maximum row
    // contiguous size. The reason is that we can't easily know if the reduced
    // axis is before or after an expanded dimension.
    if (size > strides.back() || (size == strides.back() && !have_expand)) {
      return ReductionPlan(GeneralStridedReduce, shape, strides);
    }
  }

  return ReductionPlan(GeneralReduce, shape, strides);
}

} // namespace mlx::core
