// Copyright © 2023 Apple Inc.

#include <numeric>
#include <sstream>

#include "python/src/indexing.h"

#include "mlx/ops.h"

bool is_none_slice(const py::slice& in_slice) {
  return (
      py::getattr(in_slice, "start").is_none() &&
      py::getattr(in_slice, "stop").is_none() &&
      py::getattr(in_slice, "step").is_none());
}

int get_slice_int(py::object obj, int default_val) {
  if (!obj.is_none()) {
    if (!py::isinstance<py::int_>(obj)) {
      throw std::invalid_argument("Slice indices must be integers or None.");
    }
    return py::cast<int>(py::cast<py::int_>(obj));
  }
  return default_val;
}

void get_slice_params(
    int& starts,
    int& ends,
    int& strides,
    const py::slice& in_slice,
    int axis_size) {
  // Following numpy's convention
  //    Assume n is the number of elements in the dimension being sliced.
  //    Then, if i is not given it defaults to 0 for k > 0 and n - 1 for
  //    k < 0 . If j is not given it defaults to n for k > 0 and -n-1 for
  //    k < 0 . If k is not given it defaults to 1

  strides = get_slice_int(py::getattr(in_slice, "step"), 1);
  starts = get_slice_int(
      py::getattr(in_slice, "start"), strides < 0 ? axis_size - 1 : 0);
  ends = get_slice_int(
      py::getattr(in_slice, "stop"), strides < 0 ? -axis_size - 1 : axis_size);
}

array get_int_index(py::object idx, int axis_size) {
  int idx_ = py::cast<int>(idx);
  idx_ = (idx_ < 0) ? idx_ + axis_size : idx_;

  return array(idx_, uint32);
}

bool is_valid_index_type(const py::object& obj) {
  return py::isinstance<py::slice>(obj) || py::isinstance<py::int_>(obj) ||
      py::isinstance<array>(obj) || obj.is_none() || py::ellipsis().is(obj);
}

array mlx_get_item_slice(const array& src, const py::slice& in_slice) {
  // Check input and raise error if 0 dim for parity with np
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  // Return a copy of the array if none slice is request
  if (is_none_slice(in_slice)) {
    return src;
  }

  std::vector<int> starts(src.ndim(), 0);
  std::vector<int> ends = src.shape();
  std::vector<int> strides(src.ndim(), 1);

  // Check and update slice params
  get_slice_params(starts[0], ends[0], strides[0], in_slice, ends[0]);
  return slice(src, starts, ends, strides);
}

array mlx_get_item_array(const array& src, const array& indices) {
  // Check input and raise error if 0 dim for parity with np
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  if (indices.dtype() == bool_) {
    throw std::invalid_argument("boolean indices are not yet supported");
  }

  // If only one input array is mentioned, we set axis=0 in take
  // for parity with np
  return take(src, indices, 0);
}

array mlx_get_item_int(const array& src, const py::int_& idx) {
  // Check input and raise error if 0 dim for parity with np
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  // If only one input idx is mentioned, we set axis=0 in take
  // for parity with np
  return take(src, get_int_index(idx, src.shape(0)), 0);
}

array mlx_gather_nd(
    array src,
    const std::vector<py::object>& indices,
    bool gather_first,
    int& max_dims) {
  max_dims = 0;
  std::vector<array> gather_indices;
  std::vector<bool> is_slice(indices.size(), false);
  int num_slices = 0;
  // gather all the arrays
  for (int i = 0; i < indices.size(); i++) {
    auto& idx = indices[i];

    if (py::isinstance<py::slice>(idx)) {
      int start, end, stride;
      get_slice_params(start, end, stride, idx, src.shape(i));

      // Handle negative indices
      start = (start < 0) ? start + src.shape(i) : start;
      end = (end < 0) ? end + src.shape(i) : end;

      gather_indices.push_back(arange(start, end, stride, uint32));
      num_slices++;
      is_slice[i] = true;
    } else if (py::isinstance<py::int_>(idx)) {
      gather_indices.push_back(get_int_index(idx, src.shape(i)));
    } else if (py::isinstance<array>(idx)) {
      auto arr = py::cast<array>(idx);
      max_dims = std::max(static_cast<int>(arr.ndim()), max_dims);
      gather_indices.push_back(arr);
    }
  }

  // reshape them so that the int/array indices are first
  if (gather_first) {
    int slice_index = 0;
    for (int i = 0; i < gather_indices.size(); i++) {
      if (is_slice[i]) {
        std::vector<int> index_shape(max_dims + num_slices, 1);
        index_shape[max_dims + slice_index] = gather_indices[i].shape(0);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
        slice_index++;
      } else {
        std::vector<int> index_shape = gather_indices[i].shape();
        index_shape.insert(index_shape.end(), num_slices, 1);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
      }
    }
  } else {
    // reshape them so that the int/array indices are last
    for (int i = 0; i < gather_indices.size(); i++) {
      if (i < num_slices) {
        std::vector<int> index_shape(max_dims + num_slices, 1);
        index_shape[i] = gather_indices[i].shape(0);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
      }
    }
  }

  // Do the gather
  std::vector<int> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int> slice_sizes = src.shape();
  std::fill(slice_sizes.begin(), slice_sizes.begin() + indices.size(), 1);
  src = gather(src, gather_indices, axes, slice_sizes);

  // Squeeze the dims
  std::vector<int> out_shape;
  out_shape.insert(
      out_shape.end(),
      src.shape().begin(),
      src.shape().begin() + max_dims + num_slices);
  out_shape.insert(
      out_shape.end(),
      src.shape().begin() + max_dims + num_slices + indices.size(),
      src.shape().end());
  src = reshape(src, out_shape);

  return src;
}

array mlx_get_item_nd(array src, const py::tuple& entries) {
  // No indices make this a noop
  if (entries.size() == 0) {
    return src;
  }

  // The plan is as follows:
  // 1. Replace the ellipsis with a series of slice(None)
  // 2. Loop over the indices and calculate the gather indices
  // 3. Calculate the remaining slices and reshapes

  // Ellipsis handling
  std::vector<py::object> indices;
  {
    int non_none_indices_before = 0;
    int non_none_indices_after = 0;
    std::vector<py::object> r_indices;
    int i = 0;
    for (; i < entries.size(); i++) {
      auto idx = entries[i];
      if (!is_valid_index_type(idx)) {
        throw std::invalid_argument(
            "Cannot index mlx array using the given type yet");
      }
      if (!py::ellipsis().is(idx)) {
        indices.push_back(idx);
        non_none_indices_before += !idx.is_none();
      } else {
        break;
      }
    }
    for (int j = entries.size() - 1; j > i; j--) {
      auto idx = entries[j];
      if (!is_valid_index_type(idx)) {
        throw std::invalid_argument(
            "Cannot index mlx array using the given type yet");
      }
      if (py::ellipsis().is(idx)) {
        throw std::invalid_argument(
            "An index can only have a single ellipsis (...)");
      }
      r_indices.push_back(idx);
      non_none_indices_after += !idx.is_none();
    }
    for (int axis = non_none_indices_before;
         axis < src.ndim() - non_none_indices_after;
         axis++) {
      indices.push_back(py::slice(0, src.shape(axis), 1));
    }
    indices.insert(indices.end(), r_indices.rbegin(), r_indices.rend());
  }

  // Check for the number of indices passed
  {
    int cnt = src.ndim();
    for (auto& idx : indices) {
      if (!idx.is_none()) {
        cnt--;
      }
    }
    if (cnt < 0) {
      std::ostringstream msg;
      msg << "Too many indices for array with " << src.ndim() << "dimensions.";
      throw std::invalid_argument(msg.str());
    }
  }

  // Gather handling
  //
  // Check whether we have arrays or integer indices and delegate to gather_nd
  // after removing the slices at the end and all Nones.
  std::vector<py::object> remaining_indices;
  bool have_array = false;
  {
    // First check whether the results of gather are going to be 1st or
    // normally in between.
    bool have_non_array = false;
    bool gather_first = false;
    for (auto& idx : indices) {
      if (py::isinstance<array>(idx) || py::isinstance<py::int_>(idx)) {
        if (have_array && have_non_array) {
          gather_first = true;
          break;
        }
        have_array = true;
      } else {
        have_non_array |= have_array;
      }
    }

    if (have_array) {
      int last_array;
      // Then find the last array
      for (last_array = indices.size() - 1; last_array >= 0; last_array--) {
        auto& idx = indices[last_array];
        if (py::isinstance<array>(idx) || py::isinstance<py::int_>(idx)) {
          break;
        }
      }

      std::vector<py::object> gather_indices;
      for (int i = 0; i <= last_array; i++) {
        auto& idx = indices[i];
        if (!idx.is_none()) {
          gather_indices.push_back(idx);
        }
      }
      int max_dims;
      src = mlx_gather_nd(src, gather_indices, gather_first, max_dims);

      // Reassemble the indices for the slicing or reshaping if there are any
      if (gather_first) {
        for (int i = 0; i < max_dims; i++) {
          remaining_indices.push_back(
              py::slice(py::none(), py::none(), py::none()));
        }
        for (int i = 0; i < last_array; i++) {
          auto& idx = indices[i];
          if (idx.is_none()) {
            remaining_indices.push_back(indices[i]);
          } else if (py::isinstance<py::slice>(idx)) {
            remaining_indices.push_back(
                py::slice(py::none(), py::none(), py::none()));
          }
        }
        for (int i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      } else {
        for (int i = 0; i < indices.size(); i++) {
          auto& idx = indices[i];
          if (py::isinstance<array>(idx) || py::isinstance<py::int_>(idx)) {
            break;
          } else if (idx.is_none()) {
            remaining_indices.push_back(idx);
          } else {
            remaining_indices.push_back(
                py::slice(py::none(), py::none(), py::none()));
          }
        }
        for (int i = 0; i < max_dims; i++) {
          remaining_indices.push_back(
              py::slice(py::none(), py::none(), py::none()));
        }
        for (int i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      }
    }
  }
  if (have_array && remaining_indices.empty()) {
    return src;
  }
  if (remaining_indices.empty()) {
    remaining_indices = indices;
  }

  // Slice handling
  {
    std::vector<int> starts(src.ndim(), 0);
    std::vector<int> ends = src.shape();
    std::vector<int> strides(src.ndim(), 1);
    int axis = 0;
    for (auto& idx : remaining_indices) {
      if (!idx.is_none()) {
        get_slice_params(
            starts[axis], ends[axis], strides[axis], idx, ends[axis]);
        axis++;
      }
    }
    src = slice(src, starts, ends, strides);
  }

  // Unsqueeze handling
  if (remaining_indices.size() > src.ndim()) {
    std::vector<int> out_shape;
    int axis = 0;
    for (auto& idx : remaining_indices) {
      if (idx.is_none()) {
        out_shape.push_back(1);
      } else {
        out_shape.push_back(src.shape(axis++));
      }
    }
    src = reshape(src, out_shape);
  }

  return src;
}

array mlx_get_item(const array& src, const py::object& obj) {
  if (py::isinstance<py::slice>(obj)) {
    return mlx_get_item_slice(src, obj);
  } else if (py::isinstance<array>(obj)) {
    return mlx_get_item_array(src, py::cast<array>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    return mlx_get_item_int(src, obj);
  } else if (py::isinstance<py::tuple>(obj)) {
    return mlx_get_item_nd(src, obj);
  } else if (obj.is_none()) {
    std::vector<int> s(1, 1);
    s.insert(s.end(), src.shape().begin(), src.shape().end());
    return reshape(src, s);
  }
  throw std::invalid_argument("Cannot index mlx array using the given type.");
}

array mlx_set_item_int(
    const array& src,
    const py::int_& idx,
    const array& update) {
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  // Remove any leading singleton dimensions from the update
  // and then broadcast update to shape of src[0, ...]
  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++)
    ;
  auto up_shape =
      std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto shape = src.shape();
  shape[0] = 1;
  return scatter(
      src,
      get_int_index(idx, src.shape(0)),
      broadcast_to(reshape(update, up_shape), shape),
      0);
}

array mlx_set_item_array(
    const array& src,
    const array& indices,
    const array& update) {
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  // Remove any leading singleton dimensions from the update
  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++)
    ;
  auto up_shape =
      std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto up = reshape(update, up_shape);

  // The update shape must broadcast with indices.shape + [1] + src.shape[1:]
  up_shape = indices.shape();
  up_shape.insert(up_shape.end(), src.shape().begin() + 1, src.shape().end());
  up = broadcast_to(up, up_shape);
  up_shape.insert(up_shape.begin() + indices.ndim(), 1);
  up = reshape(up, up_shape);

  return scatter(src, indices, up, 0);
}

array mlx_set_item_slice(
    const array& src,
    const py::slice& in_slice,
    const array& update) {
  // Check input and raise error if 0 dim for parity with np
  if (src.ndim() == 0) {
    throw std::invalid_argument(
        "too many indices for array: array is 0-dimensional");
  }

  // If none slice is requested broadcast the update
  // to the src size and return it.
  if (is_none_slice(in_slice)) {
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++)
      ;
    auto up_shape =
        std::vector<int>(update.shape().begin() + s, update.shape().end());
    return broadcast_to(reshape(update, up_shape), src.shape());
  }

  int start = 0;
  int end = src.shape(0);
  int stride = 1;

  // Check and update slice params
  get_slice_params(start, end, stride, in_slice, end);

  return mlx_set_item_array(src, arange(start, end, stride, uint32), update);
}

array mlx_set_item_nd(
    const array& src,
    const py::tuple& entries,
    const array& update) {
  std::vector<py::object> indices;
  int non_none_indices = 0;

  // Expand ellipses into a series of ':' slices
  {
    int non_none_indices_before = 0;
    int non_none_indices_after = 0;
    bool has_ellipsis = false;
    int indices_before = 0;
    for (int i = 0; i < entries.size(); ++i) {
      auto idx = entries[i];
      if (!is_valid_index_type(idx)) {
        throw std::invalid_argument(
            "Cannot index mlx array using the given type yet");
      } else if (!py::ellipsis().is(idx)) {
        if (!has_ellipsis) {
          indices_before++;
          non_none_indices_before += !idx.is_none();
        } else {
          non_none_indices_after += !idx.is_none();
        }
        indices.push_back(idx);
      } else if (has_ellipsis) {
        throw std::invalid_argument(
            "An index can only have a single ellipsis (...)");
      } else {
        has_ellipsis = true;
      }
    }
    if (has_ellipsis) {
      for (int axis = non_none_indices_before;
           axis < src.ndim() - non_none_indices_after;
           axis++) {
        indices.insert(
            indices.begin() + indices_before, py::slice(0, src.shape(axis), 1));
      }
      non_none_indices = src.ndim();
    } else {
      non_none_indices = non_none_indices_before + non_none_indices_after;
    }
  }

  if (non_none_indices > src.ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << src.ndim() << "dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Remove leading singletons dimensions from the update
  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++) {
  };
  auto up_shape =
      std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto up = reshape(update, up_shape);

  // If no non-None indices return the broadcasted update
  if (non_none_indices == 0) {
    return broadcast_to(up, src.shape());
  }

  unsigned long max_dim = 0;
  bool arrays_first = false;
  int num_slices = 0;
  int num_arrays = 0;
  {
    bool have_array = false;
    bool have_non_array = false;
    for (auto& idx : indices) {
      if (py::isinstance<py::slice>(idx) || idx.is_none()) {
        have_non_array = have_array;
        num_slices++;
      } else if (py::isinstance<array>(idx)) {
        have_array = true;
        if (have_array && have_non_array) {
          arrays_first = true;
        }
        max_dim = std::max(py::cast<array>(idx).ndim(), max_dim);
        num_arrays++;
      }
    }
  }

  std::vector<array> arr_indices;
  int slice_num = 0;
  int array_num = 0;
  int ax = 0;
  for (int i = 0; i < indices.size(); ++i) {
    auto& pyidx = indices[i];
    if (py::isinstance<py::slice>(pyidx)) {
      int start, end, stride;
      auto axis_size = src.shape(ax++);
      get_slice_params(start, end, stride, pyidx, axis_size);

      // Handle negative indices
      start = (start < 0) ? start + axis_size : start;
      end = (end < 0) ? end + axis_size : end;

      auto idx = arange(start, end, stride, uint32);
      std::vector<int> idx_shape(max_dim + num_slices, 1);
      auto loc = slice_num + (arrays_first ? max_dim : 0);
      slice_num++;
      idx_shape[loc] = idx.size();
      arr_indices.push_back(reshape(idx, idx_shape));
    } else if (py::isinstance<py::int_>(pyidx)) {
      arr_indices.push_back(get_int_index(pyidx, src.shape(ax++)));
    } else if (pyidx.is_none()) {
      slice_num++;
    } else if (py::isinstance<array>(pyidx)) {
      ax++;
      auto idx = py::cast<array>(pyidx);
      std::vector<int> idx_shape;
      if (!arrays_first) {
        idx_shape.insert(idx_shape.end(), slice_num, 1);
      }
      idx_shape.insert(idx_shape.end(), max_dim - idx.ndim(), 1);
      idx_shape.insert(idx_shape.end(), idx.shape().begin(), idx.shape().end());
      idx_shape.insert(
          idx_shape.end(), num_slices - (arrays_first ? 0 : slice_num), 1);
      arr_indices.push_back(reshape(idx, idx_shape));
      if (!arrays_first && ++array_num == num_arrays) {
        slice_num += max_dim;
      }
    } else {
      throw std::invalid_argument(
          "Cannot index mlx array using the given type yet");
    }
  }

  arr_indices = broadcast_arrays(arr_indices);
  up_shape = arr_indices[0].shape();
  up_shape.insert(
      up_shape.end(),
      src.shape().begin() + non_none_indices,
      src.shape().end());
  up = broadcast_to(up, up_shape);
  up_shape.insert(
      up_shape.begin() + arr_indices[0].ndim(), non_none_indices, 1);
  up = reshape(up, up_shape);

  std::vector<int> axes(arr_indices.size(), 0);
  std::iota(axes.begin(), axes.end(), 0);
  return scatter(src, arr_indices, up, axes);
}

void mlx_set_item(array& src, const py::object& obj, const ScalarOrArray& v) {
  auto vals = to_array(v, src.dtype());
  auto impl = [&src, &obj, &vals]() {
    if (py::isinstance<py::slice>(obj)) {
      return mlx_set_item_slice(src, obj, vals);
    } else if (py::isinstance<array>(obj)) {
      return mlx_set_item_array(src, py::cast<array>(obj), vals);
    } else if (py::isinstance<py::int_>(obj)) {
      return mlx_set_item_int(src, obj, vals);
    } else if (py::isinstance<py::tuple>(obj)) {
      return mlx_set_item_nd(src, obj, vals);
    } else if (obj.is_none()) {
      return broadcast_to(vals, src.shape());
    }
    throw std::invalid_argument("Cannot index mlx array using the given type.");
  };
  auto out = impl();
  src.overwrite_descriptor(out);
}
