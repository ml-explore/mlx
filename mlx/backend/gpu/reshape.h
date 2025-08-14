// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

// Copy data from |in| and transpose to |out|'s shape.
void reshape_gpu(const array& in, array& out, Stream s);

// Like the normal ops but safe to call in eval_gpu.
array flatten_in_eval(const array& x, int start_axis, int end_axis, Stream s);
array reshape_in_eval(const array& x, Shape shape, Stream s);
array swapaxes_in_eval(const array& x, int axis1, int axis2);

} // namespace mlx::core
