#pragma once

#include "mlx/array.h"

namespace mlx::core {

void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::string op,
    const Stream& s);

void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::string op,
    const Stream& s);

void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s);

void pad_gpu(
    const array& in,
    const array& val,
    array& out,
    std::vector<int> axes,
    std::vector<int> low_pad_size,
    const Stream& s);

} // namespace mlx::core