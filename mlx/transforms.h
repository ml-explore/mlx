// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"

namespace mlx::core {

void async_eval(std::vector<array> outputs);

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
void async_eval(Arrays&&... outputs) {
  async_eval(std::vector<array>{std::forward<Arrays>(outputs)...});
}

void eval(std::vector<array> outputs);

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
void eval(Arrays&&... outputs) {
  eval(std::vector<array>{std::forward<Arrays>(outputs)...});
}

/**
 *  Computes the output and vector-Jacobian product (VJP) of a function.
 *
 *  Computes the vector-Jacobian product of the vector of cotangents with the
 *  Jacobian of the function evaluated at the primals. Returns a pair of
 *  vectors of output arrays and VJP arrays.
 **/
std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotangents);

/**
 *  Computes the output and vector-Jacobian product (VJP) of a unary function.
 */
std::pair<array, array> vjp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& cotangent);

/**
 *  Computes the output and Jacobian-vector product (JVP) of a function.
 *
 *  Computes the Jacobian-vector product of the Jacobian of the function
 *  evaluated at the primals with the vector of tangents. Returns a pair of
 *  vectors of output arrays and JVP arrays.
 **/
std::pair<std::vector<array>, std::vector<array>> jvp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& tangents);

/**
 *  Computes the output and Jacobian-vector product (JVP) of a unary function.
 */
std::pair<array, array> jvp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& tangent);

// Return type of general value_and_grad: a function which takes an input
// vector of arrays and returns a pair of vectors of arrays one for the
// values and one for the gradients wrt the first value.
using ValueAndGradFn =
    std::function<std::pair<std::vector<array>, std::vector<array>>(
        const std::vector<array>&)>;
using SimpleValueAndGradFn = std::function<std::pair<array, std::vector<array>>(
    const std::vector<array>&)>;

/**
 *  Returns a function which computes the value and gradient of the input
 *  function with respect to a vector of input arrays.
 **/
ValueAndGradFn value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums);

/**
 *  Returns a function which computes the value and gradient of the input
 *  function with respect to a single input array.
 **/
ValueAndGradFn inline value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return value_and_grad(fun, std::vector<int>{argnum});
}

/**
 *  Returns a function which computes the value and gradient of the unary
 *  input function.
 **/
std::function<std::pair<array, array>(const array&)> inline value_and_grad(
    const std::function<array(const array&)>& fun) {
  return [fun](auto inputs) { return vjp(fun, inputs, array(1.0f)); };
}

SimpleValueAndGradFn inline value_and_grad(
    const std::function<array(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  return [fun, argnums](auto inputs) {
    auto result = value_and_grad(
        [fun](auto inputs) { return std::vector<array>{fun(inputs)}; },
        argnums)(inputs);

    return std::make_pair(result.first[0], result.second);
  };
}

SimpleValueAndGradFn inline value_and_grad(
    const std::function<array(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return value_and_grad(fun, std::vector<int>{argnum});
}

/**
 *  Returns a function which computes the gradient of the input function with
 *  respect to a vector of input arrays.
 *
 *  The function being differentiated takes a vector of arrays and returns an
 *  array. The vector of `argnums` specifies which the arguments to compute
 *  the gradient with respect to. At least one argument must be specified.
 **/
std::function<std::vector<array>(const std::vector<array>&)> inline grad(
    const std::function<array(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  auto fn = value_and_grad(fun, argnums);
  return [fn](const std::vector<array>& inputs) { return fn(inputs).second; };
}

/**
 *  Returns a function which computes the gradient of the input function with
 *  respect to a single input array.
 *
 *  The function being differentiated takes a vector of arrays and returns an
 *  array. The optional `argnum` index specifies which the argument to compute
 *  the gradient with respect to and defaults to 0.
 **/
std::function<std::vector<array>(const std::vector<array>&)> inline grad(
    const std::function<array(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return grad(fun, std::vector<int>{argnum});
}

/**
 *  Returns a function which computes the gradient of the unary input function.
 **/
std::function<array(const array&)> inline grad(
    const std::function<array(const array&)>& fun) {
  auto fn = value_and_grad(fun);
  return [fn](const array& input) { return fn(input).second; };
}

/**
 * Automatically vectorize a unary function over the requested axes.
 */
std::function<array(const array&)> vmap(
    const std::function<array(const array&)>& fun,
    int in_axis = 0,
    int out_axis = 0);

/**
 * Automatically vectorize a binary function over the requested axes.
 */
std::function<array(const array&, const array&)> vmap(
    const std::function<array(const array&, const array&)>& fun,
    int in_axis_a = 0,
    int in_axis_b = 0,
    int out_axis = 0);

/**
 * Automatically vectorize a function over the requested axes.
 *
 * The input function to `vmap` takes as an argument a vector of arrays and
 * returns a vector of arrays. Optionally specify the axes to vectorize over
 * with `in_axes` and `out_axes`, otherwise a default of 0 is used.
 * Returns a vectorized function with the same signature as the input
 * function.
 */
std::function<std::vector<array>(const std::vector<array>&)> vmap(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& in_axes = {},
    const std::vector<int>& out_axes = {});

/**
 * Redefine the transformations of `fun` according to the provided functions.
 *
 * Namely when calling the vjp of `fun` then `fun_vjp` will be called,
 * `fun_jvp` for the jvp and `fun_vmap` for vmap.
 *
 * If any transformation is not provided, then a default one is created by
 * calling `vjp`, `jvp` and `vmap` on the function directly.
 */
std::function<std::vector<array>(const std::vector<array>&)> custom_function(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::optional<std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<array>&)>> fun_vjp = std::nullopt,
    std::optional<std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<int>&)>> fun_jvp = std::nullopt,
    std::optional<std::function<std::pair<std::vector<array>, std::vector<int>>(
        const std::vector<array>&,
        const std::vector<int>&)>> fun_vmap = std::nullopt);

/**
 * Return a function that behaves exactly like `fun` but if the vjp of the
 * results is computed `fun_vjp` will be used instead of `vjp(fun, ...)` .
 */
std::function<std::vector<array>(const std::vector<array>&)> custom_vjp(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<array>&)> fun_vjp);

/**
 * Checkpoint the gradient of a function. Namely, discard all intermediate
 * state and recalculate it when we need to compute the gradient.
 */
std::function<std::vector<array>(const std::vector<array>&)> checkpoint(
    std::function<std::vector<array>(const std::vector<array>&)> fun);

} // namespace mlx::core
