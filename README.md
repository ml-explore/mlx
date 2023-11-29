# MLX

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html)

MLX is an array framework for machine learning on Apple silicon.

Some key features of MLX include:

 - **Familiar APIs**: MLX has a Python API which closely follows NumPy.
   MLX also has a fully featured C++ API which closely mirrors the Python API. 
   MLX also has higher level `mlx.nn` and `mlx.optimizers` with APIs that closely
   follow PyTorch to simplify building more complex models 

 - **Composable function transformations**: MLX has composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.

 - **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.

 - **Multi-device**: Operations can run on any of the supported devices (CPU,
   GPU, ...) 

 - **Unified Memory**: A noteable difference from MLX and other frameworks is
   is the *unified memory model*. Arrays in MLX live in shared memory.
   Operations on MLX arrays can be performed on any of the supported
   device types without performing data copies.

The design of MLX is inspired by frameworks like `PyTorch
<https://pytorch.org/>`_, `Jax <https://github.com/google/jax>`_, and
`ArrayFire <https://arrayfire.org/>`_.

## Quickstart

See the [quick start
guide](https://pages.github.pie.apple.com/ml-explore/framework002/build/html/quick_start.html)
in the documentation.

## Installation

MLX is available on [PyPi](https://pypi.org/project/mlx/). To install the Python API run:

```
pip install mlx
```

Checkout the
[documentation](https://ml-explore.github.io/mlx/build/html/install.html#)
for more information on building the C++ and Python APIs from source.

## Contributing 

Check out the [contribution guidelines](CONTRIBUTING.md) for more information
on contributing to MLX.
