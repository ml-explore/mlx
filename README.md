# MLX

MLX is an array framework for machine learning specifically targeting Apple
Silicon. MLX is designed with inspiration from Jax, PyTorch, ArrayFire.

[Documentation](https://ml-explore.github.io/mlx/build/html/index.html)

## Build

```
mkdir -p build && cd build
cmake .. && make -j
```

Run the C++ tests with `make test` (or `./tests/tests` for more detailed output).

### Python bidings

To install run:

`
env CMAKE_BUILD_PARALLEL_LEVEL="" pip install .
`

For developing use an editable install:

```
env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e .
```

To make sure the install is working run the tests with:

```
python -m unittest discover python/tests
```


## Contributing 

Check out the [contribution guidelines](CONTRIBUTING.md) for more information
on contributing to MLX.
