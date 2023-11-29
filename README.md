# MLX

MLX is an array framework for machine learning specifically targeting Apple
Silicon. MLX is designed with inspiration from Jax, PyTorch, ArrayFire.

[Documentation](https://at.apple.com/mlx)

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


## Develop

- Fork and submit pull requests to the repo. 

- Every PR should have passing tests and at least one review.

- If a change is likely to impact efficiency, run some of the benchmarks before
  and after the change. Examples of benchmarks can be found in `benchmarks/cpp/`.

- Install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
  This should install hooks for running `black` and `clang-format` to ensure
  consistent style for C++ and python code.

  You can also run the formatters manually as follows:

    ```
    clang-format -i file.cpp
    ```

    ```
    black file.py
    ```

    or run `pre-commit run --all-files` to check all files in the repo.
