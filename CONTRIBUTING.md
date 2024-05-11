# Contributing to MLX

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. If a change is likely to impact efficiency, run some of the benchmarks before
   and after the change. Examples of benchmarks can be found in `benchmarks/python/`.
4. If you've changed APIs, update the documentation.
5. Every PR should have passing tests and at least one review. 
6. For code formatting install `pre-commit`, `black` and `clang-format` using something like `pip install pre-commit black clang-format` and run `pre-commit install`.
   This should install hooks for running `black` and `clang-format` to ensure
   consistent style for C++ and python code.
   For convenience, you can set up the repository-specific Python virtualenv inside the `venv` directory at the root of the repository - it is ignored by git.
 
   You can also run the formatters manually as follows:
 
     ```
     clang-format -i file.cpp
     ```
 
     ```
     black file.py
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
