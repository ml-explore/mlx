# Contributing to MLX

We want to make contributing to this project as easy and transparent as
possible.

## AI Usage Policy

AI-generated code is allowed. What is not allowed is submitting code you do not
understand. You are 100% responsible for every line, however it was produced,
and must explicitly disclose the manner in which AI was employed.

It is strictly prohibited to use AI to write your posts for you (bug reports,
feature requests, pull request descriptions, Github discussions, responding to
humans, ...).

## Pull Requests

- Make sure new code is covered by tests. Add new tests if not, and confirm
  the new tests fail in the main branch.
- If performance may be impacted, run benchmarks for both the main branch and
  the pull request.
- When providing benchmarking results, include scripts and reproduction steps.
- Format the code with `uvx pre-commit run --all` before submitting a pull
  request. You can also install git hooks to run it automatically:

  ```shell
  pip install pre-commit
  pre-commit install
  ```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
