# Welcome to the MLX community!

We welcome contributors of all backgrounds and experience levels. A community with diverse perspectives builds software with lasting impact.

[🤝 How you can help](#how-you-can-help) · [⚙️ Setting up your environment](#setting-up-your-environment) · [📝 Submitting issues and pull requests](#submitting-issues-and-pull-requests) · [🔀 Pull requests](#pull-requests) · [❤️ Code of conduct](#code-of-conduct)

## How you can help

- Report bugs with clear, reproducible steps
- Improve documentation to make the project more accessible
- Triage issues by providing feedback, testing, and validation
- Propose and implement enhancements of all sizes
- Share your experience by speaking about MLX at conferences or meetups

## Setting up your environment

See [Build and Install](https://ml-explore.github.io/mlx/build/html/install.html) for setup instructions.

## Submitting issues and pull requests

We maintain high standards to ensure quality across the project.

Your contributions are evaluated on:

- Technical quality and correctness
- Adherence to existing conventions and architectural patterns
- Demonstrated understanding of the implementation and its implications
- Clarity of communication
- Maintainability

You may use productivity tools, including AI-assisted coding, to help you work more efficiently.
However, you remain fully accountable for all submitted code, issues, pull requests, and comments.
AI-generated content must meet the same standards as human-written contributions. Review, validate, and, as needed,
refine or rearrange AI-assisted work so the final contribution reflects your human creativity, understanding, and
control. Use your voice and expression when producing written materials. Misuse of AI tools in your contributions
and conversations may be considered a violation of our [Code of Conduct](#code-of-conduct).

### Issues

Before submitting an issue:

- Search [existing issues](../../issues) to avoid duplicates.
- Write a clear, concise title and a detailed description. Ensure all information is accurate and fully understood by you.
- Include precise steps to reproduce the issue, expected behavior, and actual behavior. Verify these steps yourself.
- Specify your environment, MLX version, and any relevant configurations.
- Explain the severity and impact of the problem.
- For security issues, please see the [MLX security advisories](https://github.com/ml-explore/mlx?tab=security-ov-file). Do **not** open a public GitHub issue.

#### Proposing features

We welcome feature requests. Describe the proposed changes needed and why they matter. Please do not start with a pull request. Use the issue template that best matches your needs.

### Pull requests

⚠️ **Important:** We recommend an issue for larger changes to the codebase.

Pull requests represent proposed solutions or enhancements. When submitting a pull request, verify it meets these expectations:

- They should match an open issue but may not always.
- Address the stated problem or feature request completely and effectively. You must fully understand and validate your proposed solution.
- Solutions must be thorough, handle edge cases, and integrate cleanly.
- Include comprehensive tests that validate your changes and prevent regressions.
- If performance may be impacted, run benchmarks for both the main branch and the pull request.
- When providing benchmarking results, include scripts and reproduction steps.

#### Pull request style guide and format

Follow our coding style and formatting to maintain a consistent, readable codebase.

- Write comprehensive and clear documentation. Concisely explain the 'why' behind complex decisions.
- Write clear, concise, and descriptive commit messages.
- Format the code with `pre-commit run --all` before submitting a pull request. You can also install git hooks to run it automatically:

  ```shell
  pip install pre-commit
  pre-commit install
  ```

#### Testing

All contributions require thorough testing.

**Local Testing:** Ensure all existing tests pass before submitting:

```bash
python3 python/tests/run.py
```

**Automated Tests:** New features and bug fixes require corresponding automated tests that validate the intended behavior and prevent regressions.

#### Running CI

CI runs automatically on every pull request and needs to be approved by a maintainer before it will run.

## Code of conduct

We are committed to fostering a community where different experiences and perspectives come together to create and collaborate. Good collaboration depends on honest feedback and respect for the time and effort every contributor brings to this project. [Please review our Code of Conduct](https://github.com/ml-explore/.github/blob/main/CODE_OF_CONDUCT.md); all community members are expected to adhere to it.
