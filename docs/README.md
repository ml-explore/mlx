## Build the Documentations

### Setup (do once)

Install Python packages:

```
uv pip install .[docs]
```

### Build

Build the docs from `docs/src/`

```
uv run sphinx-build -M html docs/src/ docs/build
```

View the docs:

```
open docs/build/html/index.html
```

### Publish to GitHub Pages

By default whenever a new release is published the GitHub pages would be
automatically rebuilt, but you can also manually rebuild the pages by triggering
the [Documentation workflow](https://github.com/ml-explore/mlx/actions/workflows/documentation.yml).

## Development

To enable live refresh of docs while writing:

Install sphinx autobuild
```
uv pip install sphinx-autobuild
```

Run auto build on `docs/src` folder
```
uv run sphinx-autobuild docs/src docs/build/html
```
