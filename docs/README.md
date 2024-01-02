## Build the Docs

### Setup (do once)

Install [sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html)
for example with `conda`:

```
conda install sphinx
pip install sphinx-book-theme
```

### Build

Build the docs from `mlx/docs/`

```
make html
```

View the docs by running a server in `mlx/docs/build/html/`:

```
python -m http.server <port>
```

and point your browser to `http://localhost:<port>`.

### Push to GitHub Pages

Check-out the `gh-pages` branch (`git switch gh-pages`) and build
the docs. Then force add the `build/html` directory:

`git add -f build/html`

Commit and push the changes to the `gh-pages` branch.
