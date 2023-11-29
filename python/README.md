### Packaging for PyPI

Install `build` and `twine`:

```
pip install --user --upgrade build
pip install --user --upgrade twine
```

Generate the source distribution and wheel:

```
python -m build
```

*Warning* use a test server first

#### Test Upload

Upload to test server:

```
python -m twine upload --repository testpypi dist/*
```

Install from test server and check that it works:

```
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps mlx
```

#### Upload

```
python -m twine upload dist/*
```

