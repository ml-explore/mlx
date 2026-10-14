
The example builds `mlx_sample_extensions` as a top-level Python extension
module. No `._ext` suffix or Python wrapper package is required.

## Build

```
pip install -e .
```

For faster builds during development, you can also pre-install the requirements:

```
pip install -r requirements.txt
```

And then run:

```
python setup.py build_ext -j8 --inplace
```

## Test

```
python test.py
```
