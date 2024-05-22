
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
