# Activation Metal extension

A minimal MLX Metal extension exposing `silu_and_mul`, `fatrelu_and_mul`, and
`gelu`. It uses `mlx.extension.MetalExtension`.

The kernels are adapted from Hugging Face
[`activation_metal`](https://github.com/huggingface/kernels-community/tree/47a3168d0808921eef2f7daca794a4fccae13078/activation/activation_metal)
under Apache-2.0.

## Build

```bash
pip install -r requirements.txt
python setup.py build_ext -j8 --inplace
```

The build uses Ninja and generates `_ext.pyi` by default. It falls back to
CMake's default generator when Ninja is unavailable. Either option can be
disabled with `BuildExtension.with_options(use_ninja=False,
generate_stubs=False)`.

## Test

```bash
python test.py
```
