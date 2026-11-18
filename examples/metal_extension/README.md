# Activation Metal extension

A minimal MLX Metal extension exposing `silu_and_mul`. It uses
`mlx.extension.MetalExtension` to build the top-level module
`mlx_sample_metal_extension`, without a `._ext` suffix or Python wrapper package.

The kernel is adapted from Hugging Face
[`activation_metal`](https://github.com/huggingface/kernels-community/tree/47a3168d0808921eef2f7daca794a4fccae13078/activation/activation_metal)
under Apache-2.0.

## Build

```bash
pip install -r requirements.txt
python setup.py build_ext -j8 --inplace
```

The build uses CMake's default generator and generates
`mlx_sample_metal_extension.pyi` by default.
To enable Ninja, install `ninja` and use
`BuildExtension.with_options(use_ninja=True)`. To disable stub generation, use
`BuildExtension.with_options(generate_stubs=False)`.

## Test

```bash
python test.py
```
