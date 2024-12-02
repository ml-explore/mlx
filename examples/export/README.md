## Setup

Install mlx:

```bash
pip install mlx>=0.22
```

Build the C++ examples:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

Run the Python script to export the function.

```bash
python eval_mlp.py
```

Then run the C++ program to import and run the function.

```
./build/eval_mlp
```

The two programs should output the same result.
