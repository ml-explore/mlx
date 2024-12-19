## Setup

Install MLX:

```bash
pip install mlx>=0.22
```

Build the C++ examples:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

### Eval MLP

Run the Python script to export the eval function:

```bash
python eval_mlp.py
```

Then run the C++ program to import and run the function:

```
./build/eval_mlp
```

The Python and C++ programs should output the same result.

### Train MLP

Run the Python script to export the model initialization and training
functions:

```bash
python train_mlp.py
```

Then run the C++ program to import and run the functions:

```
./build/train_mlp
```

The Python and C++ programs should output the same results.
