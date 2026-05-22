# Distributed MLX examples

Small, runnable examples of `mlx.distributed`. Each example is self-contained
and uses synthetic data so there is nothing to download.

## Data parallel training

`data_parallel/main.py` trains a small MLP on a synthetic classification task.
Each rank processes a different shard of the data and gradients are averaged
across ranks after every step using `mlx.nn.average_gradients`, which batches
several `mx.distributed.all_sum` calls together.

Run locally with two processes:

```
mlx.launch -n 2 examples/distributed/data_parallel/main.py
```

`mx.distributed` operations are no-ops when the world size is 1, so the same
script also runs unchanged with plain `python`:

```
python examples/distributed/data_parallel/main.py
```

See `docs/src/usage/distributed.rst` for backend setup (MPI, ring, JACCL,
NCCL) and `docs/src/usage/launching_distributed.rst` for details on
`mlx.launch`.
