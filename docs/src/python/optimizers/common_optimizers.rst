.. _common_optimizers:

Common Optimizers
=================

.. currentmodule:: mlx.optimizers

.. autosummary::
   :toctree: _autosummary
   :template: optimizers-template.rst

   SGD
   RMSprop
   Adagrad
   Adafactor
   AdaDelta
   Adam
   AdamW
   Adamax
   Lion
   Muon
   MultiOptimizer


Muon Optimizer
--------------

The :class:`Muon` optimizer implements the MomentUm Orthogonalized by Newton-schulz (Muon) algorithm, 
which combines momentum-based SGD with Newton-Schulz orthogonalization for hidden weight matrices.

**Key Features:**

- Applies orthogonalization only to 2D+ parameters (weight matrices)
- Falls back to standard momentum SGD for 1D parameters (biases, layer norms)
- Multiple Newton-Schulz methods: ``"auto"``, ``"cubic"``, ``"quintic"``
- Intelligent fallback from quintic to cubic for robust convergence

**Usage Guidelines:**

.. code-block:: python

   import mlx.core as mx
   import mlx.nn as nn
   from mlx.optimizers import Muon

   # Basic usage with auto method selection
   optimizer = Muon(learning_rate=0.02)

   # Specify Newton-Schulz method explicitly
   optimizer = Muon(
       learning_rate=0.02,
       momentum=0.95,
       method="auto",    # or "cubic" for stability, "quintic" for speed
       ns_steps=5,       # Newton-Schulz iteration steps
       tol=0.05         # Orthogonality tolerance for auto fallback
   )

**Learning Rate Guidelines:**

- **Vision models**: Use 5-20× higher learning rates than AdamW
- **Language models**: Start with 5-10× AdamW learning rates  
- **Typical range**: 0.01-0.1 (vs 0.001-0.01 for AdamW)

**Method Selection:**

- ``method="auto"`` (default): Try quintic first, fallback to cubic if convergence poor
- ``method="cubic"``: Always stable, guaranteed convergence, good for CPU
- ``method="quintic"``: Fastest per iteration, may need fallback on challenging shapes

**Important Notes:**

- Orthogonalization is **only applied to 2D+ parameters** (weight matrices)
- 1D parameters (biases, gains) use standard momentum SGD
- Consider ``method="cubic"`` on CPU to avoid bfloat16 overhead
