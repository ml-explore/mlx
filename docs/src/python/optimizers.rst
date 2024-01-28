.. _optimizers:

Optimizers
==========

The optimizers in MLX can be used both with :mod:`mlx.nn` but also with pure
:mod:`mlx.core` functions. A typical example involves calling
:meth:`Optimizer.update` to update a model's parameters based on the loss
gradients and subsequently calling :func:`mlx.core.eval` to evaluate both the
model's parameters and the **optimizer state**.

.. code-block:: python

    # Create a model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    # Create the gradient function and the optimizer
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(model, grads)

            # Compute the new parameters but also the optimizer state.
            mx.eval(model.parameters(), optimizer.state)

.. currentmodule:: mlx.optimizers

.. autosummary::
   :toctree: _autosummary
   :template: optimizers-template.rst

   OptimizerState
   Optimizer
   SGD
   RMSprop
   Adagrad
   Adafactor
   AdaDelta
   Adam
   AdamW
   Adamax
   Lion
