# Copyright © 2023 Apple Inc.

import time

import mlx.core as mx

lr = 0.01
num_examples = 1_000
num_features = 100
num_iters = 10_000
tic = time.time()
toc = time.time()


def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))


grad_fn = mx.grad(loss_fn)
# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))
# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
loss = loss_fn(w)
throughput = num_iters / (toc - tic)
# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,))
# True parameters
w_star = mx.random.normal((num_features,))
y = X @ w_star + eps
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)

print(
    f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)
