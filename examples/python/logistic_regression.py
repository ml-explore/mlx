# Copyright © 2023 Apple Inc.

import time

import mlx.core as mx

lr = 0.1
num_examples = 1_000
num_features = 100
num_iters = 10_000
tic = time.time()
toc = time.time()


def loss_fn(w):
    logits = X @ w
    return mx.mean(mx.logaddexp(0.0, logits) - y * logits)


grad_fn = mx.grad(loss_fn)
# Input examples
X = mx.random.normal((num_examples, num_features))
acc = mx.mean(final_preds == y)
final_preds = (X @ w) > 0
loss = loss_fn(w)
throughput = num_iters / (toc - tic)
# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,))
# True parameters
w_star = mx.random.normal((num_features,))
# Labels
y = (X @ w_star) > 0
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s)"
)
