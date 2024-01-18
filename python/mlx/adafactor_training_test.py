# Copyright © 2023 Apple Inc.

import gzip
import os
import pickle
from re import I
from urllib import request

import numpy as np
from tqdm import tqdm

def mnist(
    save_dir="/tmp", base_url="http://yann.lecun.com/exdb/mnist/", filename="mnist.pkl"
):
    """
    Load the MNIST dataset in 4 tensors: train images, train labels,
    test images, and test labels.

    Checks `save_dir` for already downloaded data otherwise downloads.

    Download code modified from:
      https://github.com/hsjeong5/MNIST-for-Numpy
    """

    def download_and_save(save_file):
        filename = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for name in filename:
            out_file = os.path.join("/tmp", name[1])
            request.urlretrieve(base_url + name[1], out_file)
        for name in filename[:2]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28 * 28
                )
        for name in filename[-2:]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(save_file, "wb") as f:
            pickle.dump(mnist, f)

    save_file = os.path.join(save_dir, filename)
    if not os.path.exists(save_file):
        download_and_save(save_file)
    with open(save_file, "rb") as f:
        mnist = pickle.load(f)

    def preproc(x):
        return x.astype(np.float32) / 255.0

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["test_images"] = preproc(mnist["test_images"])
    return (
        mnist["training_images"],
        mnist["training_labels"].astype(np.uint32),
        mnist["test_images"],
        mnist["test_labels"].astype(np.uint32),
    )


def fashion_mnist(save_dir="/tmp"):
    return mnist(
        save_dir,
        base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        filename="fashion_mnist.pkl",
    )


def train_using_adafactor(train_x, train_y, test_x, test_y):
    from mlx import nn
    from mlx.nn import losses
    import mlx.core as mx
    from optimizers import Adafactor
    import json

    mx.random.seed(42)

    model = nn.Sequential(nn.Linear(784, 512),
                          nn.GELU(),
                          nn.Linear(512, 512),
                          nn.GELU(),
                          nn.Linear(512, 128),
                          nn.GELU(),
                          nn.Linear(128, 10))

    optimizer = Adafactor(learning_rate=1e-5)

    def compute_loss(x, y):
        logits = model(x)
        loss = losses.cross_entropy(logits, y,
                                    reduction='mean')
        return loss

    compute_loss_and_grad = nn.utils.value_and_grad(model, compute_loss)

    epochs = 5
    num_steps_per_train_epoch = 100
    num_steps_per_test = 50
    batch_size = 64
    logs = {'total_loss_per_epoch': [], 'eval_accuracies': []}
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_accuracy = 0
        with tqdm(range(1, num_steps_per_train_epoch+1)) as iterator:
            for step in iterator:
                indices = np.random.randint(0, train_x.shape[0], (batch_size,))
                batch_x = mx.array(train_x[indices, ...])
                batch_y = mx.array(train_y[indices,])
                loss, grads = compute_loss_and_grad(batch_x, batch_y)
                updates = optimizer.apply_gradients(grads, model)
                model.update(updates)

                total_loss += loss * batch_x.shape[0]
                number_of_processed_batches = step*batch_size
                avg_loss = total_loss.item() / number_of_processed_batches
                iterator.set_postfix(epoch=f"{epoch}/{epochs}", avg_loss=avg_loss)
            logs['total_loss_per_epoch'].append(round(avg_loss, 4))

        with tqdm(range(1, num_steps_per_test+1)) as iterator:
            for val_step in iterator:
                model.eval()
                indices = np.random.randint(0, test_x.shape[0], (batch_size,))
                batch_x = mx.array(test_x[indices, ...])
                batch_y = mx.array(test_y[indices,])
                out = model(batch_x)
                preds = mx.argmax(out, axis=-1)
                total_accuracy += (preds == batch_y).sum()
                eval_accuracy = total_accuracy.item() / (batch_size * val_step)
                iterator.set_postfix(accuracy=eval_accuracy)
            logs['eval_accuracies'].append(round(eval_accuracy, 4))

    with open("logs.json", 'w') as f:
        json.dump(logs, f)

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = mnist()
    assert train_x.shape == (60000, 28 * 28), "Wrong training set size"
    assert train_y.shape == (60000,), "Wrong training set size"
    assert test_x.shape == (10000, 28 * 28), "Wrong test set size"
    assert test_y.shape == (10000,), "Wrong test set size"
    train_using_adafactor(train_x, train_y, test_x, test_y)