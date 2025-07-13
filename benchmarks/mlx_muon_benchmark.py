"""
MLX Muon Optimizer Benchmarks

This module provides benchmarking scripts for the Muon optimizer on vision and language modeling tasks,
following the MLX-Muon playbook validation criteria.
"""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.optimizers import AdamW, Muon

# Create results directory
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_memory() -> float:
    """Get peak memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024  # Convert to MB


class SimpleCIFAR10Model(nn.Module):
    """Simple ResNet-like model for CIFAR-10 benchmarking."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Simple CNN architecture inspired by ResNet-18 but smaller for quick benchmarking
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm(64)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class SimpleGPTModel(nn.Module):
    """Simplified GPT model for language modeling benchmarks."""

    def __init__(
        self, vocab_size=1000, d_model=256, n_heads=8, n_layers=4, block_size=128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)

        # Simple transformer blocks
        self.layers = []
        for _ in range(n_layers):
            self.layers.extend(
                [
                    nn.MultiHeadAttention(d_model, n_heads),
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model),
                    nn.LayerNorm(d_model),
                ]
            )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def __call__(self, x):
        seq_len = x.shape[1]
        pos = mx.arange(seq_len)

        # Token + positional embeddings
        x = self.embedding(x) + self.pos_embedding(pos)

        # Apply transformer blocks
        for i in range(0, len(self.layers), 6):
            attn = self.layers[i]
            ln1 = self.layers[i + 1]
            ff1 = self.layers[i + 2]
            relu = self.layers[i + 3]
            ff2 = self.layers[i + 4]
            ln2 = self.layers[i + 5]

            # Attention block with residual
            attn_out = attn(x)
            x = ln1(x + attn_out)

            # Feed-forward block with residual
            ff_out = ff2(relu(ff1(x)))
            x = ln2(x + ff_out)

        x = self.ln_f(x)
        return self.head(x)


def create_optimizer(
    optimizer_name: str, learning_rate: float, method: str = "auto"
) -> opt.Optimizer:
    """Create optimizer by name."""
    if optimizer_name.lower() == "muon":
        return Muon(
            learning_rate=learning_rate, momentum=0.95, ns_steps=5, method=method
        )
    elif optimizer_name.lower() == "adamw":
        return AdamW(learning_rate=learning_rate, betas=[0.9, 0.999], weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def generate_dummy_cifar_data(batch_size: int = 32, num_batches: int = 10):
    """Generate dummy CIFAR-10-like data."""
    for _ in range(num_batches):
        # CIFAR-10: 32x32x3 images, 10 classes
        x = mx.random.normal([batch_size, 3, 32, 32])
        y = mx.random.randint(0, 10, [batch_size])
        yield x, y


def generate_dummy_text_data(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 1000,
    num_batches: int = 10,
):
    """Generate dummy text data."""
    for _ in range(num_batches):
        # Random token sequences
        x = mx.random.randint(0, vocab_size, [batch_size, seq_len])
        # Shifted for language modeling (predict next token)
        y = mx.random.randint(0, vocab_size, [batch_size, seq_len])
        yield x, y


def benchmark_vision_cifar10(
    optimizer_name: str,
    learning_rate: float = 0.001,
    epochs: int = 1,
    method: str = "auto",
) -> Dict[str, Any]:
    """Benchmark vision model (ResNet-like) on CIFAR-10."""
    method_desc = f" (method={method})" if optimizer_name.lower() == "muon" else ""
    print(f"Benchmarking vision model with {optimizer_name}{method_desc} optimizer...")

    # Create model and optimizer
    model = SimpleCIFAR10Model()
    optimizer = create_optimizer(optimizer_name, learning_rate, method)

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Timing and metrics
    start_time = time.perf_counter()
    start_memory = measure_memory()
    total_images = 0
    final_loss = None
    optimizer_time = 0.0  # Track time spent in optimizer
    forward_backward_time = 0.0  # Track time in forward/backward pass

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(
            generate_dummy_cifar_data(batch_size=32, num_batches=50)
        ):
            # Time forward/backward pass
            fb_start = time.perf_counter()
            loss, grads = loss_and_grad_fn(model, x, y)
            fb_end = time.perf_counter()
            forward_backward_time += fb_end - fb_start

            # Time optimizer update
            opt_start = time.perf_counter()
            optimizer.update(model, grads)
            opt_end = time.perf_counter()
            optimizer_time += opt_end - opt_start

            total_images += x.shape[0]
            final_loss = loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    end_time = time.perf_counter()
    peak_memory = measure_memory()

    # Calculate metrics
    wall_clock_time = end_time - start_time
    images_per_sec = total_images / wall_clock_time
    memory_usage = peak_memory - start_memory
    optimizer_overhead_pct = (optimizer_time / wall_clock_time) * 100

    return {
        "optimizer": optimizer_name,
        "method": method if optimizer_name.lower() == "muon" else "n/a",
        "task": "vision_cifar10",
        "wall_clock_time": wall_clock_time,
        "optimizer_time": optimizer_time,
        "forward_backward_time": forward_backward_time,
        "optimizer_overhead_percent": optimizer_overhead_pct,
        "images_per_sec": images_per_sec,
        "final_loss": final_loss,
        "memory_usage_mb": memory_usage,
        "total_images": total_images,
    }


def benchmark_lm_gpt(
    optimizer_name: str,
    learning_rate: float = 0.0003,
    epochs: int = 1,
    method: str = "auto",
) -> Dict[str, Any]:
    """Benchmark language model (GPT-like) on dummy text data."""
    method_desc = f" (method={method})" if optimizer_name.lower() == "muon" else ""
    print(
        f"Benchmarking language model with {optimizer_name}{method_desc} optimizer..."
    )

    # Create model and optimizer
    model = SimpleGPTModel(vocab_size=1000, d_model=256, n_heads=8, n_layers=4)
    optimizer = create_optimizer(optimizer_name, learning_rate, method)

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Timing and metrics
    start_time = time.perf_counter()
    start_memory = measure_memory()
    total_tokens = 0
    final_loss = None
    optimizer_time = 0.0  # Track time spent in optimizer
    forward_backward_time = 0.0  # Track time in forward/backward pass

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(
            generate_dummy_text_data(batch_size=8, seq_len=128, num_batches=50)
        ):
            # Time forward/backward pass
            fb_start = time.perf_counter()
            loss, grads = loss_and_grad_fn(model, x, y)
            fb_end = time.perf_counter()
            forward_backward_time += fb_end - fb_start

            # Time optimizer update
            opt_start = time.perf_counter()
            optimizer.update(model, grads)
            opt_end = time.perf_counter()
            optimizer_time += opt_end - opt_start

            total_tokens += x.shape[0] * x.shape[1]
            final_loss = loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    end_time = time.perf_counter()
    peak_memory = measure_memory()

    # Calculate metrics
    wall_clock_time = end_time - start_time
    tokens_per_sec = total_tokens / wall_clock_time
    memory_usage = peak_memory - start_memory
    optimizer_overhead_pct = (optimizer_time / wall_clock_time) * 100

    return {
        "optimizer": optimizer_name,
        "method": method if optimizer_name.lower() == "muon" else "n/a",
        "task": "lm_gpt",
        "wall_clock_time": wall_clock_time,
        "optimizer_time": optimizer_time,
        "forward_backward_time": forward_backward_time,
        "optimizer_overhead_percent": optimizer_overhead_pct,
        "tokens_per_sec": tokens_per_sec,
        "final_loss": final_loss,
        "memory_usage_mb": memory_usage,
        "total_tokens": total_tokens,
    }


def save_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to JSON file."""
    results_file = RESULTS_DIR / filename
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


def compare_optimizers():
    """Compare Muon and AdamW optimizers on both tasks."""
    print("=" * 60)
    print("MLX Muon Optimizer Benchmark Suite")
    print("=" * 60)

    all_results = []

    # Vision benchmarks
    print("\n🖼️  VISION BENCHMARKS (CIFAR-10-like)")
    print("-" * 40)

    for optimizer_name in ["adamw", "muon"]:
        lr = (
            0.001 if optimizer_name == "adamw" else 0.02
        )  # Adjusted LRs per Muon recommendations
        results = benchmark_vision_cifar10(optimizer_name, learning_rate=lr, epochs=1)
        all_results.append(results)
        save_results(results, f"vision_{optimizer_name}_results.json")

        print(f"\n{optimizer_name.upper()} Results:")
        print(f"  Time: {results['wall_clock_time']:.2f}s")
        print(f"  Speed: {results['images_per_sec']:.1f} images/sec")
        print(f"  Final Loss: {results['final_loss']:.4f}")
        print(f"  Memory: {results['memory_usage_mb']:.1f} MB")

    # Language model benchmarks
    print("\n📝 LANGUAGE MODEL BENCHMARKS (GPT-like)")
    print("-" * 40)

    for optimizer_name in ["adamw", "muon"]:
        lr = (
            0.0003 if optimizer_name == "adamw" else 0.005
        )  # Adjusted LRs per Muon recommendations
        results = benchmark_lm_gpt(optimizer_name, learning_rate=lr, epochs=1)
        all_results.append(results)
        save_results(results, f"lm_{optimizer_name}_results.json")

        print(f"\n{optimizer_name.upper()} Results:")
        print(f"  Time: {results['wall_clock_time']:.2f}s")
        print(f"  Speed: {results['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Final Loss: {results['final_loss']:.4f}")
        print(f"  Memory: {results['memory_usage_mb']:.1f} MB")

    # Save combined results
    save_results(all_results, "combined_benchmark_results.json")

    # Print comparison summary
    print("\n📊 COMPARISON SUMMARY")
    print("-" * 40)

    vision_adamw = next(
        r
        for r in all_results
        if r["task"] == "vision_cifar10" and r["optimizer"] == "adamw"
    )
    vision_muon = next(
        r
        for r in all_results
        if r["task"] == "vision_cifar10" and r["optimizer"] == "muon"
    )
    lm_adamw = next(
        r for r in all_results if r["task"] == "lm_gpt" and r["optimizer"] == "adamw"
    )
    lm_muon = next(
        r for r in all_results if r["task"] == "lm_gpt" and r["optimizer"] == "muon"
    )

    vision_speedup = vision_muon["images_per_sec"] / vision_adamw["images_per_sec"]
    lm_speedup = lm_muon["tokens_per_sec"] / lm_adamw["tokens_per_sec"]

    print(f"Vision (CIFAR-10) Speedup: {vision_speedup:.2f}x")
    print(f"Language Model Speedup: {lm_speedup:.2f}x")

    # Validation criteria check
    print(f"\n✅ VALIDATION CRITERIA CHECK")
    print("-" * 40)

    # Speed criteria
    vision_speed_pass = vision_speedup >= 1.25  # ≥25% improvement for vision
    lm_speed_pass = lm_speedup >= 1.5  # ≥1.5x improvement for LM
    print(
        f"Vision Speed (≥1.25x): {'✅ PASS' if vision_speed_pass else '❌ FAIL'} ({vision_speedup:.2f}x)"
    )
    print(
        f"LM Speed (≥1.5x): {'✅ PASS' if lm_speed_pass else '❌ FAIL'} ({lm_speedup:.2f}x)"
    )

    # Memory criteria (≤10% extra)
    vision_mem_ratio = vision_muon["memory_usage_mb"] / max(
        vision_adamw["memory_usage_mb"], 1
    )
    lm_mem_ratio = lm_muon["memory_usage_mb"] / max(lm_adamw["memory_usage_mb"], 1)
    vision_mem_pass = vision_mem_ratio <= 1.1
    lm_mem_pass = lm_mem_ratio <= 1.1
    print(
        f"Vision Memory (≤1.1x): {'✅ PASS' if vision_mem_pass else '❌ FAIL'} ({vision_mem_ratio:.2f}x)"
    )
    print(
        f"LM Memory (≤1.1x): {'✅ PASS' if lm_mem_pass else '❌ FAIL'} ({lm_mem_ratio:.2f}x)"
    )

    all_pass = vision_speed_pass and lm_speed_pass and vision_mem_pass and lm_mem_pass
    print(
        f"\n🎯 Overall: {'✅ ALL CRITERIA PASSED' if all_pass else '❌ SOME CRITERIA FAILED'}"
    )

    return all_results


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="MLX Muon Optimizer Benchmarks")
    parser.add_argument(
        "--optimizer",
        choices=["muon", "adamw"],
        default="muon",
        help="Optimizer to benchmark",
    )
    parser.add_argument(
        "--task",
        choices=["vision", "lm", "both"],
        default="both",
        help="Task to benchmark",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare Muon vs AdamW on both tasks"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "cubic", "quintic"],
        default="auto",
        help="Newton-Schulz method for Muon optimizer (ignored for AdamW)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_optimizers()
    else:
        if args.task in ["vision", "both"]:
            lr = 0.001 if args.optimizer == "adamw" else 0.02
            results = benchmark_vision_cifar10(
                args.optimizer, learning_rate=lr, method=args.method
            )
            filename = (
                f"vision_{args.optimizer}_{args.method}_results.json"
                if args.optimizer == "muon"
                else f"vision_{args.optimizer}_results.json"
            )
            save_results(results, filename)

        if args.task in ["lm", "both"]:
            lr = 0.0003 if args.optimizer == "adamw" else 0.005
            results = benchmark_lm_gpt(
                args.optimizer, learning_rate=lr, method=args.method
            )
            filename = (
                f"lm_{args.optimizer}_{args.method}_results.json"
                if args.optimizer == "muon"
                else f"lm_{args.optimizer}_results.json"
            )
            save_results(results, filename)


if __name__ == "__main__":
    main()
