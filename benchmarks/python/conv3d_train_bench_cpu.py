import time

import mlx.core as mx
import mlx.nn
import mlx.optimizers as opt
import torch


def bench_mlx(steps: int = 20, shape=(10, 32, 32, 32, 3)) -> float:
    mx.set_default_device(mx.DeviceType.cpu)

    class BenchNetMLX(mlx.nn.Module):
        # simple encoder-decoder net

        def __init__(self, in_channels, hidden_channels=16):
            super().__init__()

            self.net = mlx.nn.Sequential(
                mlx.nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
                mlx.nn.ReLU(),
                mlx.nn.Conv3d(
                    hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
                ),
                mlx.nn.ReLU(),
                mlx.nn.ConvTranspose3d(
                    2 * hidden_channels, hidden_channels, kernel_size=3, padding=1
                ),
                mlx.nn.ReLU(),
                mlx.nn.ConvTranspose3d(
                    hidden_channels, in_channels, kernel_size=3, padding=1
                ),
            )

        def __call__(self, input):
            return self.net(input)

    benchNet = BenchNetMLX(3)
    mx.eval(benchNet.parameters())
    optim = opt.Adam(learning_rate=1e-3)

    inputs = mx.random.normal(shape)

    params = benchNet.parameters()
    optim.init(params)

    state = [benchNet.state, optim.state]

    def loss_fn(params, image):
        benchNet.update(params)
        pred_image = benchNet(image)
        return (pred_image - image).abs().mean()

    def step(params, image):
        loss, grads = mx.value_and_grad(loss_fn)(params, image)
        optim.update(benchNet, grads)
        return loss

    total_time = 0.0
    print("MLX:")
    for i in range(steps):
        start_time = time.perf_counter()

        step(benchNet.parameters(), inputs)
        mx.eval(state)
        end_time = time.perf_counter()

        print(f"{i:3d}, time={(end_time-start_time) * 1000:7.2f} ms")
        total_time += (end_time - start_time) * 1000

    return total_time


def bench_torch(steps: int = 20, shape=(10, 3, 32, 32, 32)) -> float:
    device = torch.device("cpu")

    class BenchNetTorch(torch.nn.Module):
        # simple encoder-decoder net

        def __init__(self, in_channels, hidden_channels=16):
            super().__init__()

            self.net = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(
                    2 * hidden_channels, hidden_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(
                    hidden_channels, in_channels, kernel_size=3, padding=1
                ),
            )

        def forward(self, input):
            return self.net(input)

    benchNet = BenchNetTorch(3).to(device)
    optim = torch.optim.Adam(benchNet.parameters(), lr=1e-3)

    inputs = torch.randn(*shape, device=device)

    def loss_fn(pred_image, image):
        return (pred_image - image).abs().mean()

    total_time = 0.0
    print("PyTorch:")
    for i in range(steps):
        start_time = time.perf_counter()

        optim.zero_grad()
        pred_image = benchNet(inputs)
        loss = loss_fn(pred_image, inputs)
        loss.backward()
        optim.step()

        end_time = time.perf_counter()

        print(f"{i:3d}, time={(end_time-start_time) * 1000:7.2f} ms")
        total_time += (end_time - start_time) * 1000

    return total_time


def main():
    steps = 10
    time_mlx = bench_mlx(steps)
    time_torch = bench_torch(steps)

    print(f"average time of MLX:     {time_mlx/steps:9.2f} ms")
    print(f"total time of MLX:       {time_mlx:9.2f} ms")
    print(f"average time of PyTorch: {time_torch/steps:9.2f} ms")
    print(f"total time of PyTorch:   {time_torch:9.2f} ms")

    diff = time_torch / time_mlx - 1.0
    print(f"torch/mlx diff: {100. * diff:+5.2f}%")


if __name__ == "__main__":
    main()
