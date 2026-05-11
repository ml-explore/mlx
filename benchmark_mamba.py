import mlx.core as mx
import time


def run_sweep():
    B = 4
    H = 16
    D_qk = 128
    D_v = 64

    seq_lengths = [2048, 4096, 8192, 16384, 32768]
    iters = 10

    print("=" * 80)
    print(
        f"{'Seq Length':<12} | {'SDPA (ms)':<10} | {'SDPA VRAM':<12} | {'Mamba (ms)':<12} | {'Mamba VRAM':<12}"
    )
    print("=" * 80)

    for L in seq_lengths:
        q = mx.random.normal((B, L, H, D_qk)).astype(mx.float32)
        k = mx.random.normal((B, L, H, D_qk)).astype(mx.float32)
        v = mx.random.normal((B, L, H, D_v)).astype(mx.float32)
        dt = mx.random.normal((B, H, L)).astype(mx.float32)
        trap = mx.random.normal((B, H, L)).astype(mx.float32)
        angles = mx.random.normal((B, L, H, D_qk // 2)).astype(mx.float32)

        def run_sdpa():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)

        def run_mamba():
            return mx.fast.mamba3_ssd(q, k, v, dt, trap, angles)

        # Warmup and clear cache
        mx.eval(run_sdpa(), run_mamba())
        mx.metal.clear_cache()

        # Bench SDPA Memory & Time
        mx.metal.reset_peak_memory()
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(run_sdpa())
        sdpa_time = ((time.perf_counter() - start) / iters) * 1000
        sdpa_vram = mx.metal.get_peak_memory() / (1024**2)

        mx.metal.clear_cache()

        # Bench Mamba Memory & Time
        mx.metal.reset_peak_memory()
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(run_mamba())
        mamba_time = ((time.perf_counter() - start) / iters) * 1000
        mamba_vram = mx.metal.get_peak_memory() / (1024**2)

        print(
            f"{L:<12} | {sdpa_time:<10.2f} | {sdpa_vram:<10.2f} MB | {mamba_time:<12.2f} | {mamba_vram:<10.2f} MB"
        )

    print("-" * 80)


if __name__ == "__main__":
    run_sweep()
