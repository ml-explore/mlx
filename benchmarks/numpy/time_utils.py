import time


def time_fn(fn, *args):
    print(f"Timing {fn.__name__} ...", end=" ")

    # warmup
    for _ in range(5):
        fn(*args)

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = fn(*args)
    toc = time.perf_counter()

    msec = 1e3 * (toc - tic) / num_iters
    print(f"{msec:.5f} msec")
