import mlx.core as mx
import mlx.core.distributed as dist
import sys

def test_split_same_color():
    g = dist.init()
    sub = g.split(color=0)
    assert sub.size() == g.size(), f"Expected same size, got {sub.size()} vs {g.size()}"
    assert sub.rank() == g.rank(), f"Expected same rank, got {sub.rank()} vs {g.rank()}"
    x = mx.ones(10) * (sub.rank() + 1)
    result = dist.all_sum(x, group=sub)
    mx.eval(result)
    expected = sum(range(1, sub.size() + 1))
    assert mx.allclose(result, mx.ones(10) * expected).item()
    print(f"  [rank {g.rank()}] same-color split: OK (sub size={sub.size()})")

def test_split_two_groups():
    g = dist.init()
    if g.size() < 2: return
    color = g.rank() % 2
    sub = g.split(color=color)
    expected_size = (g.size() + 1 - color) // 2
    assert sub.size() == expected_size
    x = mx.ones(10) * (sub.rank() + 1)
    result = dist.all_sum(x, group=sub)
    mx.eval(result)
    expected = sum(range(1, sub.size() + 1))
    assert mx.allclose(result, mx.ones(10) * expected).item()
    print(f"  [rank {g.rank()}] two-group split: OK (color={color}, sub rank={sub.rank()}, sub size={sub.size()})")

def test_split_three_groups():
    g = dist.init()
    if g.size() < 3: return
    color = g.rank() % 3
    sub = g.split(color=color)
    expected_size = (g.size() - color + 2) // 3
    assert sub.size() == expected_size
    x = mx.ones(10) * (sub.rank() + 1)
    result = dist.all_sum(x, group=sub)
    mx.eval(result)
    expected = sum(range(1, sub.size() + 1))
    assert mx.allclose(result, mx.ones(10) * expected).item()
    print(f"  [rank {g.rank()}] three-group split: OK (color={color}, sub rank={sub.rank()}, sub size={sub.size()})")

def test_split_with_key():
    g = dist.init()
    if g.size() < 2: return
    sub = g.split(color=0, key=g.size() - 1 - g.rank())
    assert sub.size() == g.size()
    expected_rank = g.size() - 1 - g.rank()
    assert sub.rank() == expected_rank
    print(f"  [rank {g.rank()}] key-reversed split: OK (parent rank={g.rank()} → sub rank={sub.rank()})")

def test_split_send_recv():
    g = dist.init()
    if g.size() < 2: return
    sub = g.split(color=0)
    if sub.rank() == 0:
        x = mx.array([42.0, 43.0, 44.0])
        try:
            dist.send(x, dst=1, group=sub)
            mx.eval(x)
            print(f"  [rank {g.rank()}] send/recv: sent to sub rank 1")
        except RuntimeError as e:
            if "does not support" in str(e):
                print(f"  [rank {g.rank()}] skip send/recv: not supported by sub-group backend")
            else:
                raise
    elif sub.rank() == 1:
        try:
            y = dist.recv_like(mx.zeros(3), src=0, group=sub)
            mx.eval(y)
            assert mx.allclose(y, mx.array([42.0, 43.0, 44.0])).item()
            print(f"  [rank {g.rank()}] send/recv: received OK from sub rank 0")
        except RuntimeError as e:
            if "does not support" in str(e):
                pass
            else:
                raise
    else:
        print(f"  [rank {g.rank()}] send/recv: not participating (sub rank {sub.rank()})")

if __name__ == "__main__":
    g = dist.init()
    print(f"[rank {g.rank()}/{g.size()}] JACCL split() test starting")
    tests = [
        ("same-color split", test_split_same_color),
        ("two-group split", test_split_two_groups),
        ("three-group split", test_split_three_groups),
        ("key-reversed split", test_split_with_key),
        ("send/recv on sub-group", test_split_send_recv),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [rank {g.rank()}] FAIL {name}: {e}")
            failed += 1
    dist.all_sum(mx.ones(1), group=g)
    if g.rank() == 0:
        print(f"\nResults: {passed}/{passed+failed} tests passed")
        if failed > 0: sys.exit(1)
