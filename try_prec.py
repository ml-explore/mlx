import mlx.core as mx


print(mx.array([1.23456789]))
with mx.printoptions(precision=4):
    print(mx.array([1.23456789]))


mx.set_printoptions(precision=2)
print(mx.array([1.23456789]))
