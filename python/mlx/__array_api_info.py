class ArrayNamespaceInfo:
    def __init__(self):
        pass

    def capabilities(self):
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": None,
        }

    def default_device(self):
        import mlx.core as mx

        return mx.default_device()

    def default_dtypes(self, *, device=None):
        import mlx.core as mx

        if device is not None and not isinstance(device, mx.Device):
            raise TypeError("Expected a mlx Device")
        device = device if device is not None else self.default_device()
        if device.type == mx.gpu:
            return {
                "real floating": mx.float32,
                "complex floating": mx.complex64,
                "integral": mx.int32,
                "indexing": mx.uint32,
            }
        return {
            "real floating": mx.float64,
            "complex floating": mx.complex128,
            "integral": mx.int64,
            "indexing": mx.uint64,
        }

    def devices(self):
        import mlx.core as mx

        devices = [
            mx.Device(dev_type, i)
            for dev_type in (mx.cpu, mx.gpu)
            for i in range(mx.device_count(dev_type))
        ]
        return tuple(devices)

    def dtypes(self, *, device=None, kind=None):
        pass


def __array_namespace_info__():
    return ArrayNamespaceInfo()
