class ArrayNamespaceInfo:
    def capabilities(self):
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": 10,
        }

    def default_device(self):
        import mlx.core as mx

        return mx.default_device()

    def default_dtypes(self, *, device=None):
        import mlx.core as mx

        if device is not None and not isinstance(device, mx.Device):
            raise TypeError("Expected a mlx Device")
        return {
            "real floating": mx.float32,
            "complex floating": mx.complex64,
            "integral": mx.int32,
            "indexing": mx.int32,
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
        import mlx.core as mx

        if device is not None and not isinstance(device, mx.Device):
            raise TypeError("Expected a mlx Device")
        device = device if device is not None else self.default_device()

        dtypes = {
            "bool": mx.bool_,
            "int8": mx.int8,
            "int16": mx.int16,
            "int32": mx.int32,
            "int64": mx.int64,
            "uint8": mx.uint8,
            "uint16": mx.uint16,
            "uint32": mx.uint32,
            "uint64": mx.uint64,
            "float32": mx.float32,
            "complex64": mx.complex64,
        }
        if device.type == mx.cpu:
            dtypes["float64"] = mx.float64
        if kind is None:
            return dtypes

        signed = {"int8", "int16", "int32", "int64"}
        unsigned = {"uint8", "uint16", "uint32", "uint64"}
        real = {"float32", "float64"}
        complex_ = {"complex64"}
        kinds = {
            "bool": {"bool"},
            "signed integer": signed,
            "unsigned integer": unsigned,
            "integral": signed | unsigned,
            "real floating": real,
            "complex floating": complex_,
            "numeric": signed | unsigned | real | complex_,
        }
        kind = (kind,) if isinstance(kind, str) else kind
        if not isinstance(kind, tuple) or any(k not in kinds for k in kind):
            raise ValueError(f"Unsupported dtype kind: {kind!r}")
        names = {name for k in kind for name in kinds[k]}
        return {name: dtype for name, dtype in dtypes.items() if name in names}


def __array_namespace_info__():
    return ArrayNamespaceInfo()
