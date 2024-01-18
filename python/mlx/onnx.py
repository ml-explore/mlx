import importlib
from typing import Any, Tuple

import mlx.core as mx
import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype

onnx_ops = importlib.import_module("mlx.onnx_ops")


class MlxBackend:
    def __init__(self, model: onnx.ModelProto):
        self._model = model
        self._cache = {}
        self.initializer_arrays()

    def initializer_arrays(self):
        for i in self._model.graph.initializer:
            if i.name in self._cache:
                continue
            self._cache[i.name] = self.parse_array(i)

    def parse_array(self, inp: onnx.TensorProto) -> mx.array:
        if inp.data_type == onnx.TensorProto.FLOAT and len(inp.float_data) > 0:
            return mx.array(
                np.array(inp.float_data, dtype=np.float32).reshape(inp.dims),
                dtype=mx.float32,
            )
        elif inp.data_type == onnx.TensorProto.INT32 and len(inp.int32_data) > 0:
            return mx.array(
                np.array(inp.int32_data, dtype=np.int32).reshape(inp.dims),
                dtype=mx.int32,
            )
        elif inp.data_type == onnx.TensorProto.INT64 and len(inp.int64_data) > 0:
            return mx.array(
                np.array(inp.int64_data, dtype=np.int64).reshape(inp.dims),
                dtype=mx.int64,
            )
        elif len(inp.raw_data) > 0:
            return mx.array(
                np.frombuffer(
                    inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)
                ).reshape(inp.dims)
            )
        else:
            raise NotImplementedError(
                f"Not implemented for {inp.data_type} {inp.name} {inp.dims}"
            )
        return mx.ones(inp.dims, dtype=mx.float32)

    def get_input_dict(self, inputs):
        input_names = [x.name for x in self._model.graph.input]
        init_names = set([x.name for x in self._model.graph.initializer])
        real_inputs = [x for x in input_names if x not in init_names]
        return dict(zip(real_inputs, inputs))

    def parse_attributes(self, attrs):
        res = {}
        for x in attrs:
            if x.type == onnx.AttributeProto.FLOAT:
                res[x.name] = float(x.f)
            elif x.type == onnx.AttributeProto.INT:
                res[x.name] = int(x.i)
            elif x.type == onnx.AttributeProto.STRING:
                res[x.name] = str(x.s)
            elif x.type == onnx.AttributeProto.TENSOR:
                res[x.name] = self.parse_array(x.t)
            elif x.type == onnx.AttributeProto.FLOATS:
                res[x.name] = tuple(float(f) for f in x.floats)
            elif x.type == onnx.AttributeProto.INTS:
                res[x.name] = tuple(int(i) for i in x.ints)
            elif x.type == onnx.AttributeProto.STRINGS:
                res[x.name] = tuple(str(s) for s in x.strings)
            elif x.type == onnx.AttributeProto.GRAPH:
                raise NotImplementedError(f"Attribute type graph not implemented")
            else:
                raise NotImplementedError(f"Attribute type {x.type} not implemented")
        return res

    def run(self, inputs, **kwargs: Any) -> Tuple[mx.array, ...]:
        self.initializer_arrays()
        inputs = self.get_input_dict(inputs)
        for i in self._model.graph.input:
            if i.name in self._cache:
                continue
            if i.name in inputs:
                if isinstance(inputs[i.name], mx.array):
                    self._cache[i.name] = inputs[i.name]
                elif isinstance(inputs[i.name], list):
                    self._cache[i.name] = [mx.array(x) for x in inputs[i.name]]
                elif isinstance(inputs[i.name], np.ndarray):
                    self._cache[i.name] = mx.array(inputs[i.name])
                else:
                    raise NotImplementedError(
                        f"Input type {type(inputs[i.name])} not implemented"
                    )
        for i, node in enumerate(self._model.graph.node):
            args = [self._cache[x] for x in node.input]
            opt = self.parse_attributes(node.attribute)

            if hasattr(onnx_ops, node.op_type):
                res = getattr(onnx_ops, node.op_type)(*args, **opt)
            else:
                raise NotImplementedError(f"Operation {node.op_type} not implemented")

            if not isinstance(res, tuple):
                res = (res,)

            for i in range(len(node.output)):
                self._cache[node.output[i]] = res[i]
        return tuple(self._cache[out.name] for out in self._model.graph.output)
