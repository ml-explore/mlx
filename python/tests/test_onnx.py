import unittest

import mlx.core as mx
import numpy as np
import onnx.backend.test
from mlx.onnx import MlxBackend


# need to conver to numpy for the testing suite
class TestMlxBackend(MlxBackend):
    def __init__(self, model):
        super().__init__(model)

    def run(self, inputs, **kwargs):
        t = super().run(inputs, **kwargs)
        return tuple(
            np.array(x) if isinstance(x, mx.array) else [np.array(i) for i in x]
            for x in t
        )


class TestMlxBackendWrapper:
    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str):
        return TestMlxBackend(model)

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device.lower() in ["cpu", "gpu"]


btest = onnx.backend.test.BackendTest(TestMlxBackendWrapper, __name__)

# btest.include("")

# TODO: these are upcasting to float32
btest.exclude("test_div_uint8_cpu")
btest.exclude("test_pow_types_int32_float32_cpu")
btest.exclude("test_pow_types_int64_float32_cpu")
btest.exclude("test_matmulinteger_*")
btest.exclude("test_clip_default_int8_min_cpu")

# TODO: Debug these errors
btest.exclude("test_clip_default_max_cpu")
btest.exclude("test_clip_default_inbounds_cpu")
btest.exclude("test_clip_default_int8_max_cpu")
btest.exclude("test_clip_default_int8_inbounds_cpu")
btest.exclude("test_reduce_min_empty_set_cpu")

# TODO: Implement
btest.exclude("test_pad_*")
btest.exclude("test_topk*")
btest.exclude("test_maxpool_*")
btest.exclude("test_maxunpool_*")
btest.exclude("test_batchnorm_*")
btest.exclude("test_instancenorm_*")
btest.exclude("test_gelu_tanh_*")
btest.exclude("test_bitwise_*")
btest.exclude("test_gathernd_*")


# TODO: need to go through and handle these better
btest.exclude("test_cast_*")
btest.exclude("test_castlike_*")
btest.exclude("test_argmax_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmax_negative_axis_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmax_no_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_no_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_negative_axis_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_keepdims_example_select_last_index_cpu")

# TODO: Reenable when float64 support is added back
btest.exclude("test_max_float64_cpu")
btest.exclude("test_min_float64_cpu")

# TODO: Graph tests
btest.exclude("test_range_float_type_positive_delta_expanded_cpu")
btest.exclude("test_range_int32_type_negative_delta_expanded_cpu")

# TODO: Add gradient support
btest.exclude("test_gradient_*")

# TODO: There is a bug in mlx round impl, -2.5 -> -3 instead of -2
btest.exclude("test_round_*")

# TODO: Investigate
btest.exclude("test_operator_pad_*")
btest.exclude("test_sequence_*")
btest.exclude("test_strnorm_*")
btest.exclude("test_bitshift_*")
btest.exclude("string")

# float64 datatype
btest.exclude("test_reduce_log_sum_exp_*")
btest.exclude("test_operator_addconstant_cpu")
btest.exclude("test_operator_add_size1_singleton_broadcast_cpu")
btest.exclude("test_operator_add_broadcast_cpu")
btest.exclude("test_operator_add_size1_broadcast_cpu")
btest.exclude("test_operator_add_size1_right_broadcast_cpu")
btest.exclude("test_cumsum_*")
btest.exclude("test_eyelike_with_dtype_cpu")

# skip models
for x in btest.test_suite:
    if "OnnxBackendRealModelTest" in str(type(x)):
        btest.exclude(str(x).split(" ")[0])

globals().update(btest.enable_report().test_cases)

if __name__ == "__main__":
    unittest.main()
