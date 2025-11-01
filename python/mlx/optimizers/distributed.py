import mlx.core as mx
from mlx.nn import Module
from mlx.nn.optimizers import Optimizer


class DistributedOptimizer(Optimizer):

    def update(self, model: Module, gradients: dict):

        gradient_sclice = mx.distributed.reduce_scatter(
            gradients
        )  # this is the shard that owned by this worker
        model_slice.update(
            self.apply_gradients(gradient_sclice, model_slice)
        )  # apply gradients only to the shard owned by this worker
        mx.distributed.all_gather(
            model.parameters
        )  # gather all the updated parameters from all workers
