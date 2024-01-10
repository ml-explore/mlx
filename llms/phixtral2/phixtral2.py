import glob
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

def moe_factory() -> dict:
    return {
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }

@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32
    moe: dict = field(default_factory=moe_factory) 

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, rotary_dim: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(rotary_dim, traditional=False)
        self.Wqkv = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask=None, cache=None):
        qkv = self.Wqkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))

class MOE(nn.Module):
    def __init__(self, args: ModelArgs, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]
        self.mlp = [MLP(self.dim, self.hidden_dim) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.model_dim, self.num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)
        inds = mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.concatenate([self.mlp[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt[None, :])
        yc = mx.concatenate(y)

        return yc.reshape(orig_shape)


class ParallelBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.mixer = RoPEAttention(dims, config.num_heads, config.rotary_dim)
        self.ln = LayerNorm(dims)
        self.moe = MOE(config, dims, mlp_dims)

    def __call__(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.mixer(h, mask, cache)
        ff_h = self.moe(h)
        return attn_h + ff_h + x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embd = Embd(config)
        self.h = [ParallelBlock(config) for i in range(config.num_layers)]

    def __call__(self, x, mask, cache):
        x = self.embd(x)
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])
        return x, cache


class Embd(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)

    def __call__(self, x):
        return self.wte(x)


class OutputHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.ln = LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs):
        return self.linear(self.ln(inputs))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.transformer = TransformerDecoder(config)
        self.lm_head = OutputHead(config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        return self.lm_head(y), cache


def generate(prompt: mx.array, model: Model, temp: float = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def load(path_or_hf_repo: str):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)
        model_args = ModelArgs.from_dict(config)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model = Model(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    return model, tokenizer

# debugging
if __name__ == "__main__":
    path_or_hf_repo = "mlabonne/phixtral-4x2_8"
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )
    print(f"{model_path=}")

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)
        model_args = ModelArgs.from_dict(config)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model = Model(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
