# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
import phixtral2
import transformers


def generate(
    model: phixtral2.Model,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
):
    print("[INFO] Generating with Phi-2...", flush=True)
    print(prompt, end="", flush=True)
    prompt = tokenizer(
        prompt,
        return_tensors="np",
        return_attention_mask=False,
    )[
        "input_ids"
    ][0]
    prompt = mx.array(prompt)

    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        phixtral2.generate(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()

        tokens.append(token.item())
        # if (n + 1) % 10 == 0:
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    prompt_tps = prompt.size / prompt_time
    gen_tps = (len(tokens) - 1) / gen_time
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="Write a detailed analogy between mathematics and a lighthouse.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()
    mx.random.seed(args.seed)
    model, tokenizer = phixtral2.load(args.model)
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temp)
