#!/usr/bin/env python3
import argparse
import collections
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import tokenize_context, to_torch


def load_model(checkpoint_dir: Path):
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir)


def load_tokenizer(checkpoint_dir: Path):
    return AutoTokenizer.from_pretrained(
        checkpoint_dir,
        pad_token="[PAD]",  # TODO should we set this?
    )


def generate_answer(
    model,
    tokenizer,
    device: torch.device,
    context: List[str],
    top_k: int = 50,
    top_p: float = 0.95,
    **kwargs,
) -> str:

    input_ids = to_torch([tokenize_context(context, tokenizer)]).to(device)

    output_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        **kwargs,
    )
    end_of_context = input_ids.shape[-1]
    return tokenizer.decode(
        output_ids[:, end_of_context:][0],
        skip_special_tokens=True,
    )


def generate(
    *,
    model,
    tokenizer,
    device: torch.device,
    prefix: str,
    steps: int = 10,
    num_context: int = 7,
):

    print(f"PREFIX: {prefix}")
    context = collections.deque([prefix], maxlen=num_context)

    for step in range(steps):

        answer = generate_answer(
            model,
            tokenizer,
            device,
            context,
        )
        print(answer)

        context.append(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--num-context", default=7, type=int)
    parser.add_argument("--steps", default=10, type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_dir).to(device)
    tokenizer = load_tokenizer(args.checkpoint_dir)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prefix=args.prefix,
        device=device,
        steps=args.steps,
        num_context=args.num_context,
    )


if __name__ == "__main__":
    main()
