#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(checkpoint_dir: Path):
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir)


def load_tokenizer(checkpoint_dir: Path):
    return AutoTokenizer.from_pretrained(
        checkpoint_dir,
        pad_token="[PAD]",  # TODO should we set this?
    )


def generate(
    *,
    model,
    tokenizer,
    prefix,
    device,
    steps: int = 10,
    num_context: int = 7,
):

    input_ids = tokenizer.encode(prefix + tokenizer.eos_token, return_tensors="pt").to(
        device
    )

    print(f"PREFIX: {prefix}")

    for step in range(steps):

        output_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

        print(
            tokenizer.decode(
                output_ids[:, input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )

        input_ids = output_ids


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
