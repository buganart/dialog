#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(*, prefix, num_context, checkpoint_dir, steps):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small",
        pad_token="[PAD]",  # TODO should we set this?
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

    input_ids = tokenizer.encode(prefix + tokenizer.eos_token, return_tensors="pt")

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
    return generate(**vars(args))


if __name__ == "__main__":
    main()
