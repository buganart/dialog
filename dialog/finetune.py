import argparse
import itertools
import pprint
import warnings
from pathlib import Path

import torch
import tqdm
import wandb
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def construct_conv(lines, tokenizer, eos=True):
    conv = [tokenizer.encode(line) + [tokenizer.eos_token_id] for line in lines]
    return list(itertools.chain.from_iterable(conv))


def contextualize(lines, num_context):
    return [
        lines[conv_start_index : conv_start_index + num_context]
        for conv_start_index, _ in enumerate(lines[num_context:])
    ]


def read_lines_warn_on_error(path):
    try:
        return read_lines(path)
    except UnicodeDecodeError as exc:

        warnings.warn(
            f"Errors when reading file {path}: {exc}\n"
            "Will replace offending characters."
        )

        return read_lines(path, errors="replace")


def read_lines(path, errors=None):
    with open(path, errors=errors) as f:
        return [line.strip() for line in f]


def read_text_dir(text_dir):
    text_file_paths = list(Path(text_dir).rglob("*.txt"))

    print("Using text files:")
    for path in text_file_paths:
        print(f"=> {path}")
    print()

    return [read_lines_warn_on_error(path) for path in text_file_paths]


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, contexted):

        self.examples = []
        for lines in tqdm.tqdm(contexted, desc="Tokenizing"):
            conv = construct_conv(lines, tokenizer)
            self.examples.append(conv)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def train(
    *,
    text_dir: str,
    save_dir: str,
    num_context: int = 7,
    save_steps: int = 500,
    pretrained_model: str = "microsoft/DialoGPT-small",
    extra_trainer_args=None,
):

    documents = read_text_dir(text_dir)

    contexted = list(
        itertools.chain.from_iterable(
            contextualize(lines, num_context) for lines in documents
        )
    )

    num_samples = len(contexted)

    print("Example with context:")
    for line in contexted[0]:
        print(f"=> {line}")
    print()

    split_at = int(num_samples * 0.8)
    train_split, val_split = contexted[:split_at], contexted[split_at:]

    wandb.init(
        dir=save_dir,
        reinit=True,
        resume=False,
        save_code=True,
    )

    default_trainer_args = dict(
        output_dir=wandb.run.dir,  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        save_steps=save_steps,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=10,
        run_name=None,
        load_best_model_at_end=True,
    )

    args = {**default_trainer_args, **(extra_trainer_args or {})}
    training_args = TrainingArguments(**args)

    print("TRAINING ARGS")
    pprint.pprint(training_args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model,
        pad_token="[PAD]",  # TODO should we set this?
    )

    train_dataset = ConversationDataset(tokenizer, train_split)
    val_dataset = ConversationDataset(tokenizer, val_split)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Finished training.")
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", required=True, type=Path)
    parser.add_argument("--num-context", default=7, type=int)
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--save-steps", default=500, type=int)
    args = parser.parse_args()
    return train(**vars(args))


if __name__ == "__main__":
    main()
