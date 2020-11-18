import itertools

from typing import List

import tqdm
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer


from data_collator import DataCollatorForLanguageModeling


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small",
        pad_token="[PAD]",
    )

    contexted = []

    num_context = 7

    with open("rick_morty_test_data.txt") as f:
        lines = [line.strip() for line in f]

    for conv_start_index, _ in enumerate(lines[num_context:]):
        sentences = lines[conv_start_index : conv_start_index + num_context]
        contexted.append(sentences)

    num_samples = len(contexted)

    print("EXAMPLE WITH CONTEXT")
    print(contexted[0])
    print()

    def construct_conv(lines, tokenizer, eos=True):
        conv = [tokenizer.encode(line) + [tokenizer.eos_token_id] for line in lines]
        flat = list(itertools.chain.from_iterable(conv))

        # TODO Do we really need to pad and truncate here?
        seq_len = 512
        flat = flat[:seq_len]
        return tokenizer.pad(
            {"input_ids": flat}, padding="max_length", max_length=seq_len
        )["input_ids"]

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

    split_at = int(num_samples * 0.8)
    train_split, val_split = contexted[:split_at], contexted[split_at:]

    train_dataset = ConversationDataset(tokenizer, train_split)
    val_dataset = ConversationDataset(tokenizer, val_split)

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
    )

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
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
    )

    trainer.train()
    print("trained")
    trainer.evaluate()
    print("evaluated")


if __name__ == "__main__":
    main()
