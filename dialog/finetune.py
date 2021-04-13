import argparse
import itertools
import pprint
from pathlib import Path

import wandb
from spacy.lang.en import English
from spacy.language import Language
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .data import ConversationDataset, contextualize


def read_file(path, encoding):
    with open(path, encoding=encoding) as f:
        return f.read()


def read_file_try_encodings(path, encodings=("utf8", "windows-1252")):
    exceptions = []
    for encoding in encodings:
        try:
            return read_file(path, encoding)
        except UnicodeDecodeError as exc:
            exceptions.append(exc)
    raise ValueError(
        f"Reading file {path} with encodings {encodings} failed: {exceptions}"
    )


def create_nlp():
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    return nlp


def extract_sentences(text, nlp: Language):
    document = nlp(text)
    return [sentence.text.strip() for sentence in document.sents]


def read_text_dir(text_dir, nlp: Language):
    text_file_paths = list(Path(text_dir).rglob("*.txt"))

    print("Using text files:")
    for path in text_file_paths:
        print(f"=> {path}")
    print()

    texts = [read_file_try_encodings(path) for path in text_file_paths]
    sentences = [extract_sentences(text, nlp) for text in texts]
    return sentences


def train(
    *,
    text_dir: str,
    save_dir: str,
    num_context: int = 7,
    batch_size: int = 2,
    save_steps: int = 500,
    pretrained_model: str = "microsoft/DialoGPT-small",
    extra_trainer_args=None,
):

    nlp = create_nlp()
    documents = read_text_dir(text_dir, nlp)

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
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
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
